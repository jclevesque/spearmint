# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import optparse
import tempfile
import datetime
import multiprocessing
import importlib
import time
import imp
import os
import sys
import re
import signal
import socket

try: import simplejson as json
except ImportError: import json


# TODO: this shouldn't be necessary when the project is installed like a normal
# python lib.  For now though, this lets you symlink to supermint from your path and run it
# from anywhere.
sys.path.append(os.path.realpath(__file__))

from spearmint.ExperimentGrid  import *
from spearmint.helpers         import *
import spearmint.helpers as helpers
from spearmint.runner import job_runner
from spearmint import chooser

# Use a global for the web process so we can kill it cleanly on exit
web_proc = None

# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --run-job argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.


def parse_args():
    parser = optparse.OptionParser(usage="\n\tspearmint [options] <experiment/config.pb>")

    parser.add_option("--max-concurrent", dest="max_concurrent",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=10000)

    #New parameters to support my overly complex stuff
    parser.add_option("--jobs-per-node", dest="jobs_per_node",
                      type="int", default=1)
    parser.add_option("--nb-dist-nodes", dest="nb_dist_nodes",
                      type="int", default=1)
    parser.add_option("--nb-mini-batches", dest="nb_mini_batches",
                      type="int", default=0)
    parser.add_option("--distant-driver", dest="distant_driver",
                      type="string", default="")

    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments [SequentialChooser, RandomChooser, GPEIOptChooser, GPEIOptChooser, GPEIperSecChooser, GPEIChooser]",
                      type="string", default="GPEIOptChooser")
    parser.add_option("--driver", dest="driver",
                      help="Runtime driver for jobs (local, or sge)",
                      type="string", default="local")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=20000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--run-job", dest="job",
                      help="Run a job in wrapper mode.",
                      type="string", default="")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3.0)
    parser.add_option("-w", "--web-status", action="store_true",
                      help="Serve an experiment status web page.",
                      dest="web_status")
    parser.add_option("--port",
                      help="Specify a port to use for the status web interface.",
                      dest="web_status_port", type="int", default=0)
    parser.add_option("-v", "--verbose", action="store_true",
                      help="Print verbose debug output.")

    (options, args) = parser.parse_args()
    options.driver_params = {}

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    return options, args


def get_available_port(portnum):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', portnum))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_web_view(options, experiment_config, chooser):
    '''Start the web view in a separate process.'''

    from spearmint.web.app import app
    port = get_available_port(options.web_status_port)
    print("Using port: " + str(port))
    app.set_experiment_config(experiment_config)
    app.set_chooser(options.chooser_module,chooser)
    debug = (options.verbose == True)
    start_web_app = lambda: app.run(debug=debug, port=port)
    proc = multiprocessing.Process(target=start_web_app)
    proc.start()

    return proc


def main(options=None, experiment_config=None, expt_dir=None):
    #If nothing given, get arguments from sys.argv. Otherwise they are provided
    #by external caller.
    if options == None:
        (options, args) = parse_args()

        if options.job:
            job_runner(load_job(options.job))
            return 0

        experiment_config = args[0]
        expt_dir  = os.path.dirname(os.path.realpath(experiment_config))
    log("Using experiment configuration: " + str(experiment_config))
    log("experiment dir: " + expt_dir)

    if not os.path.exists(expt_dir):
        log("Cannot find experiment directory '%s'. "
            "Aborting." % (expt_dir))
        sys.exit(-1)

    check_experiment_dirs(expt_dir)

    # Load up the chooser module.
    module = load_module('chooser', options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    if options.web_status:
        web_proc = start_web_view(options, experiment_config, chooser)

    module = load_module('driver', options.driver)
    driver = module.init(run_func=options.run_func)

    if options.jobs_per_node != -1:
        module = load_module('driver', options.distant_driver)
        distant_driver = module.init(**options.distant_driver_params)
    else:
        distant_driver = None

    #Jobs per node is used for hybrid jobs.
    if options.jobs_per_node != -1:
        start_time = time.time()
        total_time = 0
        last_exp_time = 0
        loops = 0
        while True:
            if options.nb_dist_nodes != 1 or (total_time + 1.5 * last_exp_time > 20*60*60):
                #Launch new distant job without selecting any experiment, they
                #will be selected on the distant node.
                log("Launching on new distant nodes.")
                for i in range(options.nb_dist_nodes): #only the first execution should
                                                      #launch more than one distributed job.
                    out = dispatch_empty_job(expt_dir, distant_driver, options)
                    if out == 0:
                        raise Exception("Error trying to dispatch empty job with distant driver.")
                return
            else:
                pids = []
                for i in range(options.jobs_per_node):
                    out, pid = attempt_dispatch(experiment_config, expt_dir, chooser, driver, options)
                    if out == 0:
                        break #stop the local dispatch loop.
                    pids.append(pid)
                if len(pids) == 0:
                    #we are done, no more processes launched.
                    break
                #Wait for all local jobs.
                log("Waiting for local processes.")
                for pid in pids:
                    try:
                        os.waitpid(pid, 0)
                    except:
                        pass
                loops += 1
                last_exp_time = time.time() - total_time - start_time
                total_time = time.time() - start_time
                log("All processes done executing %i times (this batch took %f mins, total time: %f\
 mins)." % (loops, last_exp_time / 60, total_time / 60))
    else:
        #This process won't end until we run out of jobs or time.
        while True:
            out, _ = attempt_dispatch(experiment_config, expt_dir, chooser, driver, options)

            if out == 0:
                break

            # This is polling frequency. A higher frequency means that the algorithm
            # picks up results more quickly after they finish, but also significantly
            # increases overhead.
            time.sleep(options.polling_time)


# TODO:
#  * move check_pending_jobs out of ExperimentGrid, and implement two simple
#  driver classes to handle local execution and SGE execution.
#  * take cmdline engine arg into account, and submit job accordingly

def attempt_dispatch(expt_config, expt_dir, chooser, driver, options):
    '''
    Dispatches a job containing `num_jobs` jobs, if the number of jobs is greater
    than 1 they will all have the same proc_id.
    '''
    log("\n" + "-" * 40)
    if isinstance(expt_config, str):
        expt = load_experiment(expt_config)
    else:
        expt = expt_config


    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               expt.variable,
                               options.grid_size,
                               options.grid_seed)

    jobs = []
    num_jobs = 1
    for n in range(num_jobs):
        # Print out the current best function value.
        best_val, best_job = expt_grid.get_best()
        if best_job >= 0:
            log("Current best: %f (job %d)" % (best_val, best_job))
        else:
            log("Current best: No results returned yet.")

        # Gets you everything - NaN for unknown values & durations.
        grid, values, durations = expt_grid.get_grid()

        # Returns lists of indices.
        candidates = expt_grid.get_candidates()
        pending    = expt_grid.get_pending()
        complete   = expt_grid.get_complete()

        n_candidates = candidates.shape[0]
        n_pending    = pending.shape[0]
        n_complete   = complete.shape[0]
        log("%d candidates   %d pending   %d complete" %
            (n_candidates, n_pending, n_complete))

        # Verify that pending jobs are actually running, and add them back to the
        # revisit this.
        # candidate set if they have crashed or gotten lost.
        #for job_id in pending:
        #    proc_id = expt_grid.get_proc_id(job_id)
        #    if proc_id != -1 and not driver.is_proc_alive(job_id, proc_id):
        #        log("Set job %d back to candidate status." % (job_id))
        #        expt_grid.set_candidate(job_id)

        # Track the time series of optimization.
        write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_complete)

        # Print out the best job results
        write_best_job(expt_dir, best_val, best_job, expt_grid)

        if n_complete >= options.max_finished_jobs:
            log("Maximum number of finished jobs (%d) reached."
                "Exiting" % options.max_finished_jobs)
            return 0, None

        if n_candidates == 0:
            log("There are no candidates left. Exiting.")
            return 0, None

        #Don't launch unless we can launch the complete bundle.
        if n_pending >= options.max_concurrent or (n == 0 and n_pending + num_jobs > options.max_concurrent):
            log("Maximum number of jobs (%d) pending." % (options.max_concurrent))
            return 1, None
        else:
            # Ask the chooser to pick the next candidate
            log("Choosing next candidate... ")
            time_cand_start = time.time()
            job_id = chooser.next(grid, values, durations, candidates, pending, complete)
            time_cand = time.time() - time_cand_start
            log("Chose a candidate (took %i secs)." % (time_cand))

            # If the job_id is a tuple, then the chooser picked a new job.
            # We have to add this to our grid
            if isinstance(job_id, tuple):
                (job_id, candidate) = job_id
                job_id = expt_grid.add_to_grid(candidate)

            log("selected job %d from the grid." % (job_id))

            # Convert this back into an interpretable job and add metadata.
            job = Job()
            job.id        = job_id
            job.expt_dir  = expt_dir
            job.name      = expt.name
            job.language  = expt.language
            job.status    = 'submitted'
            job.submit_t  = int(time.time())
            job.param.extend(expt_grid.get_params(job_id))
            if options.nb_mini_batches > 0:
                batch_i = expt_grid.mini_batch_i
                expt_grid.mini_batch_i = (batch_i + 1) % options.nb_mini_batches
                batch_param = Parameter()
                batch_param.name = 'batch_i'
                batch_param.int_val.append(batch_i)
                job.param.extend([batch_param])

            save_job(job)
            if num_jobs == 1:
                pid = driver.submit_job(job)
                if pid != None:
                    log("submitted - pid = %s" % (pid))
                    expt_grid.set_submitted(job_id, pid)
                else:
                    log("Failed to submit job!")
                    log("Deleting job file.")
                    os.unlink(job_file_for(job))
            else:
                jobs.append(job)
                #Temporary, we don't have a proc id yet.
                expt_grid.set_submitted(job_id, -1)

    #Delayed submit when there is more than one job bundled.
    if num_jobs > 1:
        pid = driver.submit_job(jobs)
        if pid != None:
            log("Submitted %i jobs with pid = %s" % (num_jobs, pid))
            for j in jobs:
                expt_grid.set_submitted(j.id, pid)
        else:
            log("Failed to submit job!")
            log("Deleting job files.")
            for j in jobs:
                os.unlink(job_file_for(j))

    return 2, pid


def dispatch_empty_job(expt_dir, driver, options):
    '''
    Dispatches a distant job containing nothing yet.
    '''
    pid = driver.submit_empty_job(expt_dir)
    if pid != None:
        log("Submitted - pid = %s" % (pid))
        return 1
    else:
        log("Failed to submit job!")
        return 0


def load_module(folder, module_name):
    # Load up the job execution driver.
    try:
        module  = __import__(folder + '.' + module_name)
    except:
        #Ugly hack
        try:
            module = __import__('spearmint.' + folder + '.' + module_name)
            module = module.__getattribute__(folder).__getattribute__(module_name)
        except:
            raise
    return module


def write_trace(expt_dir, best_val, best_job,
                n_candidates, n_pending, n_complete):
    '''Append current experiment state to trace file.'''
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      n_candidates, n_pending, n_complete))
    trace_fh.close()


def write_best_job(expt_dir, best_val, best_job, expt_grid):
    '''Write out the best_job_and_result.txt file containing the top results.'''

    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" %
                      (best_val, best_job))
    for best_params in expt_grid.get_params(best_job):
        best_job_fh.write(str(best_params))
    best_job_fh.close()


def check_experiment_dirs(expt_dir):
    '''Make output and jobs sub directories.'''

    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)

    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)


# Cleanup locks and processes on ctl-c
def sigint_handler(signal, frame):
    if web_proc:
        print("closing web server...", end=' ')
        web_proc.terminate()
        print("done")
    sys.exit(0)


if __name__=='__main__':
    print("setting up signal handler...")
    signal.signal(signal.SIGINT, sigint_handler)
    main()

