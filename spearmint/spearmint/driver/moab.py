import os
import sys
import re
import subprocess
import drmaa

from .dispatch import DispatchDriver
from .. import helpers

class MoabDriver(DispatchDriver):
    def __init__(self, job_names_suffix, job_id_suffix='', extra_sub_args='', **kwargs):
        self.job_names_suffix = job_names_suffix
        self.job_id_suffix = job_id_suffix
        self.extra_sub_args = extra_sub_args

    def submit_job(self, job):
        output_file = helpers.job_output_file(job)
        job_file    = helpers.job_file_for(job)
        mint_path   = sys.argv[0]
        script  = 'python3 %s --run-job "%s" .' % (mint_path, job_file)
        script = script.encode('ASCII')

        sub_cmd    = "msub -S /bin/bash -N %s-%d -j oe -o %s -l nodes=1:ppn=8" % (
            job.name + self.job_names_suffix, job.id, output_file)

        sub_cmd = sub_cmd + ' ' + self.extra_sub_args

        process = subprocess.Popen(sub_cmd.encode('ASCII'),
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True)
        msub_output, _ = process.communicate(input=script)
        process.stdin.close()

        msub_output = msub_output.decode()

        # Parse out the job id.
        match = re.search(r'\d{5,25}', msub_output)

        if match:
            external_job_id = int(match.group())
        else:
            raise Exception('Error while submitting moab job. Could not retrieve job id. msub output : %s' % msub_output)

        #This external job ID is pretty useless, we need to extract the internal job id.
        output = subprocess.check_output(["checkjob", "-v", str(external_job_id)])

        match = re.search(r'DstRMJID: (.*)' + self.job_id_suffix, output.decode())
        if match:
            internal_job_id = match.group(1)
        else:
            raise Exception("Couldn't find internal job id, required for drmaa operations. msub command output : %s. checkjob output : %s" % (msub_output, output))

        return internal_job_id

    def is_proc_alive(self, job_id, torque_id):
        torque_id = str(torque_id) + self.job_id_suffix
        try:
            s = drmaa.Session()
            s.initialize()

            reset_job = False

            try:
                status = s.jobStatus(torque_id)
            except:
                helpers.log("EXC: %s\n" % (str(sys.exc_info()[0])))
                helpers.log("Could not find Torque id for job %s (%s)\n" % (job_id, torque_id))
                status = -1
                reset_job = True

            if status == drmaa.JobState.UNDETERMINED:
                helpers.log("Job %s (%s) in undetermined state.\n" % (job_id, torque_id))
                reset_job = True

            elif status == drmaa.JobState.QUEUED_ACTIVE:
                helpers.log("Job %s (%s) waiting in queue.\n" % (job_id, torque_id))

            elif status == drmaa.JobState.RUNNING:
                helpers.log("Job %s (%s) is running.\n" % (job_id, torque_id))

            elif status in [drmaa.JobState.SYSTEM_ON_HOLD,
                            drmaa.JobState.USER_ON_HOLD,
                            drmaa.JobState.USER_SYSTEM_ON_HOLD,
                            drmaa.JobState.SYSTEM_SUSPENDED,
                            drmaa.JobState.USER_SUSPENDED]:
                helpers.log("Job %s (%s) is held or suspended.\n" % (job_id, torque_id))
                reset_job = True

            elif status == drmaa.JobState.DONE:
                helpers.log("Job %s (%s) is finished.\n" % (job_id, torque_id))

            elif status == drmaa.JobState.FAILED:
                helpers.log("Job %s (%s) failed.\n" % (job_id, torque_id))
                reset_job = True

            if reset_job:


                try:
                    # Kill the job.
                #    s.control(str(sgeid), drmaa.JobControlAction.TERMINATE)
                #    helpers.log("Killed SGE job %s.\n" % (torque_id))
                    helpers.log("Would try to kill SGE job %s.\n" % (torque_id))
                except:
                    helpers.log("Failed to kill SGE job %s.\n" % (torque_id))

                return False
            else:
                return True

        finally:
            s.exit()


def init(**kwargs):
    return MoabDriver(**kwargs)

