import glob
import os
from distutils.core import setup

# TODO: probably best to put all python files except for the top level run
# script into a package (sub-directory), but for now we can do this.
MODULES = ['spearmint.ExperimentGrid', 'spearmint.gp', 'spearmint.helpers', 'spearmint.Locker', 'spearmint.runner', 'spearmint.sobol_lib', 'spearmint.spearmint_pb2', 'spearmint.supermint', 'spearmint.util', 'spearmint.main']

setup(name='spearmint',
      description="Practical Bayesian Optimization of Machine Learning Algorithms",
      author="Jasper Snoek, Hugo Larochelle, Ryan P. Adams",
      url="https://github.com/JasperSnoek/spearmint",
      version='1.0',
      license='GPLv3',
      packages=['spearmint', 'spearmint.driver', 'spearmint.chooser'],
      py_modules=MODULES
     )
