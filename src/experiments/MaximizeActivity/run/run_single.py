'''
TODO Experiment description
'''

import matplotlib

from experiments.MaximizeActivity.args import ARGS
from  snslib.experiment.utils.misc import run_single
from  snslib.experiment.maximize_activity import MaximizeActivityExperiment

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=MaximizeActivityExperiment)
