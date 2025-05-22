'''
TODO Experiment description
'''

import matplotlib

from experiments.Stretch_and_Squeeze.args import ARGS
from  snslib.experiment.utils.misc import run_single
from  snslib.experiment.stretch_and_squeeze import StretchSqueezeExperiment_randinit

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=StretchSqueezeExperiment_randinit)