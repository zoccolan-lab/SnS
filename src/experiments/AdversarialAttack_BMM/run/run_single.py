'''
TODO Experiment description
'''

import matplotlib

from experiments.AdversarialAttack_BMM.args import ARGS
from src.snslib.experiment.utils.misc import run_single
from src.snslib.experiment.adversarial_attack_max import StretchSqueezeExperiment

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=StretchSqueezeExperiment)