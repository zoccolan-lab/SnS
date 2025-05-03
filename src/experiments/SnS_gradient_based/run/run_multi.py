'''
TODO Experiment description
'''

from experiments.SnS_gradient_based.args import ARGS
from src.snslib.experiment.utils.misc import run_multi
from src.snslib.experiment.SnS_gradient_based import SnS_gradient_based, SnSGDMultiExperiment

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=SnS_gradient_based,
    multi_exp_type=SnSGDMultiExperiment
)
