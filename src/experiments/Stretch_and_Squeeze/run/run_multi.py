'''
TODO Experiment description
'''

from experiments.Stretch_and_Squeeze.args import ARGS
from src.snslib.experiment.utils.misc import run_multi
from src.snslib.experiment.stretch_and_squeeze import StretchSqueezeLayerMultiExperiment, StretchSqueezeExperiment

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=StretchSqueezeExperiment,
    multi_exp_type=StretchSqueezeLayerMultiExperiment
)
