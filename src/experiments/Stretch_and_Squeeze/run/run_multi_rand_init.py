from experiments.Stretch_and_Squeeze.args import ARGS
from  snslib.experiment.utils.misc import run_multi
from  snslib.experiment.stretch_and_squeeze import StretchSqueezeLayerMultiExperiment, StretchSqueezeExperiment_randinit

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=StretchSqueezeExperiment_randinit,
    multi_exp_type=StretchSqueezeLayerMultiExperiment
)
