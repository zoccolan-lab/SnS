from experiments.AdversarialAttack_BMM.args import ARGS
from src.snslib.experiment.utils.misc import run_multi
from src.snslib.experiment.adversarial_attack_max import StretchSqueezeLayerMultiExperiment, StretchSqueezeExperiment_randinit

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=StretchSqueezeExperiment_randinit,
    multi_exp_type=StretchSqueezeLayerMultiExperiment
)
