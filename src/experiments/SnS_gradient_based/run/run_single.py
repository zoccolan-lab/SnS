from experiments.SnS_gradient_based.args import ARGS
from experiment.utils.misc import run_single
from src.snslib.experiment.SnS_gradient_based import SnS_gradient_based

#matplotlib.use('TKAgg')

if __name__ == '__main__': run_single(args_conf=ARGS, exp_type=SnS_gradient_based)