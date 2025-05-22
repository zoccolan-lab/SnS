'''
TODO Experiment description
'''


from experiments.MaximizeActivity.args import ARGS
from  snslib.experiment.utils.misc import run_multi
from  snslib.experiment.maximize_activity import MaximizeActivityExperiment, NeuronReferenceMultiExperiment

if __name__ == '__main__': run_multi(
    args_conf=ARGS, 
    exp_type=MaximizeActivityExperiment,
    multi_exp_type=NeuronReferenceMultiExperiment
)