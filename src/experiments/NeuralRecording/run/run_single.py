'''
TODO Experiment description
'''

from  snslib.experiment.utils.misc import run_single
from experiments.NeuralRecording.args import ARGS
from  snslib.experiment.neural_recording import NeuralRecordingExperiment


if __name__ == '__main__':
    
    run_single(args_conf=ARGS, exp_type=NeuralRecordingExperiment)


