'''
TODO Experiment description
'''

from experiment.utils.misc import run_single
from experiment.NeuralRecording.args import ARGS
from experiment.NeuralRecording.neural_recording import NeuralRecordingExperiment


if __name__ == '__main__':
    
    run_single(args_conf=ARGS, exp_type=NeuralRecordingExperiment)


