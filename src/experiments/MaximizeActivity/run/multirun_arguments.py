
from functools import reduce
import operator
import random
from typing import List, Set, Tuple

import numpy as np

from  snslib.experiment.utils.misc import ref_code_recovery
from  snslib.core.utils.io_ import load_pickle
from  snslib.core.utils.misc import copy_exec
from  snslib.core.utils.parameters import ArgParams
from  snslib.experiment.utils.args import REFERENCES, ExperimentArgParams
from  snslib.core.subject import TorchNetworkSubject


NAME   = f'your_experiment_name'

# PARAMETERS

# number of iterations
ITER     = 500

# network architecture, follow torchvision.models nomenclature
NET             = 'resnet50'

# robust network architecture. If standard network is used, leave empty ''. 
# If you want to use a robust network, specify 'imagenet_l2_3_0.pt' for robust ResNet50
ROBUST_VARIANT  = '' 

# deepsim generator variant, leave as is to follow paper experiments
REF_GEN_VARIANT = ['fc7']

# layer to record activity from. Set to [26] for mid level neurons,
# [52] for high level neurons, [56] for readout units.
REF_LAYERS      = [56]

# number of neurons to record activity from (random selection)
#NOTE : To specify a list of units, you must decomment the variable REF_NEURONS below.
NUM_NEURONS = 100

# global seed for reproducibility
GLOBAL_SEED = 31415

# number of seeds for optimization initialization
N_SEEDS = 4

# --- REFERENCES ---
def get_rnd(
    seed=None,
    n_seeds=10,
    r_range: Tuple[Tuple[int, int], ...] | Tuple[int, int] = (1, 10),
    add_parenthesis: bool = True,
    avoid_numbers: Set[int] | Tuple[Set[int],...] = None
):
    if isinstance(r_range[0], int):
        if len(r_range) == 2:
            r_range = (r_range,)
        elif isinstance(r_range[0], int):
            r_range = tuple((0,r) for r in r_range)  
        else:  # fast parsing of tuple(tuple[int, int], ...)
            r_range = tuple((r,) for r in r_range)

    if seed is not None:
        random.seed(seed)

    if avoid_numbers is None:
        avoid_numbers = tuple(set() for _ in range(len(r_range)))

    if not isinstance(avoid_numbers, tuple): avoid_numbers =(avoid_numbers,)
    # Calculate the total number of possible unique numbers
    total_possible_numbers = reduce(operator.mul,[end - start + 1 - len(avoid_numbers[i]) 
                                                  for i,(start, end) in enumerate(r_range)])
    # Security check
    if n_seeds > total_possible_numbers:
        raise ValueError("Requested more unique numbers than possible to sample given the range and avoid_numbers.")


    unique_numbers = set()

    while len(unique_numbers) < n_seeds:
        idx = []
        for i,inner in enumerate(r_range):
            start, end = inner
            an = avoid_numbers[i]

            # Generate a random number that is not in avoid_numbers
            while True:
                rand_num = random.randint(start, end)
                if rand_num not in an:
                    break

            idx.append(str(rand_num))

        if add_parenthesis:
            unique_numbers.add('(' + ' '.join(idx) + ')')
        else:
            unique_numbers.add(' '.join(idx))

    return list(unique_numbers)



subject = TorchNetworkSubject(
    NET,
    inp_shape=(1, 3, 224, 224),
)
LNAME = subject.layer_names[REF_LAYERS[0]]
layer_shape = subject.layer_shapes[REF_LAYERS[0]]


net_key = NET+'_r' if ROBUST_VARIANT else NET
try:
    reference_file      = load_pickle(REFERENCES)
    refs = ref_code_recovery(reference_file = reference_file, 
                    keys = {'network': net_key, 
                            'gen_var': REF_GEN_VARIANT[0], 
                            'layer': LNAME}, 
                    ref_file_name = REFERENCES)
    neurons_present = set([int(key.strip('[]')) for key in refs.keys()])
except:
    neurons_present = None


REF_SEED        = get_rnd(seed=GLOBAL_SEED, n_seeds=N_SEEDS, add_parenthesis=False) 
if ROBUST_VARIANT:
    if ROBUST_VARIANT == 'imagenet_l2_3_0.pt':
        SBJ_LOADER = 'madryLab_robust_load'
    else:
        SBJ_LOADER = 'robustBench_load'
else:
    SBJ_LOADER = 'torch_load_pretrained'
    
print('layer shape',layer_shape)
layer_shape = tuple([e-1 for e in layer_shape]) if len(layer_shape) == 2 else tuple([e-1 for e in layer_shape[1:]])
REF_NEURONS = get_rnd(seed=GLOBAL_SEED, n_seeds=NUM_NEURONS, r_range=layer_shape, avoid_numbers = neurons_present) #  

# These are the readout neurons used in the paper

# REF_NEURONS = [1, 19, 40, 43, 86, 94, 101, 118, 179, 220, 230, 233, 249,
# 258, 259, 284, 288, 311, 316, 323, 324, 383, 404, 429, 437, 442, 468, 470, 476, 477, 478,
# 523, 527, 533, 555, 579, 589, 604, 609, 610, 624, 639, 647, 650, 654, 659, 678, 681, 682,
# 690, 722, 734, 749, 755, 759, 763, 768, 769, 787, 805, 810, 828, 858, 865, 875, 889, 925,
# 926, 941, 948, 956, 961, 968, 973, 975, 986, 999]

def get_args_reference() -> Tuple[str, str, str]:
    
    args = [
        (gen_var, f'{layer}=[{neuron}]', seed)
        for gen_var in REF_GEN_VARIANT
        for layer   in REF_LAYERS
        for neuron  in REF_NEURONS
        for seed    in REF_SEED
    ]
    
    gen_var_str = '#'.join(a for a, _, _ in args)
    rec_str     = '"' + '#'.join(a for _, a, _ in args) + '"'
    seed_str    = '#'.join(a for _, _, a in args)
    
    return gen_var_str, rec_str, seed_str


if __name__ == '__main__':

    print('Multiple run: ')
    print('[1] Create references')
    
    option = int(input('Choose option: '))
    
    match option:
        
        case 1:
            
            gen_var_str, rec_layer_str, seed_str = get_args_reference()
            
            args = {
                str(ExperimentArgParams.GenVariant     ) : gen_var_str,
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : rec_layer_str,
                str(ExperimentArgParams.NetworkName    ) : NET,
                str(          ArgParams.RandomSeed     ) : seed_str,
                str(ExperimentArgParams.WeightLoadFunction): SBJ_LOADER
            }
            if ROBUST_VARIANT : args[str(ExperimentArgParams.CustomWeightsVariant)] = ROBUST_VARIANT
            file = 'run_multiple_references.py'
            
        case _:
            
            print('Invalid option')
            
    args[str(ArgParams          .ExperimentName)] = NAME
    args[str(ArgParams          .NumIterations )] = str(ITER)
    args[str(ExperimentArgParams.Template      )] = 'T'
    
    copy_exec(file=file, args=args ) 
