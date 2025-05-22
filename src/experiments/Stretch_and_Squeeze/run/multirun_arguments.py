import math
from os import path
from typing import List, Tuple

import numpy as np

from  snslib.experiment.utils.args import REFERENCES, ExperimentArgParams
from  snslib.experiment.utils.misc import ref_code_recovery
from  snslib.core.subject import TorchNetworkSubject
from  snslib.core.utils.io_ import load_pickle
from  snslib.core.utils.misc       import copy_exec
from  snslib.core.utils.parameters import ArgParams
from  snslib.experiment.utils.parsing import get_rnd

# Full path to where the low_spaces.txt file is located
SPACE_TXT = 'path/to/low_spaces.txt'

# --PARAMETERS--
TASK = ['invariance'] #['adversarial', 'invariance']
NAME                = f'your_multirun_name'
GLOBAL_RSEED        = 50000
NR_SEEDS            = 10
ITER                = 500
OPTIMIZER           = 'cmaes'#'genetic'

GEN_VARIANT         = 'fc7'
NET                 = 'resnet50'
ROBUST_VARIANT      = '' #imagenet_l2_3_0.pt for robust network

# -- NETWORK PARAMETERS --
# 0: PIXEL SPACE
# 26: LAYER3_CONV1 (26_conv2d_25)
# 52: LAYER4_CONV7 (52_conv2d_51)
# 56: readout (56_linear_01)

LOW_LY              = 52 
HIGH_LY             = 56


# NOTE : if you want to run a SnS experiment with specific neurons, you should specify them in a list
# that should be specified in the section "NEURONS USED IN THE PAPER EXPERIMENTS"

# NOTE : If you want to performe experiments with subsampled representation spaces, you should specify the parameters of subsampling 
# in the section "SUBSAMPLING EXPERIMENT SPECIFICATIONS" below

N_NEURONS           = 100 

# -- PREDEFINED PARAMETERS --
TEMPLATE            = 'T'
NOISE_STRENGTH     = 0.01

# obsolete
Task2Bound = {
    'invariance': f'{LOW_LY}=N, {HIGH_LY}=N%', #<10%
    'adversarial': f'{LOW_LY}=N, {HIGH_LY}=N' #>100%
}

#--PROCESSING PARAMETERS--
if ROBUST_VARIANT:
    if ROBUST_VARIANT == 'imagenet_l2_3_0.pt':
        SBJ_LOADER = 'madryLab_robust_load'
    else:
        SBJ_LOADER = 'robustBench_load'
else:
    SBJ_LOADER = 'torch_load_pretrained'
    
RND_SEED            = get_rnd(seed=GLOBAL_RSEED, n_seeds=NR_SEEDS, add_parenthesis=False) #n_seeds=10


Task2Sign = {
    'invariance': f'{LOW_LY}=-1, {HIGH_LY}=1',
    'adversarial': f'{LOW_LY}=1, {HIGH_LY}=-1'
}

Task2NatStat = {
    'invariance': 'max',
    'adversarial': 'min'
}

subject = TorchNetworkSubject(
    NET,
    inp_shape=(1, 3, 224, 224),
)
LNAME = subject.layer_names[HIGH_LY]
LOW_L_SZ = subject.layer_shapes[LOW_LY]

#select neurons and seeds
reference_file      = load_pickle(REFERENCES)
net_key = NET+'_r' if ROBUST_VARIANT else NET


refs = ref_code_recovery(reference_file = reference_file, 
                  keys = {'network': net_key, 
                          'gen_var': GEN_VARIANT, 
                          'layer': LNAME}, 
                  ref_file_name = REFERENCES)
    
neurons_available   = list(refs.keys())
N_NEURONS = min(N_NEURONS, len(neurons_available))
neurons_idxs        = list(map(int,get_rnd(seed = GLOBAL_RSEED, n_seeds = N_NEURONS, 
                    r_range = (0,len(neurons_available)-1), add_parenthesis = False)))
NEURONS = [neurons_available[i] for i in neurons_idxs]
print(neurons_available)


# NEURONS USED IN THE PAPER EXPERIMENTS
#these are the neurons in common betw vanilla and robust references

# -- READOUT --
NEURONS = [1, 19, 40, 43, 86, 94, 101, 118, 179, 220, 230, 233, 249,
        258, 259, 284, 288, 311, 316, 323, 324, 383, 404, 429, 437, 442, 468, 470, 476, 477, 478,
        523, 527, 533, 555, 579, 589, 604, 609, 610, 624, 639, 647, 650, 654, 659, 678, 681, 682,
        690, 722, 734, 749, 755, 759, 763, 768, 769, 787, 805, 810, 828, 858, 865, 875, 889, 925,
        926, 941, 948, 956, 961, 968, 973, 975, 986, 999]
NEURONS =[f'[{n}]' for n in NEURONS]

#-- CONV 25 --
#NEURONS = ['[0, 10, 8]', '[229, 7, 0]', '[155, 8, 2]', '[161, 6, 23]', '[222, 3, 16]', '[106, 11, 9]',
#        '[247, 0, 22]', '[224, 18, 11]','[59, 9, 18]','[180, 8, 0]','[241, 15, 12]','[241, 8, 4]',
#        '[142, 2, 0]','[12, 7, 13]','[25, 25, 21]','[63, 26, 19]','[91, 7, 8]','[59, 20, 4]','[20, 10, 27]',
#        '[106, 3, 18]','[197, 11, 25]','[142, 4, 18]', '[149, 21, 17]','[134, 22, 22]','[238, 19, 23]',
#        '[234, 23, 20]', '[22, 24, 22]','[131, 12, 6]','[167, 19, 6]','[107, 24, 1]','[40, 22, 3]','[177, 6, 10]',
#        '[140, 16, 16]','[55, 2, 22]','[77, 17, 11]','[162, 13, 23]','[107, 19, 21]','[102, 21, 17]',
#        '[187, 13, 11]','[214, 4, 22]','[59, 11, 0]','[238, 23, 7]','[3, 22, 27]','[115, 14, 3]',
#        '[45, 7, 24]','[245, 27, 23]','[191, 12, 5]','[221, 20, 8]','[144, 14, 1]','[194, 25, 17]']

#-- CONV 51 --
# NEURONS = ['[448, 4, 2]', '[361, 2, 0]', '[40, 2, 6]', '[262, 3, 1]', '[468, 5, 5]', '[388, 6, 4]', '[7, 5, 6]',
#  '[444, 0, 4]', '[24, 1, 3]', '[394, 2, 6]', '[311, 2, 0]', '[127, 6, 4]', '[490, 6, 5]', '[288, 3, 0]',
#  '[214, 4, 5]', '[214, 6, 0]', '[81, 5, 0]', '[483, 2, 1]', '[429, 1, 5]', '[494, 0, 5]', '[50, 6, 5]',
#  '[476, 5, 1]', '[44, 6, 5]', '[212, 2, 2]', '[477, 4, 5]', '[284, 0, 0]', '[323, 1, 5]', '[119, 2, 4]',
#  '[118, 2, 0]', '[281, 4, 4]', '[1, 2, 2]', '[383, 3, 1]', '[119, 5, 1]', '[284, 1, 4]', '[182, 1, 2]',
#  '[205, 5, 4]', '[110, 0, 5]', '[90, 1, 6]', '[483, 3, 3]', '[324, 3, 5]', '[442, 5, 2]', '[458, 1, 0]',
#  '[299, 5, 4]', '[269, 5, 5]', '[230, 3, 0]', '[334, 4, 1]', '[213, 0, 4]', '[155, 4, 2]', '[355, 1, 2]',
#  '[374, 3, 2]']


RSEEDS = []
for n in NEURONS:
    rseeds_available = list(refs[n].keys())
    rs_idxs = list(map(int,get_rnd(seed = GLOBAL_RSEED, n_seeds = 1, 
                    r_range = (0,len(rseeds_available)), add_parenthesis = False)))
    RSEEDS.append(rseeds_available[rs_idxs[0]])
    

# --- SUBSAMPLING EXPERIMENT SPECIFICATIONS ---

# -- PARAMETERS --
SPACE_SIZE = 1000
NUM_SPACES = 10


# -- PROCESSING SPACES --
SPACE_STRINGS = ['['+' '.join(get_rnd(
    seed=n,
    n_seeds=SPACE_SIZE,
    r_range = tuple([sz-1 for sz in LOW_L_SZ[1:]]), #the 1st dimension is a mock 1
    add_parenthesis  = True,
    avoid_numbers = None
))+']' for n in range(NUM_SPACES)]

TXT_SPACE = []
# Write spaces to file
with open(SPACE_TXT, 'w') as f:
    for idx,space in enumerate(SPACE_STRINGS):
        f.write(space + '\n')
        TXT_SPACE.append(f"s{idx}")

# PARSING VIA .txt
        
def get_mrun_SnS_args():
    
    args = [
        (rs, f'{LOW_LY}=[], {HIGH_LY}={"["+n.replace("[", "(").replace("]", ")").replace(",", "")+"]" if "," in n else n}',
        f'G={GEN_VARIANT}, L={HIGH_LY}, N={n}, S={nrs}',
        Task2Sign[task], Task2Bound[task], Task2NatStat[task])
        for rs in RND_SEED
        for n,nrs in zip(NEURONS,RSEEDS)
        for task in TASK
    ]
    
    rand_seeds      = '"' + "#".join([str(rs)    for rs, _, _,   _,    _,     _  in args])+ '"'
    rec_score_ly    = '"' + "#".join([str(n)     for _,  n, _,   _,    _,     _  in args])+ '"'
    ref_p           = '"' + "#".join([str(ref)   for _,  _, ref, _,    _ ,    _  in args])+ '"'
    signatures      = '"' + "#".join([str(sign)  for _,  _, _,   sign, _,     _  in args])+ '"'
    bounds          = '"' + "#".join([str(bound) for _,  _, _,   _,    bound, _  in args]) + '"'
    nat_thresh      = '"' + "#".join([str(nt)    for _,  _, _,   _,       _,  nt in args]) + '"'
    
    return rand_seeds, rec_score_ly, ref_p, signatures, bounds, nat_thresh
    


def get_SnS_restricted_space_args():

    args = [
        (rs, f'{LOW_LY}={space}, {HIGH_LY}={"["+n.replace("[", "(").replace("]", ")").replace(",", "")+"]" if "," in n else n}',
        f'G={GEN_VARIANT}, L={HIGH_LY}, N={n}, S={nrs}',
        Task2Sign[task], Task2Bound[task], Task2NatStat[task]
        )
        for rs,space in zip(RND_SEED, TXT_SPACE)
        for n,nrs in zip(NEURONS,RSEEDS)
        for task in TASK
    ]
    
    rand_seeds      = '"' + "#".join([str(rs)    for rs, _, _,   _,    _    , _  in args])+ '"'
    rec_score_ly    = '"' + "#".join([str(n)     for _,  n, _,   _,    _    , _  in args])+ '"'
    ref_p           = '"' + "#".join([str(ref)   for _,  _, ref, _,    _    , _  in args])+ '"'
    signatures      = '"' + "#".join([str(sign)  for _,  _, _,   sign, _    , _  in args])+ '"'
    bounds          = '"' + "#".join([str(bound) for _,  _, _,   _,    bound, _  in args]) + '"'
    nat_thresh      = '"' + "#".join([str(nt)    for _,  _, _,   _,    _    , nt  in args]) + '"'
    
    return rand_seeds, rec_score_ly, ref_p, signatures, bounds,nat_thresh

if __name__ == '__main__':
    
    args = {}
    
    print('Multiple run: ')
    print('[1] SnS multi experiment')
    print('[2] SnS multi experiment - invariance task with restricted space')
    choice = int(input('Choice: '))
    
    match choice:
        
        case 1:
            
            if TASK[0] == 'adversarial':
                rand_seeds, rec_score_ly, ref_p, signatures, bounds, nat_thresh = get_mrun_SnS_args()
                
                args[str(ArgParams.RandomSeed)]                 = rand_seeds
                args[str(ExperimentArgParams.RecordingLayers)]  = rec_score_ly
                args[str(ExperimentArgParams.ScoringLayers)]    = rec_score_ly
                args[str(ExperimentArgParams.ReferenceInfo)]    = ref_p
                args[str(ExperimentArgParams.ScoringSignature)] = signatures
                args[str(ExperimentArgParams.Bounds)]           = bounds
                args[str(ExperimentArgParams.Nrec_aggregate)]   = nat_thresh
                args[str(ExperimentArgParams.Within_pareto_order)] = 'random'
                file = 'run_multi.py'
                
            elif TASK[0] == 'invariance':
                rand_seeds, rec_score_ly, ref_p, signatures, bounds, nat_thresh = get_mrun_SnS_args()
                args[str(ArgParams.RandomSeed)]                 = rand_seeds
                args[str(ExperimentArgParams.RecordingLayers)]  = rec_score_ly
                args[str(ExperimentArgParams.ScoringLayers)]    = rec_score_ly
                args[str(ExperimentArgParams.ReferenceInfo)]    = ref_p
                args[str(ExperimentArgParams.ScoringSignature)] = signatures
                args[str(ExperimentArgParams.Bounds)]           = bounds
                args[str(ExperimentArgParams.Nrec_aggregate)]   = nat_thresh
                
                args[str(ExperimentArgParams.Within_pareto_order)] = 'onevar'
                
                file = 'run_multi_rand_init.py'
                    
        case 2:
            rand_seeds, rec_score_ly, ref_p, signatures, bounds, nat_thresh = get_SnS_restricted_space_args()
            
            args[str(ArgParams.RandomSeed)]                 = rand_seeds
            args[str(ExperimentArgParams.RecordingLayers)]  = rec_score_ly
            args[str(ExperimentArgParams.ScoringLayers)]    = rec_score_ly
            args[str(ExperimentArgParams.ReferenceInfo)]    = ref_p
            args[str(ExperimentArgParams.ScoringSignature)] = signatures
            args[str(ExperimentArgParams.Bounds)]           = bounds
            args[str(ExperimentArgParams.Nrec_aggregate)]   = nat_thresh
            args[str(ExperimentArgParams.Within_pareto_order)] = 'onevar'
            
            file = 'run_multi_rand_init.py'
        case 0:
            
            print('Exit')
            
        case _:
            raise ValueError('Invalid choice')
    
    args[str(          ArgParams.NumIterations )] = ITER
    args[str(          ArgParams.ExperimentName)] = NAME
    args[str(ExperimentArgParams.Template      )] = TEMPLATE
    args[str(ExperimentArgParams.GenVariant    )] = GEN_VARIANT
    args[str(ExperimentArgParams.NetworkName   )] = NET
    if ROBUST_VARIANT : args[str(ExperimentArgParams.CustomWeightsVariant)] = ROBUST_VARIANT
    args[str(ExperimentArgParams.WeightLoadFunction)] = SBJ_LOADER
    args[str(ExperimentArgParams.OptimType)] = OPTIMIZER
    args[str(ExperimentArgParams.Noise_strength)] = NOISE_STRENGTH
    

    copy_exec(file=file, args=args)