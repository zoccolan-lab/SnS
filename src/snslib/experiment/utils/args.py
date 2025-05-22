from __future__ import annotations

from os import path
from enum import Enum

from snslib.core.utils.parameters import ArgParam
from snslib.core.utils.io_ import read_json

SCRIPT_DIR     = path.abspath(path.join(__file__, '..', '..'))
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')

local_setting = read_json(LOCAL_SETTINGS)

OUT_DIR            : str = local_setting.get('out_dir',      None)
WEIGHTS            : str = local_setting.get('weights',      None)
DATASET            : str = local_setting.get('dataset',      None)
IMAGE              : str = local_setting.get('image',        None)
ALEXNET_DIR        : str = local_setting.get('alexnet_dir',  None)
FEATURE_MAPS       : str = local_setting.get('feature_maps', None)
REFERENCES         : str = local_setting.get('references',   None)
CUSTOM_WEIGHTS     : str = local_setting.get('custom_weights',  None)
NATURAL_RECORDINGS : str = local_setting.get('natural_recordings', None)

LAYERS_NEURONS_SPECIFICATION = '''
You can specify recording and scoring layers as follows:
layer_num_id = [neurons to record]. If empty, all neurons from the layer will be recorded.
If you are recording from a linear layer, indicate target neurons as follows:
[A B C D ...]               <-- each neuron is identified by a single number, requiring no parenthesis

If you are recording from a convolutional layer, indicate target neurons as follows:
[(A1 A2 A3) (B1 B2 B3) ...] <-- each neuron is identified by a tuple of numbers
'''

class ExperimentArgParams(Enum):
    
    # Generator
    GenWeights         = ArgParam(name="weights",            type=str,   help="Path to folder with DeepSim generator weights")
    GenVariant         = ArgParam(name="variant",            type=str,   help="Variant of DeepSim generator to use (default: `fc7`)")
    
    # Natural image dataloader
    Dataset            = ArgParam(name="dataset",            type=str,   help="Path to mini-imagenet dataset")
    BatchSize          = ArgParam(name="batch_size",         type=int,   help="Natural image dataloader batch size")
    Template           = ArgParam(name="template",           type=str,   help="String of True(T) and False(F) as the basic sequence of the image presentation mask (i.e. the sequence of natural and optimized images that will be scored during optimization)")
    Shuffle            = ArgParam(name="shuffle",            type=bool,  help="If to shuffle mask template")
    
   

    
    # Recording
    ImageIds           = ArgParam(name="image_ids",          type=str,   help="Image indexes for recording separated by a comma")
    LogCheckpoint      = ArgParam(name="log_chk",            type=int,   help="Logger iteration checkpoint")
    Nat_recs           = ArgParam(name="nat_recs",           type=str,   help="Path to natural images recordings")
    Nrec_aggregate     = ArgParam(name="nrec_aggregate",     type=str,   help="Way to aggregate natural recordings for the neurons of interest")
    
    # Subject
    NetworkName         = ArgParam(name="net_name",           type=str,   help="SubjectNetwork name")
    RecordingLayers     = ArgParam(name="rec_layers",         type=str,   help=f"Recording layers with specification\n{LAYERS_NEURONS_SPECIFICATION}")
    CustomWeightsPath   = ArgParam(name="robust_path",        type=str,   help="Path to weights of robust version of the subject network")
    CustomWeightsVariant= ArgParam(name="robust_variant",     type=str,   help="Variant of robust network")
    WeightLoadFunction  = ArgParam(name="w_load_funct",       type=str,   help="Function to load the torch network subject")
    
    # outdated recording parsing
    Rec_low             = ArgParam(name="rec_low",            type=str,   help="Recording low layer")
    Rec_high            = ArgParam(name="rec_high",           type=str,   help="Recording high layer")
    
    # Scorer
    ScoringSignature   = ArgParam(name="scr_sign",           type=str,   help="Scoring signature for WeightedPairSimilarityScorer")
    Bounds             = ArgParam(name="bounds",             type=str,   help="Bounds for the WeightedPairSimilarityScorer") # outdated
    ScoringLayers      = ArgParam(name="scr_layers",         type=str,   help=f"Target scoring layers and neurons with specification\n{LAYERS_NEURONS_SPECIFICATION}")
    UnitsReduction     = ArgParam(name="units_reduction",    type=str,   help="Name of reducing function across units")
    LayerReduction     = ArgParam(name="layer_reduction",    type=str,   help="Name of reducing function across layers")
    Distance           = ArgParam(name="distance",           type=str,   help="Distance metric for the scorer")
    Reference          = ArgParam(name="reference",          type=str,   help="Path to file containing reference supestimuli mapping layer->neuron->rand_seed->superstimuli")
    ReferenceInfo      = ArgParam(name="reference_info",     type=str,   help="Reference info in format L=<layer>, N=<neuron>, S=<seed>")
    Within_pareto_order= ArgParam(name="w_pareto_order",     type=str,   help="Ordering method used within each Pareto front (either random (adversarial task), or onevar (invariance task))")
    
    # outdated scoring parsing
    Score_low             = ArgParam(name="score_low",            type=str,   help="Scoring low layer")
    Score_high            = ArgParam(name="score_high",           type=str,   help="Scoring high layer")
    
    # Optimizer
    OptimType          = ArgParam(name="optimizer_type",     type=str,   help="Type of optimizer. Either `genetic` or `cmaes` (cmaes was used in the paper)")
    PopulationSize     = ArgParam(name="pop_size",           type=int,   help="Starting number of the population")
    MutationRate       = ArgParam(name="mut_rate",           type=float, help="Mutation rate for the optimizer (genetic)")
    MutationSize       = ArgParam(name="mut_size",           type=float, help="Mutation size for the optimizer (genetic)")
    NumParents         = ArgParam(name="n_parents",          type=int,   help="Number of parents for the optimizer (genetic)")
    TopK               = ArgParam(name="topk",               type=int,   help="Number of codes of previous generation to keep (genetic)")
    Temperature        = ArgParam(name="temperature",        type=float, help="Temperature for the optimizer (genetic)")
    TemperatureFactor  = ArgParam(name="temperature_factor", type=float, help="Temperature for the optimizer (genetic)")
    RandomDistr        = ArgParam(name="random_distr",       type=str,   help="Random distribution for the codes initialization")
    AllowClones        = ArgParam(name="allow_clones",       type=str,   help="Random distribution for the codes initialization (genetic)")
    RandomScale        = ArgParam(name="random_scale",       type=float, help="Random scale for the random distribution sampling")
    Sigma0             = ArgParam(name="sigma0",             type=float, help="Initial variance for CMAES covariance matrix")
    Noise_strength     = ArgParam(name="noise_strength",     type=float, help="Noise strength for initial code")
    

    
    # --- MAGIC METHODS ---
    
    def __str__ (self)  -> str: return self.value.name
    def __repr__(self)  -> str: return str(self)
    
    
    @classmethod
    def from_str(cls, name: str) -> ArgParam:
        ''' Return the argument from the string name. '''
        
        for arg in cls:
            if str(arg) == name: return arg.value
        
        raise ValueError(f'Argument with name {name} not found')
    

