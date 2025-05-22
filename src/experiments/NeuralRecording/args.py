from  snslib.experiment.utils.args import CUSTOM_WEIGHTS, DATASET, OUT_DIR, ExperimentArgParams
from  snslib.core.utils.io_ import load_pickle
from  snslib.core.utils.misc import deep_get
from  snslib.core.utils.parameters import ArgParams, ParamConfig

# --- Parameters ---
NET_TYPE = 'resnet50' #add '_r' if you are recording from the robust network

#Option 1) Use the reference recording from a previous run to identify your target neurons 
# (important for deep layers), 
#REF_REC = load_pickle('path/to/data.pkl')
#REC_LY = '56_linear_01'
#REF_NEURONS = deep_get(REF_REC, ['reference',NET_TYPE,'fc7',REC_LY])
#REC_SPACE = REC_LY.split('_')[0]+'=['+' '.join(REF_NEURONS).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')+']'
#Option 2) If you want to record all the neurons of the layer (good for the readout layer)
REC_SPACE = "56=[]"


if NET_TYPE.split('_')[-1] == 'r':
    ROBUST_VARIANT = 'imagenet_l2_3_0.pt'
    SBJ_LOADER = 'madryLab_robust_load'
   
else:
    ROBUST_VARIANT = ''
    SBJ_LOADER = 'torch_load_pretrained'

ARGS: ParamConfig = {
    
    # Subject
    ExperimentArgParams.NetworkName.value     : "resnet50",
    ExperimentArgParams.RecordingLayers.value : REC_SPACE,
    ExperimentArgParams.CustomWeightsPath.value : CUSTOM_WEIGHTS     , 
    ExperimentArgParams.CustomWeightsVariant.value : ROBUST_VARIANT              , 
    ExperimentArgParams.WeightLoadFunction.value : SBJ_LOADER      , 
    
    
    # Dataset
    # Path to the ImageNet training set or miniImageNet
    ExperimentArgParams.Dataset.value       : "/data/ImageNet/train", 
    ExperimentArgParams.ImageIds.value      : "",
    ExperimentArgParams.LogCheckpoint.value : 1,
    
    # Logger
    ArgParams.ExperimentName.value    : "my_neurec_experiment",
    ArgParams.ExperimentVersion.value : 0,
    ArgParams.OutputDirectory.value   : OUT_DIR,
}