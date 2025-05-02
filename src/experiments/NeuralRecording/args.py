from experiment.utils.args import CUSTOM_WEIGHTS, DATASET, OUT_DIR, ExperimentArgParams
from pxdream.utils.io_ import load_pickle
from pxdream.utils.misc import deep_get
from pxdream.utils.parameters import ArgParams, ParamConfig

REF_REC = load_pickle('/home/ltausani/Desktop/Zout/MaximizeActivity/08425_references_rnet50_robust__01conv01/data.pkl')
REC_LY = '56_linear_01'
NET_TYPE = 'resnet50'
# REF_NEURONS = deep_get(REF_REC, ['reference',NET_TYPE,'fc7',REC_LY])
# REC_SPACE = REC_LY.split('_')[0]+'=['+' '.join(REF_NEURONS).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')+']'

REC_SPACE = "56=[]"

if NET_TYPE.split('_')[-1] == 'r':
    ROBUST_VARIANT = 'imagenet_l2_3_0.pt'
    SBJ_LOADER = 'madryLab_robust_load'
    #SBJ_LOADER = 'robustBench_load' #Not used at the moment
else:
    ROBUST_VARIANT = ''
    SBJ_LOADER = 'torch_load_pretrained'

ARGS: ParamConfig = {
    
    # Subejct
    ExperimentArgParams.NetworkName.value     : "resnet50",
    ExperimentArgParams.RecordingLayers.value : REC_SPACE,#"54=[(560 6 1)]"
    ExperimentArgParams.CustomWeightsPath.value : CUSTOM_WEIGHTS     , 
    ExperimentArgParams.CustomWeightsVariant.value : ROBUST_VARIANT              , #'imagenet_l2_3_0.pt''
    ExperimentArgParams.WeightLoadFunction.value : SBJ_LOADER      , #torch_load
    
    
    # Dataset
    ExperimentArgParams.Dataset.value       : "/data/ImageNet/train",
    ExperimentArgParams.ImageIds.value      : "",
    ExperimentArgParams.LogCheckpoint.value : 1,
    
    # Logger
    ArgParams.ExperimentName.value    : "neurec_rnet50_vaniGlia_rout_and_lbls",
    ArgParams.ExperimentVersion.value : 0,
    ArgParams.OutputDirectory.value   : OUT_DIR,
}