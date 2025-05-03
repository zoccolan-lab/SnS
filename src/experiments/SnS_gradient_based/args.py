from src.snslib.experiment.utils.args import DATASET, NATURAL_RECORDINGS, OUT_DIR, WEIGHTS, REFERENCES, CUSTOM_WEIGHTS, ExperimentArgParams
from src.snslib.core.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , 
    ExperimentArgParams.GenVariant       .value : "fc7"              ,

    
    ExperimentArgParams.Nat_recs         .  value : NATURAL_RECORDINGS,
    ExperimentArgParams.Nrec_aggregate   .value : 'max', #max

    # Subject
    ExperimentArgParams.NetworkName             .value : 'resnet50',        # resnet50
    ExperimentArgParams.RecordingLayers         .value : "26=[], 56=[19]"  , # 56 resnet50
    ExperimentArgParams.CustomWeightsPath       .value : CUSTOM_WEIGHTS, 
    ExperimentArgParams.CustomWeightsVariant    .value : 'imagenet_l2_3_0.pt', # 'imagenet_l2_3_0.pt'
    ExperimentArgParams.WeightLoadFunction.value       : 'madryLab_robust_load', #torch_load
    
    # Scorer
    ExperimentArgParams.Reference        .value : REFERENCES ,
    ExperimentArgParams.ReferenceInfo    .value : "G=fc7, L=56, N=[19], S=1"  ,#482726
    ExperimentArgParams.ScoringSignature .value : "26=-1, 56=1"       ,

    #GD SnS
    ExperimentArgParams.LearningRate     .value : 0.05             ,


    # Logger
    ArgParams          .ExperimentName   .value : "SnS_gradient_based", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 500              ,
    ArgParams          .RandomSeed       .value : 20           ,#50000

}