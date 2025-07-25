from  snslib.experiment.utils.args import DATASET, OUT_DIR, WEIGHTS,REFERENCES, CUSTOM_WEIGHTS,ExperimentArgParams
from  snslib.core.utils.parameters import ArgParams, ParamConfig

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , 
    ExperimentArgParams.GenVariant       .value : "fc7"              ,

    # Natural Images
    ExperimentArgParams.Template         .value : "T"               , 
    ExperimentArgParams.Dataset          .value : DATASET            ,
    ExperimentArgParams.Shuffle          .value : False              , 
    ExperimentArgParams.BatchSize        .value : 16                 , 

    # Subject
    ExperimentArgParams.NetworkName      .value : "resnet50"          , 
    ExperimentArgParams.RecordingLayers  .value : "56=[1]"           , 
    ExperimentArgParams.CustomWeightsPath.value : CUSTOM_WEIGHTS     , # 'path/to/robust/model/weights' as specified in local_settings.json ,
    ExperimentArgParams.CustomWeightsVariant.value : ''              , # 'imagenet_l2_3_0.pt'
    ExperimentArgParams.WeightLoadFunction.value : 'torch_load_pretrained' ,
    
    #, Scorer
    ExperimentArgParams.ScoringLayers    .value : "56=[1]"            ,
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,
    ExperimentArgParams.Reference   .value : REFERENCES,

    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,
    ExperimentArgParams.OptimType        .value : "cmaes"              ,

    # Logger
    ArgParams          .ExperimentName   .value : "maximize_activity", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 500                ,
    ArgParams          .RandomSeed       .value : 50001              ,   #50000
    ArgParams          .Render           .value : False,

}
