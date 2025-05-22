from  snslib.experiment.utils.args import DATASET, NATURAL_RECORDINGS, OUT_DIR, WEIGHTS, REFERENCES, CUSTOM_WEIGHTS, ExperimentArgParams
from  snslib.core.utils.parameters import ArgParams, ParamConfig
import os

ARGS: ParamConfig = {

    # Natural image dataloader
    ExperimentArgParams.GenWeights       .value : WEIGHTS            , # Path to the weights of the image generator (fc7.pt path)
    ExperimentArgParams.GenVariant       .value : "fc7"              , # image generator variant

    # Natural Images
    ExperimentArgParams.Template         .value : "T"                , 
    ExperimentArgParams.Dataset          .value : DATASET            , # Path to the ImageNet training set or miniImageNet
    ExperimentArgParams.Shuffle          .value : False              , 
    ExperimentArgParams.BatchSize        .value : 16                 , 
    
    ExperimentArgParams.Nat_recs         .  value : NATURAL_RECORDINGS,
    ExperimentArgParams.Nrec_aggregate   .value : "max",

    # Subject
    ExperimentArgParams.NetworkName             .value : 'resnet50',        # resnet50
    ExperimentArgParams.RecordingLayers         .value : "0=[], 56=[1]"  , # 56 resnet50
    ExperimentArgParams.CustomWeightsPath       .value : CUSTOM_WEIGHTS, 
    ExperimentArgParams.CustomWeightsVariant    .value : '', # if you want to use the robust resnet50, use 'imagenet_l2_3_0.pt'
    ExperimentArgParams.WeightLoadFunction.value       : 'torch_load', # if using robust resnet50, use 'madryLab_robust_load'
    
    ExperimentArgParams.Rec_low .value : "",
    ExperimentArgParams.Rec_high .value : "",
    # Scorer
    ExperimentArgParams.ScoringLayers    .value : "0=[],56=[1]"           , # layer where to record the activity, leave empty for scoring the whole layer
    ExperimentArgParams.ScoringSignature .value : "0=-1, 56=1"       , # -1 is stretching, 1 is squeezing
    ExperimentArgParams.Bounds           .value : "0=N, 56=N"       , # obsolete
    ExperimentArgParams.Distance         .value : "euclidean"        , 
    ExperimentArgParams.UnitsReduction   .value : "mean"             ,
    ExperimentArgParams.LayerReduction   .value : "mean"             ,
    ExperimentArgParams.Reference        .value : REFERENCES ,
    ExperimentArgParams.ReferenceInfo    .value : "G=fc7, L=56, N=[1], S=10"  ,
    # ordering of the pareto front, should be onevar when doing Invariance task, otherwise 'random' for adversarial task
    ExperimentArgParams.Within_pareto_order.value : 'onevar',           
    
    ExperimentArgParams.Score_low .value : "",
    ExperimentArgParams.Score_high .value : "",
    
    # Optimizer
    ExperimentArgParams.PopulationSize   .value : 50                 ,
    ExperimentArgParams.Sigma0           .value : 1.0                ,
    ExperimentArgParams.OptimType        .value : 'cmaes'            ,
    ExperimentArgParams.Noise_strength   .value : 0.01              ,

    # Logger
    ArgParams          .ExperimentName   .value : "invariance", 
    ArgParams          .ExperimentVersion.value : 0                  , 
    ArgParams          .OutputDirectory  .value : OUT_DIR            , 

    # Globals
    ArgParams          .NumIterations    .value : 500              ,
    ArgParams          .RandomSeed       .value : 20           ,
    ArgParams          .Render           .value : False,

}