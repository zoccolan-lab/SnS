import os
import re
from typing import Any, Dict, List, Type, cast
from snslib.experiment.utils.args import ExperimentArgParams
from snslib.experiment.utils.parsing import parse_boolean_string
from snslib.core.experiment import Experiment, MultiExperiment, ZdreamExperiment
from snslib.core.utils.parameters import ArgParam, ArgParams, ParamConfig, Parameter
from snslib.core.utils.misc import deep_get, overwrite_dict
from snslib.core.utils.logger import DisplayScreen, Logger, LoguruLogger, SilentLogger
from IPython import get_ipython
import argparse

SPACE_TXT = '/home/ltausani/Documents/GitHub/ZXDREAM/experiment/AdversarialAttack_BMM/run/low_spaces.txt'


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Eseguito in un Jupyter Notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Eseguito in un terminale IPython
        else:
            return False  # Altro ambiente
    except NameError:
        return False      # Non eseguito in IPython
# --- SINGLE RUN --- 

def param_from_str(name: str) -> ArgParam:
    try:               return           ArgParam.from_str(name)
    except ValueError: return ExperimentArgParams.from_str(name)

def convert_argparams_to_dict(argparams_dict):
    return {
        param.name: value
        for param, value in argparams_dict.items()
    }
def update_argparams(argparams_dict: Dict[ArgParam, Any], updates: Dict[str, Any]) -> Dict[ArgParam, Any]:
    for param in argparams_dict.keys():
        if param.name in updates:
            argparams_dict[param] = updates[param.name]
    return argparams_dict

def run_single(
    args_conf : ParamConfig,
    exp_type  : Type[Experiment],
    changes   : Dict[str, Any] = {}
):
    if in_notebook():
        cmd_conf = convert_argparams_to_dict(args_conf)
    else:
        # Parse cmd arguments
        parser   = ArgParams.get_parser(args=list(args_conf.keys()))
        cmd_conf = vars(parser.parse_args())
        cmd_conf = {
            param_from_str(arg_name) : value 
            for arg_name, value in cmd_conf.items() if value
        }

    # Merge configurations
    full_conf = overwrite_dict(args_conf, cmd_conf)
    full_conf = update_argparams(full_conf, changes)
    
    # Rendering
    if full_conf.get(ArgParams.Render, False):
        
        # Hold main display screen reference
        main_screen = DisplayScreen.set_main_screen()

        # Add close screen flag on as the experiment only involves one run
        full_conf[ArgParams.CloseScreen.value] = True
        
    experiment = exp_type.from_config(full_conf)
    experiment.run()
    

# --- MULTI RUN ---

def run_multi(
    args_conf      : ParamConfig,
    exp_type       : Type[Experiment],
    multi_exp_type : Type[MultiExperiment],
):       
    # Pre-parser solo per controllare se c'è un file
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument('--cmd2exec', type=str)
    
    # Ignora errori per argomenti sconosciuti
    initial_args, _ = pre_parser.parse_known_args()
    if initial_args.cmd2exec and initial_args.cmd2exec.endswith('.txt'):
        # Legge gli argomenti dal file
        with open(initial_args.cmd2exec, 'r') as f:
            import shlex
            args_list = shlex.split(f.read().strip())
        # Ora usa il parser normale con gli argomenti dal file
        parser = ArgParams.get_parser(args=list(args_conf.keys()), multirun=True)
        #exclude the python commands
        cmd_conf = vars(parser.parse_args(args_list[2:]))
    else:
        # Comportamento originale
        parser = ArgParams.get_parser(args=list(args_conf.keys()), multirun=True)
        cmd_conf = vars(parser.parse_args())
    
    #ADD PARSING SPACES FROM  low_spaces.txt file
    with open(SPACE_TXT, 'r') as f: spaces_lines = f.readlines()
    for key in ['scr_layers', 'rec_layers']:
        if key not in cmd_conf:
            continue
        tasks = cmd_conf[key].split('#')
        tasks = [re.sub(r's(\d+)', lambda m: spaces_lines[int(m.group(1))].strip(), task) for task in tasks]
        cmd_conf[key] = '#'.join(tasks)

    cmd_conf = {
        param_from_str(arg_name) : value 
        for arg_name, value in cmd_conf.items() if value
    }
    
    experiment = multi_exp_type.from_args(
        arg_conf     = cmd_conf,
        default_conf = args_conf,
        exp_type     = exp_type
    )
    
    experiment.run()



class BaseZdreamMultiExperiment(MultiExperiment):
    ''' Generic class handling different multi-experiment types. '''
    
    def __init__(
        self, 
        experiment:      Type['ZdreamExperiment'], 
        experiment_conf: Dict[ArgParam, List[Parameter]], 
        default_conf:    ParamConfig
    ) -> None:
        
        super().__init__(experiment, experiment_conf, default_conf)
        
        self._Exp: ZdreamExperiment = cast(ZdreamExperiment, self._Exp)
        
    @property
    def _logger_type(self) -> Type[Logger]: return LoguruLogger

    def _get_display_screens(self) -> List[DisplayScreen]:

        # Screen for synthetic images
        screens = [ 
            DisplayScreen(
                title=self._Exp.GEN_IMG_SCREEN,                 # type: ignore
                display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE
            )
        ]

        # Add screen for natural images if at least one will use it
        use_nat = any(
            parse_boolean_string(str(conf[ExperimentArgParams.Template.value])).count(False) > 0 
            for conf in self._search_config
        )
        
        if use_nat:
            screens.append(
                DisplayScreen(
                    title=self._Exp.NAT_IMG_SCREEN,                 # type: ignore
                    display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE
                )
            )

        return screens
    
# --- MISC --- 
    
def make_dir(path: str, logger: Logger = SilentLogger()) -> str:
    
    logger.info(f'Creating directory: {path}')
    os.makedirs(path, exist_ok=True)
    
    return path


def ref_code_recovery(reference_file: dict, 
                      keys: dict, 
                      ref_file_name: str = 'unspecified'):
    """
    Code to retrieve reference data from the reference file, a nested dictionary containing the
    various references computed with XDREAM.
    :param reference_file: dictionary containing the reference data. It is a nested dictionary with
        keys organized as follows:
        - reference
        - network
        - generator (gen_var)
        - layer
        - neuron (written as '[number]')
        - seed (NOTE: this is an int)
        - code.
    :param keys: dictionary containing the keys to access the reference data. 
        It should NOT contain the key 'reference', but should contain, if needed the key 'code' in
        a dummy form, i.e. 'code': 'code'.
    
    """
    #identify keys before and after gen
    idx_gen = list(keys.keys()).index('gen_var')
    pre_gen_keys = ['reference']+list(keys.values())[:idx_gen+1]
    post_gen_keys = list(keys.values())[idx_gen+1:]
    
    ly_type_nr = '_'.join(keys['layer'].split('_')[1:])
    ref_till_gen = deep_get(dictionary= reference_file, keys = pre_gen_keys)
    try: #Normal access to code
        ref_code = deep_get(dictionary= ref_till_gen, keys = post_gen_keys)
    except KeyError: #Access to code with outdated layer name
        #All layer nomenclatures share the layer typing and number associated to layer type
        #So, let's check if the queried layer shares layer type number with any of 
        # the keys in the reference file
        if ly_type_nr in ['_'.join(k.split('_')[1:]) for k in ref_till_gen.keys()]:
            #find the outdated layer name for the queried layer
            lname_idx = ['_'.join(k.split('_')[1:]) for k in ref_till_gen.keys()].index('_'.join(keys['layer'].split('_')[1:]))
            lname_alt = list(ref_till_gen.keys())[lname_idx]
            old_ly = keys['layer']
            keys['layer'] = lname_alt
            post_gen_keys = list(keys.values())[idx_gen+1:]
            ref_code = deep_get(dictionary= ref_till_gen, keys = post_gen_keys)

            print(f'Layer {old_ly} not found, using {lname_alt} instead')
        else:
            raise ValueError(f'No reference found for gen_variant {keys["gen_var"]}, layer {keys["layer"]}, neuron {keys["neuron"]}, seed {keys["seed"]} in file {ref_file_name}')
        
    return ref_code