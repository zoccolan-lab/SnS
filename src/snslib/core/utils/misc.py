'''
This is a general purpose file containing utility functions that are used across the entire Zdream framework.
'''

from collections import defaultdict
from copy import deepcopy
import platform
from os import path
from subprocess import PIPE, Popen
from typing import Tuple, TypeVar, Callable, Dict, List, Any, Union, cast
from scipy.spatial.distance import pdist

from functools import partial, reduce
from operator import getitem

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from pandas import DataFrame
from numpy.typing import NDArray
import pandas as pd

from .logger import Logger, SilentLogger



from .types import RFBox
import subprocess

# --- TYPING ---

# Type generics
T = TypeVar('T')
D = TypeVar('D')

# Default for None with value
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

# Default for None with producer function
def lazydefault(var : T | None, expr : Callable[[], D]) -> T | D:
    return expr() if var is None else var

# --- NUMPY ---

def fit_bbox(
    data : NDArray | None,
    axes : Tuple[int, ...] = (-2, -1)
) -> RFBox:
    '''
    Fit a bounding box for non-zero entries of a
    numpy array along provided directions.
    
    :param grad: Array representing the data we want
        to draw bounding box over. Typical use case
        is the computed (input-)gradient of a subject
        given some hidden units activation states
    :type grad: Numpy array or None
    :param axes: Array dimension along which bounding
        boxes should be computed
    :type axes: Tuple of ints
    
    :returns: Computed bounding box in the format:
        (x1_min, x1_max, x2_min, x2_max, ...)
    :rtype: Tuple of ints
    '''
    if data is None: return 0, 0, 0, 0  # type: ignore
    
    # Get non-zero coordinates of gradient    
    coords = data.nonzero() 
    
    bbox = []
    # Loop over the spatial coordinates
    for axis in axes:
        bbox.append((coords[axis].min(), coords[axis].max() + 1))
        
    return tuple(bbox)


def to_numpy(data: List | Tuple | Tensor | DataFrame) -> NDArray:
    '''
    Convert data from different formats into a numpy NDArray.
    
    :param data: Data structure to convert into a numpy array.
    :type data: List | Tuple | Tensor | DataFrame
    
    :return: Data converted into a numpy array.
    :rtype: NDArray
    '''

    try:
        if isinstance(data, DataFrame):
            numpy_array = data.to_numpy()
        elif isinstance(data, Tensor):
            numpy_array = data.numpy()
        elif isinstance(data, List) or isinstance(data, Tuple):
            numpy_array = np.array(data)
        else:
            raise TypeError(f'Invalid input type {type(data)} for array conversion. ')
        return numpy_array
    
    except Exception as e:
        raise RuntimeError(f"Error during numpy array conversion: {e}")

# --- TORCH ---

# Default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputLayer(nn.Module):
    ''' Class representing a trivial input layer for an ANN '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x : Tensor) -> Tensor:
        return x

    def _get_name(self) -> str:
        return 'Input'

def unpack(model : nn.Module) -> nn.ModuleList:
    '''
    Utils function to extract the layer hierarchy from a torch Module.
    This function recursively inspects each module children and progressively 
    build the hierarchy of layers that is then return to the user.
    
    :param model: Torch model whose hierarchy we want to unpack.
    :type model: torch.nn.Module
    
    :returns: List of sub-modules (layers) that compose the model hierarchy.
    :rtype: nn.ModuleList
    '''
    
    children = [unpack(children) for children in model.children()]
    unpacked = [model] if list(model.children()) == [] else []

    for c in children: unpacked.extend(c)
    
    return nn.ModuleList(unpacked)

# NOTE: Code taken from github issue: https://github.com/pytorch/vision/issues/6699
def replace_inplace(module : nn.Module) -> None:
    '''
    Recursively replaces instances of nn.ReLU and nn.ReLU6 modules within a given
    nn.Module with instances of nn.ReLU with inplace=False.

    :param module: The PyTorch module whose ReLU modules need to be replaced.
    :type module: nn.Module
    '''

    reassign = {}
    
    for name, mod in module.named_children(): 

        replace_inplace(mod) 

        # NOTE: Checking for explicit type instead of instance 
        #       as we only want to replace modules of the exact type 
        #       not inherited classes 
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6: 
            reassign[name] = nn.ReLU(inplace=False) 

    for key, value in reassign.items(): 
        module._modules[key] = value 

# --- IMAGES ---

def preprocess_image(image_fp: str, resize: Tuple[int, int] | None)  -> NDArray:
    '''
    Preprocess an input image by resizing and batching.

    :param image_fp: The file path to the input image.
    :type image_fp: str
    :param resize: Optional parameter to resize the image to the specified dimensions, defaults to None.
    :type resize: Tuple[int, int] | None
    :return: The preprocessed image as a NumPy array.
    :rtype: NDArray
    '''
    
    # Load image and convert to three channels
    img = Image.open(image_fp).convert("RGB")

    # Optional resizing
    if resize:
        img = img.resize(resize)
    
    # Array shape conversion
    img_arr = np.asarray(img) / 255.
    img_arr = rearrange(img_arr, 'h w c -> 1 c h w')
    
    return img_arr

def concatenate_images(img_list: Tensor | List[Tensor], nrow: int = 2) -> Image.Image:
    ''' 
    Concatenate an input number of images as tensors into a single image
    with the specified number of rows.
    NOTE: nrow is the number of COLUMNS, not rows.
    '''
    
    grid_images = make_grid(img_list, nrow=nrow)
    grid_images = to_pil_image(grid_images)
    grid_images = cast(Image.Image, grid_images)
    
    return grid_images

# --- STATISTICS

def SEM(
        data: List[float] | Tuple[float] | NDArray, 
        axis: int = 0
    ) -> NDArray:
    '''
    Compute standard error of the mean (SEM) for a sequence of numbers

    :param data: Data to which compute the statistics.
    :type data: List[float] | Tuple[float]
    :param axis: Axis to compute SEM (valid for NDArray data only), default to zero.
    :type axis: int
    :return: Computed statistics SEMs
    :rtype: NDArray
    '''

    # Convert data into NDArray
    if not isinstance(data, np.ndarray):
        data = to_numpy(data)
    
    # Compute standard deviation and sample size on the specified axis
    std_dev = np.nanstd(data, axis=axis) if data.ndim > 1 else np.nanstd(data)
    sample_size = data.shape[axis]
    
    # Compute SEM
    sem = std_dev / np.sqrt(sample_size)

    return sem


def harmonic_mean(a: float, b: float) -> float:
    '''
    Compute the harmonic mean of two given numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: Harmonic mean of the two numbers.
    :rtype: float
    '''
    
    
    return  2 / (1 / a + 1 / b) 


def growth_scale(start=0, step=0.05, growth_factor=1.5):
    '''
    Generate an infinite scale of growth sequence using a generator.

    :param start: Starting value of the sequence, defaults to 0.
    :type start: float, optional
    :param step: Step size between terms, defaults to 0.05.
    :type step: float, optional
    :param growth_factor: Growth factor between terms, defaults to 1.5.
    :type growth_factor: float, optional
    :yield: Next value in the growth scale sequence.
    :rtype: float
    '''
    
    current_value = start
    yield current_value
    while True:
        current_value += step
        step *= growth_factor
        yield current_value


# --- TIME ---

def stringfy_time(sec: int | float) -> str:
    ''' Converts number of seconds into a hour-minute-second string representation. '''

    # Round seconds
    sec = int(sec)

    # Compute hours, minutes and seconds
    hours = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60

    # Handle possible formats
    time_str = ""
    if hours > 0:
        time_str += f"{hours} hour{'s' if hours > 1 else ''}, "
    if minutes > 0:
        time_str += f"{minutes} minute{'s' if minutes > 1 else ''}, "
    time_str += f"{sec} second{'s' if sec > 1 else ''}"
    
    return time_str

# --- DICTIONARIES ---

def overwrite_dict(a: Dict, b: Dict) -> Dict:
    ''' 
    Overwrite keys of a nested dictionary A with those of a 
    second flat dictionary B is their values are not none.
    '''

    def overwrite_dict_aux(a_: Dict, b_: Dict):
    
        for key in a_:
            if isinstance(a_[key], Dict):
                overwrite_dict_aux(a_[key], b_)
            elif key in b_:
                a_[key] = b_[key]
        return a_
    
    # Create new dictionary
    a_copy = deepcopy(a)

    overwrite_dict_aux(a_=a_copy, b_=b)
    
    return a_copy

def flatten_dict(d: Dict)-> Dict:
    '''
    Recursively flat an input nested dictionary.
    '''

    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            nested_flattened = flatten_dict(v)
            for nk, nv in nested_flattened.items():
                flattened_dict[nk] = nv
        else:
            flattened_dict[k] = v
    return flattened_dict

# --- EXECUTION ---

# NOTE: This requires `sudo apt install xsel`
def copy_on_clipboard(command: str):
    ''' Copies input string to clipboard '''
    match platform.system():
        
        case 'Linux':

            # Byte conversion
            cmd_ =  bytes(command, encoding='utf-8')
            
            # Copy
            #p = Popen(['xclip', '-selection', 'clipboard'], stdin=PIPE)
            p = Popen(['xsel', '-bi'], stdin=PIPE)
            p.communicate(input=cmd_)

        case 'Windows':

            subprocess.run(f'echo {command} | clip', shell=True)

        

def copy_exec(
        file: str,
        program: str = 'python',
        args: Dict[str, str] = dict(),
    ) -> str:
    ''' 
    Copies a program execution command line to clipboard given 
    program name, file name and the list of name-value arguments in dictionary form
    It returns the command string
    '''
    
    cmd = f'{program} {file} ' + " ".join(f'--{k} {v}' for k, v in args.items())
    with open('cmd2exec.txt', 'w') as file:
            file.write(cmd)
    copy_on_clipboard(cmd)

    return cmd

def minmax_norm(vector):
    min_val = np.min(vector)
    return (vector - min_val) / (np.max(vector) - min_val)

def defaultdict_list():
    return defaultdict(list)


def load_npy_npz(in_dir:str, fnames:list[tuple[str, str]], logger: Logger = SilentLogger()):
    '''
    Load a .npy/.npz file (e.g. a experiment state) from a folder where the file
    was dumped. It raises a warning for not present states.

    :param in_dir: Directory where states are dumped.
    :type in_dir: str
    :param fnames: List of file names to load, structured as
        tuples (fname, extension).
    :type fnames: list[tuple[str, str]]
    :param logger: Logger to log i/o information. If not specified
        a `SilentLogger` is used. 
    :type logger: Logger | None, optional
    '''
    logger.info(f'Loading experiment state from {in_dir}')

    loaded = dict()
    for name, ext in fnames:

        # File path
        fp = path.join(in_dir, f'{name}.{ext}')

        # Loading function depending on file extension
        match ext:
            case 'npy': load_fun = np.load
            case 'npz': load_fun = lambda x: dict(np.load(x))

        # Loading state
        if path.exists(fp):
            logger.info(f"> Loading {name} history from {fp}")
            loaded[name] = load_fun(fp)

        # Warning if the state is not present
        else:
            logger.warn(f"> Unable to fetch {fp}")
            loaded[name] = None
    return loaded

def resize_image_tensor(img_tensor: Tensor, size: Tuple[int, int]) -> Tensor:
    resized_img_tensor = F.interpolate(img_tensor, size=(size[-2], size[-1]),
    mode='bilinear', align_corners=False).view(img_tensor.shape[0],-1).cpu().numpy().astype('float32')
    return resized_img_tensor


def deep_get(dictionary: dict, keys: list[str]) -> any:
    """
    Efficiently access nested dictionary values using a sequence of keys.

    :param dictionary: The nested dictionary to traverse
    :type dictionary: dict
    :param keys: Sequence of keys to access nested values
    :type keys: list[str]
    :return: The value at the specified nested path
    :rtype: any
    
    Details:
        - Uses reduce() to fold the sequence of getitem operations
        - getitem(d,k) is equivalent to d[k] but as a function
        - reduce applies getitem sequentially: 
          dict[k1][k2][k3] becomes reduce(getitem, [k1,k2,k3], dict)
    
    Example:
        data = {'a': {'b': {'c': 1}}}
        deep_get(data, ['a','b','c']) # returns 1
    """
    return reduce(getitem, keys, dictionary)

def aggregate_matrix(matrix: NDArray, 
                    row_aggregator: Callable[[NDArray], NDArray] = np.max, 
                    final_aggregator: Callable[[NDArray], NDArray | float] | None = None) -> NDArray | float:
    """
    Aggregates a 2D NumPy matrix using two aggregation functions.

    Parameters:
    matrix (NDArray): 2D NumPy array to aggregate.
    row_aggregator (Callable[[NDArray], NDArray]): Function to aggregate within each row.
    final_aggregator (Callable[[NDArray], NDArray | float] | None): Function to aggregate the results of the row aggregation.

    Returns:
    NDArray | float: The final aggregated result.
    """
    # Apply the row aggregator to each row
    row_results = row_aggregator(matrix)
    
    # Apply the final aggregator to the results of the row aggregation
    if final_aggregator is None:
        final_result = row_results
    else:   
        final_result = final_aggregator(row_results)
    
    return final_result

def aggregate_df(df: pd.DataFrame,
                 f_aggr_single_cell: callable = np.mean,
                 f_aggr_betw_cells: callable = partial(np.mean,axis=0)) -> pd.DataFrame:
    """
    Aggregates data from multiple cells using two aggregation functions.
    This function applies two levels of aggregation:
    1. Within each cell using f_aggr_single_cell
    2. Between cells using f_aggr_betw_cells
    Args:
        df (pd.DataFrame): Input DataFrame containing data to aggregate
        f_aggr_single_cell (callable, optional): Function to aggregate data within each cell. 
            Defaults to np.mean.
        f_aggr_betw_cells (callable, optional): Function to aggregate data between cells. 
            Defaults to np.mean(axis=1).
    Returns:
        pd.DataFrame: Aggregated DataFrame after applying both levels of aggregation
    Examples:
        >>> df = pd.DataFrame({'A': [[1,2,3], [4,5,6]], 'B': [[7,8,9], [10,11,12]]})
        >>> aggregate_df(df)
        A    4.0
        B   10.0
        dtype: float64
    """
    df = df.applymap(lambda x: f_aggr_single_cell(x))
    return f_aggr_betw_cells(df)

def get_max_depth(d, current_level=0):
    """
    Determina la profondità massima del dizionario
    """
    max_depth = current_level
    
    for k, v in d.items():
        if isinstance(v, dict):
            sub_depth = get_max_depth(v, current_level + 1)
            max_depth = max(max_depth, sub_depth)
    
    return max_depth

def get_keys_at_level(d, level=0):
    """
    Ottiene tutte le chiavi al livello gerarchico specificato,
    supportando indici negativi come in Python (-1 = ultimo livello, ecc.)
    
    Args:
        d (dict): Dizionario da analizzare
        level (int): Livello gerarchico delle chiavi desiderato
    
    Returns:
        list: Lista di chiavi al livello specificato
    """
    # Prima, trova la profondità massima
    max_depth = get_max_depth(d)
    
    # Gestisci indici negativi (come in Python)
    if level < 0:
        level = max_depth + level + 1  # +1 perché max_depth è 0-indexed
    
    # Assicurati che level sia valido
    if level < 0 or level > max_depth:
        return []
    
    # Funzione interna per ottenere le chiavi al livello specificato
    def _get_keys(d, target_level, current_level=0):
        if current_level == target_level:
            return list(d.keys())
        
        result = []
        for k, v in d.items():
            if isinstance(v, dict):
                result.extend(_get_keys(v, target_level, current_level + 1))
        
        return result
    
    return _get_keys(d, level)

def deep_update(refs, refs_update):
    """
    Recursively merge two dictionaries by updating 'refs' with values from 'refs_update'.
    
    For every key in 'refs_update':
      - If the key does not exist in 'refs', the key-value pair is added.
      - If the key exists in both dictionaries and both values are dicts, the function is called recursively
        to merge the nested dictionaries.
      - If the key exists in both and at least one corresponding value is not a dict, 
        the value from 'refs_update' overrides the one in 'refs'.
    
    This function modifies the 'refs' dictionary in-place and also returns it.
    
    Parameters:
    -----------
    refs : dict
        The dictionary to be updated.
    refs_update : dict
        The dictionary whose values will be merged into 'refs'.
    
    Returns:
    --------
    dict
        The updated 'refs' dictionary.
    
    Example:
    --------
    >>> refs = {'a': 1, 'b': {'c': 3}}
    >>> refs_update = {'b': {'d': 4}, 'e': 5}
    >>> updated_refs = deep_update(refs, refs_update)
    >>> print(updated_refs)
    {'a': 1, 'b': {'c': 3, 'd': 4}, 'e': 5}
    """
    for key, upd_val in refs_update.items():
        # If the key doesn't exist in refs, add it.
        if key not in refs:
            refs[key] = upd_val
        else:
            # If both values are dictionaries, update recursively.
            if isinstance(refs[key], dict) and isinstance(upd_val, dict):
                deep_update(refs[key], upd_val)
            else:
                # Optionally, override the value in refs.
                refs[key] = upd_val
    return refs