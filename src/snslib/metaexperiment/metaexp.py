from dataclasses import dataclass
import os
import pickle
from typing import Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pprint
import argparse

from experiment.AdversarialAttack_BMM.plots import pf1_fromPKL, plot_metaexp_p1
from metaexperiment.metaexp_functs import get_df_summary
from pxdream.utils.io_ import load_pickle, read_json
from tqdm import tqdm

from pxdream.utils.misc import get_max_depth

HYPERPARAMS_FP = '/home/ltausani/Documents/GitHub/ZXDREAM/metaexperiment/hyperparams_meta_an.json'
def agg_stats(x: pd.Series) -> dict:
    """
    Calculates the mean and standard error of the mean (SEM) for a series of data.
    :param x: The data series for which to calculate the statistics.
    :type x: pandas.Series
    :return: A dictionary containing the mean and SEM of the data.
    :rtype: dict
    """
    if x.dtype in ['float64', 'int64']:
        return {
            'mean': abs(x.mean()),
            'std': x.std(),
            'sem': x.std()/np.sqrt(len(x))
        }
    return {
        'mean': x.iloc[0],
    }


def merge_data_pkl(dict_list: list[dict]) -> dict:
    """Merge a list of dictionaries preserving original data types for each key."""
    if not dict_list:
        return {}
    if len(dict_list) == 1:
        return dict_list[0]
        
    result = dict_list[0]
    for next_dict in dict_list[1:]:
        merged = {}
        common_keys = set(result.keys()) & set(next_dict.keys())
        
        for key in common_keys:
            val1, val2 = result[key], next_dict[key]
            
            if isinstance(val1, list):
                merged[key] = val1 + val2
            elif isinstance(val1, str):
                merged[key] = val1 + "\n" + val2
            else:
                raise TypeError(f"Unsupported type {type(val1)} for key {key}")
                
        result = merged
        
    return result

@dataclass
class SnS_metadata:
    """Hierarchical structure for neural network experiments data (with 'layer' as an additional level)."""
    data: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]]]
    
    @classmethod
    def from_json(cls, json_path: str, recalculate: bool = False) -> 'SnS_metadata':
        
        """
        this method constructs a tree structure from the multiexperiments' json files.
        Every last parent node is a multiexperiment, the leafs are:
        - path to results
        - dataframe summary
        - label (a string containing the name of every traversed node in the tree)        
        """
        
        
        #load the json file containing the path to the datafolders
        with open(json_path, 'r') as f:
            paths_tree = json.load(f)
        data = {}
        # Loop through the new tree structure: nn_type -> upper_ly -> model_type -> layer -> iterations -> constraint
        for nn_type, nn_data in tqdm(paths_tree.items(), desc="Neural Network Types"):
            data[nn_type] = {}
            for upper_ly, upper_data in tqdm(nn_data.items(), desc="Upper Level", leave=False):
                data[nn_type][upper_ly] = {}
                for model_type, model_data in tqdm(upper_data.items(), desc="Model Types", leave=False):
                    data[nn_type][upper_ly][model_type] = {}
                    for layer, layer_data in tqdm(model_data.items(), desc="Layers", leave=False):
                        data[nn_type][upper_ly][model_type][layer] = {}
                        for iterations, iter_data in tqdm(layer_data.items(), desc="Iterations", leave=False):
                            data[nn_type][upper_ly][model_type][layer][iterations] = {}
                            for constraint, dir_paths in tqdm(iter_data.items(), desc="Constraints", leave=False):
                                if isinstance(dir_paths, str): dir_paths = [dir_paths]
                                data[nn_type][upper_ly][model_type][layer][iterations][constraint] = {}
                                data[nn_type][upper_ly][model_type][layer][iterations][constraint]['path'] = dir_paths
                                data[nn_type][upper_ly][model_type][layer][iterations][constraint]['label'] = '#'.join([nn_type, upper_ly, model_type, layer, iterations, constraint])
                                dfs = []
                                #data_pkl = []
                                #store the df summary in the metadata
                                is_GD_exp = os.path.dirname(dir_paths[0]).split(os.sep)[-1] == 'SnS_gradient_based'
                                for dir_path in tqdm(dir_paths, desc="Directory Paths", leave=False):
                                    try:
                                        if not recalculate:
                                            try:
                                                df = pd.read_json(os.path.join(dir_path, 'data_summary.json'))
                                            except:
                                                df = get_df_summary(load_pickle(os.path.join(dir_path, 'data.pkl')), savepath=dir_path, is_GD_exp=is_GD_exp)
                                        else:
                                            df = get_df_summary(load_pickle(os.path.join(dir_path, 'data.pkl')), savepath=dir_path, is_GD_exp=is_GD_exp)
                                        dfs.append(df)
                                        #data_pkl.append(load_pickle(os.path.join(dir_path, 'data.pkl')))
                                    except (NotADirectoryError, FileNotFoundError) as e:
                                        print(f"Warning: {e}")
                                        continue
                                if dfs:
                                    data[nn_type][upper_ly][model_type][layer][iterations][constraint]['df'] = pd.concat(dfs, ignore_index=True)
                                # (Optionally add splines code here as in your original version)
        return cls(data)

    def update_from_json(self, json_path: str, recalculate: bool = False) -> None:

        
        """
        Updates the internal data structure from a JSON file containing metadata paths and recalculates summaries if needed.
        Args:
            json_path (str): Path to the JSON file containing the metadata paths tree.
            recalculate (bool, optional): If True, recalculates data summaries even if they already exist. Defaults to False.
        Raises:
            NotADirectoryError: If a specified directory path is invalid.
            FileNotFoundError: If a required file is not found.
        Notes:
            - The JSON file should have a hierarchical structure with nested dictionaries representing neural network types,
              upper levels, model types, layers, iterations, and constraints.
            - For each constraint, the method processes directory paths, loads data summaries, and stores them in the internal
              data structure.
            - If recalculation is enabled, existing summaries are overwritten by recalculated ones.
            - Warnings are printed for missing or invalid directories/files.    
        """
        
        with open(json_path, 'r') as f:
            paths_tree = json.load(f)
        
        for nn_type, nn_data in tqdm(paths_tree.items(), desc="Neural Network Types"):
            if nn_type not in self.data:
                self.data[nn_type] = {}
            for upper_ly, upper_data in tqdm(nn_data.items(), desc="Upper Level", leave=False):
                if upper_ly not in self.data[nn_type]:
                    self.data[nn_type][upper_ly] = {}
                for model_type, model_data in tqdm(upper_data.items(), desc="Model Types", leave=False):
                    if model_type not in self.data[nn_type][upper_ly]:
                        self.data[nn_type][upper_ly][model_type] = {}
                    for layer, layer_data in tqdm(model_data.items(), desc="Layers", leave=False):
                        if layer not in self.data[nn_type][upper_ly][model_type]:
                            self.data[nn_type][upper_ly][model_type][layer] = {}
                        for iterations, iter_data in tqdm(layer_data.items(), desc="Iterations", leave=False):
                            if iterations not in self.data[nn_type][upper_ly][model_type][layer]:
                                self.data[nn_type][upper_ly][model_type][layer][iterations] = {}
                            for constraint, dir_paths in tqdm(iter_data.items(), desc="Constraints", leave=False):
                                if constraint in self.data[nn_type][upper_ly][model_type][layer][iterations]:
                                    continue
                                if isinstance(dir_paths, str):
                                    dir_paths = [dir_paths]
                                self.data[nn_type][upper_ly][model_type][layer][iterations][constraint] = {}
                                self.data[nn_type][upper_ly][model_type][layer][iterations][constraint]['path'] = dir_paths
                                self.data[nn_type][upper_ly][model_type][layer][iterations][constraint]['label'] = '#'.join([nn_type, upper_ly, model_type, layer, iterations, constraint])
                                dfs = []
                                #data_pkl = []
                                #store the df summary in the metadata
                                is_GD_exp = os.path.dirname(dir_paths[0]).split(os.sep)[-1] == 'SnS_gradient_based'
                                for dir_path in tqdm(dir_paths, desc="Directory Paths", leave=False):
                                    try:
                                        if not recalculate:
                                            try:
                                                df = pd.read_json(os.path.join(dir_path, 'data_summary.json'))
                                            except:
                                                df = get_df_summary(load_pickle(os.path.join(dir_path, 'data.pkl')), savepath=dir_path, is_GD_exp=is_GD_exp)
                                        else:
                                            df = get_df_summary(load_pickle(os.path.join(dir_path, 'data.pkl')), savepath=dir_path, is_GD_exp=is_GD_exp)
                                        dfs.append(df)
                                        #data_pkl.append(load_pickle(os.path.join(dir_path, 'data.pkl')))
                                    except (NotADirectoryError, FileNotFoundError) as e:
                                        print(f"Warning: {e}")
                                        continue
                                if dfs:
                                    self.data[nn_type][upper_ly][model_type][layer][iterations][constraint]['df'] = pd.concat(dfs, ignore_index=True)

    
    def get_experiments(self, queries: list[list[str]]|None = None):
        """get experiments of interest from the metadata tree

        :param queries: queries for getting your metadata, defaults to None
        :type queries: list[list[str]] | None, optional
        :return: a dictionary containing the leaves of the tree, i.e. the experiments of interest
        :rtype: _type_
        """
        if queries is None: queries = [[]]
        data_of_interest = {}
        # Collect all experiments under the partial path for each query q
        for q in queries:
            if len(q) < get_max_depth(self.data):
                # if the query is none, all nn_types are selected
                nn_list = [q[0]] if len(q) > 0 else list(self.data.keys())
                for nn_type in nn_list:
                    upper_list = [q[1]] if len(q) > 1 else list(self.data[nn_type].keys())
                    for upper_ly in upper_list:
                        model_list = [q[2]] if len(q) > 2 else list(self.data[nn_type][upper_ly].keys())
                        for model_type in model_list:
                            layer_list = [q[3]] if len(q) > 3 else list(self.data[nn_type][upper_ly][model_type].keys())
                            for layer in layer_list:
                                it_list = [q[4]] if len(q) > 4 else list(self.data[nn_type][upper_ly][model_type][layer].keys())
                                for it in it_list:
                                    for c, experiment in self.data[nn_type][upper_ly][model_type][layer][it].items():
                                        if 'df' in experiment:
                                            data_of_interest.update({experiment.get('label', ''): {'df': experiment['df'], 'path': experiment['path'], 'label': experiment.get('label', '')}})
            else:
                nn_type, upper_ly, model_type, layer, iterations, constraint = q
                exp = self.data[nn_type][upper_ly][model_type][layer][iterations][constraint]
                data_of_interest.update({exp['label']: {'df': exp['df'], 'path': exp['path'], 'label': exp['label']}})
        return data_of_interest


    def apply_analysis(self,
                    callables: list[callable], 
                    queries: list[list[str]]|None = None,
                    hyperparams_fp: str = HYPERPARAMS_FP) :
        """_summary_

        :param callables: _description_
        :type callables: list[callable]
        :param hyperparams_fp: _description_, defaults to HYPERPARAMS_FP
        :type hyperparams_fp: str, optional
        """
        exps = self.get_experiments(queries = queries)
        hyperparams = read_json(hyperparams_fp)
        savepath = None
        if "savepath" in hyperparams:
            savepath = hyperparams.pop("savepath")
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
        if "plotting" in hyperparams:
            plotting = hyperparams.pop("plotting")
            for k in plotting.keys():
                if k in exps:
                    exps[k].update(plotting.get(k, {}))
        
        for func in callables:
            func_params = hyperparams.get(func.__name__, {})
            if not isinstance(func_params, list): func_params = [func_params]
            for fp in func_params:
                if "savepath" in func.__code__.co_varnames and savepath is not None:
                    fp["savepath"] = savepath
                func(exps = exps, **fp)

    def save_pkl(self, output_path: Union[str, Path], protocol: Optional[int] = None) -> None:
        output_path = Path(output_path)
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.data, f, protocol=protocol)
        except Exception as e:
            raise IOError(f"Error saving metadata to {output_path}: {str(e)}")

    @classmethod
    def load_pkl(cls, input_path: Union[str, Path]) -> 'SnS_metadata':
        input_path = Path(input_path)
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            return cls(data=data)
        except Exception as e:
            raise IOError(f"Error loading metadata from {input_path}: {str(e)}")


    @property
    def tree_structure(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]]]:
        tree = {}
        for nn_type, nn_data in self.data.items():
            tree[nn_type] = {}
            for upper_ly, upper_data in nn_data.items():
                tree[nn_type][upper_ly] = {}
                for model_type, model_data in upper_data.items():
                    tree[nn_type][upper_ly][model_type] = {}
                    for layer, layer_data in model_data.items():
                        tree[nn_type][upper_ly][model_type][layer] = {}
                        for iterations, iter_data in layer_data.items():
                            tree[nn_type][upper_ly][model_type][layer][iterations] = list(iter_data.keys())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(tree)
        return tree
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and save SnS metadata.")
    parser.add_argument("--json_path", type=str, default = "SnS_multiexp_dirs.json", help="Path to the JSON file containing the metadata paths.")
    parser.add_argument("--output_path", type=str, default = "metaexp.pkl", help="Path to save the output pickle file.")
    
    args = parser.parse_args()
    
    # Load metadata from JSON
    metadata = SnS_metadata.from_json(args.json_path)
    
    # Save metadata to pickle file
    metadata.save_pkl(args.output_path)