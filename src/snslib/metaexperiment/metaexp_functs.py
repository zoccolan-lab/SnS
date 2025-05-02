from collections import defaultdict
from functools import partial
import os
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch
from experiment.utils.args import NATURAL_RECORDINGS, WEIGHTS
from experiment.utils.misc import ref_code_recovery
from pxdream.generator import DeePSiMGenerator
from pxdream.utils.io_ import load_pickle
from pxdream.utils.misc import aggregate_matrix, deep_get, resize_image_tensor

NAT_STAT_AGGREGATOR = {
    'max': partial(aggregate_matrix, row_aggregator = partial(np.max, axis = 1)),
    'percentile_99': partial(aggregate_matrix, row_aggregator = partial(np.percentile, q = 99, axis = 1)),
    'min': partial(aggregate_matrix, row_aggregator = partial(np.min, axis = 1)),
    'mean': partial(aggregate_matrix, row_aggregator = partial(np.mean, axis = 1)),
    'median': partial(aggregate_matrix, row_aggregator = partial(np.median, axis = 1)),
    'percentile_10': partial(aggregate_matrix, row_aggregator = partial(np.percentile, q = 10, axis = 1)),
    'percentile_8': partial(aggregate_matrix, row_aggregator = partial(np.percentile, q = 8, axis = 1))
}

STOPPING_CRITERIA = {
    "invariance": lambda x,thr: x > thr,
    "adversarial": lambda x,thr: x <= thr
}

NAT_LABELS = '/home/ltausani/Desktop/Zout/NeuralRecording/neurec_rnet50_robustL2_rout_and_lbls/neurec_rnet50_robustL2_rout_and_lbls-0/labels.npy'


def set_stopping(exp):
    
    #get the dataframe and load the .pkl file
    df = exp['df']
    df= collect_natstats(exp)
    data_pkl = load_pickle(os.path.join(exp['path'][0], 'data.pkl'))
    
    d_new = {
        'stop': [],
        'p1_stop': [],
        'dist_up_stop': [],
        'dist_low_stop': []
    }
    for i,row in df.iterrows():   
        stop_crit = STOPPING_CRITERIA[row['task']]
        trgt_ly = row['upper_ly']
        trgt_activ = data_pkl['reference_activ'][i] + data_pkl['layer_scores'][i][trgt_ly]
        lower_ly = row['lower_ly']
        nat_thresh = row['nat_thresh']
        median_a_pgen = NAT_STAT_AGGREGATOR['percentile_8'](trgt_activ)
        compare_with_thresh = stop_crit(median_a_pgen,nat_thresh)
        try:
            t_i = np.where(compare_with_thresh[:-1] != compare_with_thresh[1:])[0][0]
            stop_value = t_i if compare_with_thresh[t_i] else t_i + 1
            p1_stop_value = data_pkl['p1_front'][i][np.where(data_pkl['p1_front'][i][:, 0] <= stop_value)[0][-1]]
            dist_up_stop_value = data_pkl['layer_scores'][i][trgt_ly][p1_stop_value[0]][p1_stop_value[1]]
            dist_low_stop_value = data_pkl['layer_scores'][i][lower_ly][p1_stop_value[0]][p1_stop_value[1]]
            
            d_new['stop'].append(stop_value)
            d_new['p1_stop'].append(p1_stop_value)
            d_new['dist_up_stop'].append(dist_up_stop_value)
            d_new['dist_low_stop'].append(dist_low_stop_value)
        except:
            d_new['stop'].append(np.nan)
            d_new['p1_stop'].append((np.nan, np.nan))
            d_new['dist_up_stop'].append(np.nan)
            d_new['dist_low_stop'].append(np.nan)
    exp['df'] = df = pd.concat([df, pd.DataFrame(d_new)], axis=1)        
    return df


def collect_natstats(exp,
                    nat_rec_fp: str = NATURAL_RECORDINGS,
                    other_nat_stats: list[str] = ['mean'],
                    save: bool = False) -> pd.DataFrame:
    """
    Funzione mock per la raccolta delle statistiche naturali.
    
    Args:
        df: Il DataFrame dell'esperimento.
        d: Il dizionario ottenuto dal pickle (può essere None se il caricamento fallisce).
        
    Returns:
        Un DataFrame con le statistiche naturali.
    """
    df = exp['df']
    #load the natural recordings
    nrec = load_pickle(nat_rec_fp)
    #get the info needed from the dataframe
    tasks =df['task'].unique()
    net_sbj = df['net_sbj'].unique()[0]
    net_type = 'robust_l2' if df['robust'].unique()[0] else 'vanilla'
    up_ly = '_'.join(df['upper_ly'].unique()[0].split('_')[1:])
    up_ly = [i for i in deep_get(dictionary=nrec, keys=[net_sbj, net_type]).keys() if up_ly in i]
    assert  len(up_ly) == 1, f"Upper layer not found in the natural recordings"
    up_ly = up_ly[0]
    
    for t in tasks:
        unique_units = df[df['task'] == t]['high_target'].unique()
        nat_aggr = NAT_STAT_AGGREGATOR['max' if t == 'invariance' else 'min']
        nat_thresh = nat_aggr(deep_get(dictionary= nrec, keys=[net_sbj, net_type, up_ly]))[unique_units]
        mapping_dict = dict(zip(unique_units, nat_thresh))
        df.loc[df['task'] == t, 'nat_thresh'] = df.loc[df['task'] == t, 'high_target'].map(mapping_dict)
    
    for ost in other_nat_stats:
        unique_units = df['high_target'].unique()
        nat_aggr = NAT_STAT_AGGREGATOR[ost]
        nat_thresh = nat_aggr(deep_get(dictionary= nrec, keys=[net_sbj, net_type, up_ly]))[unique_units]
        mapping_dict = dict(zip(unique_units, nat_thresh))
        df[f'nat_{ost}'] = df['high_target'].map(mapping_dict)
    
    exp['df'] = df
    if save:
        df.to_csv(os.path.join(exp['path'][0], 'data_summary.csv'))  
    
    
    return df


def compute_max_pix_dist(SnS_mexp_data):

    """
    Computes the maximum pixel-wise distance for a set of reference images generated
    from the provided experimental data.

    This function processes reference information, 
    generates corresponding images using a pre-trained generator, and calculates 
    the maximum pixel-wise distance for each generated image.

    :param SnS_mexp_data: A dictionary containing experimental data with the following keys:
        - 'reference_info': List of dictionaries with reference file information.
        - 'net_sbj': List of network subject identifiers.
        - 'robust': List of boolean values indicating robustness.
        - 'upper_ly': List of upper layer identifiers.
    :type SnS_mexp_data: dict

    :return: A NumPy array containing the maximum pixel-wise distances for each reference image.
    :rtype: numpy.ndarray
    """
    #get the codes for reference images
    references =[]
    for ref_i,ns,is_r, up_ly in zip(SnS_mexp_data['reference_info'],SnS_mexp_data['net_sbj'],SnS_mexp_data['robust'], SnS_mexp_data['upper_ly']):
        ref_f = load_pickle(ref_i['ref_file'])
        ref_code = ref_code_recovery(reference_file = ref_f, 
                    keys = {'network': ns+'_r' if is_r else ns, 
                            'gen_var': ref_i['gen_var'], 
                            'layer': up_ly,
                            'neuron': ref_i['neuron'],
                            'seed': ref_i['seed'],
                            'code':'code'}, 
                    ref_file_name = ref_i['ref_file'])         
        references.append(ref_code)
    references = np.vstack(references)
    generator = DeePSiMGenerator(
        root    = WEIGHTS,
        variant = ref_i['gen_var']
    ).to('cuda')
    #generate the reference images from the codes
    ref_imgs = generator(references)
    ref_imgs = torch.from_numpy(resize_image_tensor(ref_imgs, (224,224)))
    #get the maximum pixel-wise distance for each image, that is the distance between the reference image and its complement 1- ref_i
    max_pix_dist =np.array([torch.norm(torch.max(ref_i, 1 - ref_i)).detach().cpu().numpy()
                        for ref_i in ref_imgs])
    return max_pix_dist


def get_df_summary(SnS_mexp_data: Dict[str, Any], 
                savepath: str | None = None,
                is_GD_exp: bool = False) -> pd.DataFrame:
    
    """
    Generate a summary DataFrame from a SnS multiexperiment.
    This function processes the raw experiment data dictionary into a structured DataFrame,
    extracting and transforming key metrics and results. 
    Parameters:
    -----------
    SnS_mexp_data : Dict[str, Any]
        Dictionary containing multiexperiment data
    savepath : str or None, optional (default=None)
        Path where the summary DataFrame will be saved as JSON.
        If None, the DataFrame is only returned without saving.
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing processed experiment results 
        
    Notes:
    ------
    - Converts task signatures to human-readable task names ('adversarial' or 'invariance')
    - Extracts reference activations and formats high targets for readability
    - Processes performance metrics from the Pareto front (p1_front)
    - Computes the intersection of important features when multiple methods are used
    - Adds pixel distance metrics when input layer analysis is available
    """
    #convert the task signature to a human readable form
    Sign2Task = {
        1 : 'adversarial',
        -1 : 'invariance'
    }
    
    # Define a function to check if a column is valid
    # (i.e., contains only strings, ints, floats, or booleans)
    def is_valid_column(column):
        valid_types = (str, int, float, bool)
        return all(isinstance(item, valid_types) for item in column)

    # Filter the dictionary to keep only valid columns
    filtered_data = {key: value for key, value in SnS_mexp_data.items() if is_valid_column(value)}
    # Convert the filtered dictionary to a DataFrame
    df = pd.DataFrame(filtered_data)
    # Add to the DataFrame the task signature, reference activations, and high targets
    df['task'] = [Sign2Task[list(x.values())[0]] for x in SnS_mexp_data['task_signature']]
    df['ref_activ'] = [float(x[0][0]) for x in SnS_mexp_data['reference_activ']]
    df['high_target'] = [f"({', '.join(str(int(xi[0])) for xi in x)})" for x in SnS_mexp_data['high_target']]
    
    if is_GD_exp:
        for i, row in df.iterrows():
            exp_ly_score = SnS_mexp_data['layer_scores'][i]
            lys = list(exp_ly_score.keys())
            for ly in lys:
                df.loc[i, f'end_{ly}'] = abs(exp_ly_score[ly][-1])
                if row['num_iter']<499:
                    df.loc[i, f'nat_stat_early_stopping_{ly}'] = abs(exp_ly_score[ly][-1])        
    else:
        #Get statistics of the p1 front solutions.
        #p1_cols is a list of dictionaries where for each experiment 
        #you have the statistics of each checkpoint pareto front
        p1_cols = [
            {
                lbl.split('#')[0]: {
                    'p1_n_it': lbl.split('#')[1],
                    'p1_idxs': v,
                    #for every layer of scores (i.e. distances in that layer wrt the reference), 
                    #get the ly scores of the p1_front solutions v
                    **{k: SnS_mexp_data['layer_scores'][i][k][v[:, 0], v[:, 1]] for k in SnS_mexp_data['layer_scores'][0].keys()}
                }
                for lbl, v in p1.items()
            }
            #note: i is the number of the experiment within the multiexperiment
            for i, p1 in enumerate(SnS_mexp_data['p1_front'])
        ]
        #get the unique names of the checkpoints (unique_p1_cols) and the statistics 
        #taken for each checpoint p1 (unique_p1_vars)
        unique_p1_cols = list(set([k for p1d in p1_cols for k in p1d.keys()]))
        unique_p1_vars = list(p1_cols[0]['end'].keys())

        #for every checkpoint of every experiment, 
        #get the statistics of the p1 front solutions and add them to the dataframe
        for p1c in unique_p1_cols:
            for p1v in unique_p1_vars:
                var = []
                for p1dict in p1_cols:
                    try:
                        var.append(p1dict[p1c][p1v])
                    except:
                        var.append(np.nan)
                df[p1c + '_' + p1v] = var
                
        #for every experiment, get the indexes of all the p1 front solutions
        # (i.e. the set between the p1 front solutions of all the checkpoints)
        df["all_p1s"] = [
            np.unique(
                np.vstack([
                    row[f"{col}_p1_idxs"]
                    for col in unique_p1_cols
                    if not isinstance(row[f"{col}_p1_idxs"], float)
                ]),
                axis=0
            )
            for _, row in df.iterrows()
        ]
    
    #if the input layer is used, compute the maximal pixel distance        
    if '00_input_01' in df['lower_ly'].values:
        df['max_pix_dist'] = compute_max_pix_dist(SnS_mexp_data)
    
    if savepath is not None:
        df.to_json(os.path.join(savepath, 'data_summary.json'))

    return df

def nat_percentiles(exps: Dict[str, Any],
                    end_type = 'end',
                    nat_rec_fp: str = NATURAL_RECORDINGS,
                    save: bool = True,
                    **kwargs) -> pd.DataFrame:
    
    """
    Calculate where SnS images fall within the distribution of natural recordings, expressed as percentiles.
    """
    nrec = load_pickle(nat_rec_fp)
    lbls = np.load(NAT_LABELS)
    nat_stats = defaultdict(list)
    # Iterates through each multiexperiment in the input dictionary

    for k,v in exps.items():
        df = v['df']
        unique_combinations = df[['net_sbj', 'robust', 'upper_ly', 'high_target']].drop_duplicates()
        # For each experiment type, defined by the above combinations, 
        # finds matching data in the natural recordings
        for _, combo in unique_combinations.iterrows():
            filtered_rows = df[
                np.logical_and.reduce([(df[c] == combo[c]) for c in unique_combinations.columns])
            ]
            nat_ly_rec = deep_get(dictionary= nrec, keys = [combo['net_sbj'], 'robust_l2' if combo['robust'] else 'vanilla', combo['upper_ly']])
            #map the high target from a string to a tuple of integers
            htrgt_tuple = tuple(map(int, str(combo['high_target']).strip('()').split(',')))
            #get the linear index of the high target in the natural recordings
            if len(htrgt_tuple) == 3:
                linear_idx = next((i for i, triple in enumerate(zip(*nat_ly_rec['labels'])) if triple == htrgt_tuple), None)
            else:
                linear_idx = int(np.where(nat_ly_rec['labels'] == htrgt_tuple[0])[0])
            nat_distr = nat_ly_rec['data'][linear_idx,:]
            if combo["upper_ly"] == '56_linear_01': #TODO: change if using new nets
                same_cat = nat_distr[np.where(lbls == htrgt_tuple[0])[0]]
                other_cats = nat_distr[np.where(lbls != htrgt_tuple[0])[0]]
            y = filtered_rows[f'{end_type}_{combo["upper_ly"]}']
            ref_activ = filtered_rows['ref_activ']
            y_last = ref_activ - abs(y.apply(lambda x: x[-1]))
            # Calculates where the SnS images activations fall within the distribution of natural recordings
            percentiles = np.searchsorted(np.sort(nat_distr), y_last) / len(nat_distr) * 100
            df.loc[filtered_rows.index, 'nat_percentiles'] = percentiles
            
            #store stats for successive plotting
            nat_max = np.max(nat_distr)
            nat_stats['References'].append(ref_activ.to_numpy()/nat_max)
            nat_stats['SnS'].append(y_last.to_numpy()/nat_max)
            if combo["upper_ly"] == '56_linear_01': #TODO: change if using new nets
                nat_stats['Same_cat'].append(same_cat/nat_max)
                nat_stats['Other_cats'].append(other_cats/nat_max)
            else:
                nat_stats['Natural images'].append(nat_distr/nat_max)
                
        #add the updated dataframe to the experiment
        exps[k]['df'] = df
        if save:
            df.to_json(os.path.join(exps[k]['path'][0], 'data_summary.json'))

        nat_stats = {key: np.concatenate(value) for key, value in nat_stats.items()}

        return nat_stats