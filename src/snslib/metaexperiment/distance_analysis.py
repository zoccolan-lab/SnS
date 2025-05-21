
from collections import defaultdict
from functools import partial, reduce
import json
import os
import random
from typing import Any, Dict, Iterable, Literal
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
import torch
import torchvision.transforms as T
from PIL          import Image


from snslib.experiment.utils.args import CUSTOM_WEIGHTS, DATASET, REFERENCES, WEIGHTS
from snslib.experiment.utils.misc import ref_code_recovery
from snslib.metaexperiment.metaexp import SnS_metadata
from snslib.metaexperiment.plots import pad_tensor_lists, vertical_stack_images, wrap_text
from snslib.core.generator import DeePSiMGenerator
from snslib.core.subject import TorchNetworkSubject
from snslib.core.utils.dataset import MiniImageNet
from snslib.core.utils.io_ import load_pickle, read_json, save_json
from snslib.core.utils.misc import aggregate_df
from snslib.core.utils.probe import RecordingProbe
from snslib.core.utils.torch_net_load_functs import NET_LOAD_DICT, madryLab_robust_load, robustBench_load, torch_load
from torch.utils.data import DataLoader
import argparse
import seaborn as sns

from snslib.core.utils.types import States

DEVICE ='cuda' if torch.cuda.is_available() else 'cpu'

def organize_distances_SnS(distance_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    distances = {}
    distances['with_ref'] = distance_df['ref'].drop(index=['ref']).to_numpy()
    except_ref = distance_df.drop(index=['ref']).drop(columns=['ref'])
    mask = np.tril(np.ones(except_ref.shape), k=0).astype(bool)
    distances['betw_inv'] = except_ref.to_numpy()[~mask]
    return distances

def organize_distances_XDREAM(distance_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    mask = np.tril(np.ones(distance_df.shape), k=0).astype(bool)
    distances = distance_df.to_numpy()[~mask]
    return distances


def distance_plot(results_df: pd.DataFrame,
                  results_df_sem: pd.DataFrame,
                  savepath: str = None,
                  y_lbl = 'Distance',
                  plotting_params: Dict[str, str] = None,
                  plot_only_in_params: bool = True):
    if plotting_params is None:
        plotting_params = {}
    
    fig, ax = plt.subplots(figsize=(20, 8))
    for index, row in results_df.iterrows():
        if index in plotting_params.keys():
            ax.plot(results_df.columns, row, 
                    color = plotting_params[index]['color'], 
                    linestyle = plotting_params[index]['linestyle'], 
                    label=plotting_params[index]['label'] if 'label' in plotting_params[index] else index, 
                    marker='',linewidth=5, clip_on=False)
            ax.fill_between(results_df.columns, row - results_df_sem.loc[index], row + results_df_sem.loc[index], alpha=0.2, color=plotting_params[index]['color'])
        elif not plot_only_in_params:
            ax.plot(results_df.columns, row, label=index)
        else:
            continue


    ax.set_xlabel('Representation space', fontsize=32)
    # Split the y-label into two lines
    ax.set_ylabel(y_lbl.replace(' to ', '\n'), fontsize=32, labelpad=15)
    #ax.set_title('Comparison of Distances Across Layers', fontsize=24)
    ax.legend(ncol=2, fontsize=26, frameon=False)
 
    
    ax.tick_params(axis='both', direction='out', labelsize=26, length=18, width=4, pad=15)
    ax.spines['left'].set_position(('outward', 20))  
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_position(('outward', 20))  
    ax.spines['bottom'].set_linewidth(4)
    
    # Hide the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    
    xt = ax.get_xticks()
    ax.spines['bottom'].set_bounds(xt[0], xt[-1])
    
    # Set y-axis limits and ticks
    ax.set_ylim(0.75, 2.25)
    ax.set_yticks(np.arange(0.75, 2.5, 0.25))
    ax.spines['left'].set_bounds(0.75, 2.25)
    processed_map = {}
    
    for key in plotting_params:
        if '#' in key:
            parts = key.split('#')
            if len(parts) >= 4:
           
                token = parts[3]
              
                trimmed = token[3:] if len(token) > 3 else token
           
                if trimmed.startswith("input"):
                    final_label = "Pixel space"
                else:
                    final_label = trimmed.replace("_", " ")
                
                #
                processed_map[token] = final_label
                processed_map[trimmed] = final_label

              
                if "conv_" in token and "conv2d_" not in token:
                    alt_token = token.replace("conv_", "conv2d_")
                    alt_trimmed = trimmed.replace("conv_", "conv2d_")

                    # Determine the specific label for the alias if needed
                    alias_label = final_label # Default label

                    # Apply specific labels based on the prompt's conditions
                    # Check if the original token contains conv_25 (and not conv2d_25 implicitly via the outer if)
                    if "conv_25" in token:
                        alias_label = "Layer3_conv1"
                    # Check if the original token contains conv_51 (and not conv2d_51 implicitly via the outer if)
                    elif "conv_51" in token:
                         alias_label = "Layer4_conv7"
                    # Note: If token contains neither conv_25 nor conv_51, alias_label remains final_label

                    processed_map[alt_token] = alias_label
                    processed_map[alt_trimmed] = alias_label
 
    



    all_tick_positions = list(range(len(results_df.columns)))
    ax.set_xticks(all_tick_positions)

    tick_labels = []
    highlighted_positions = []

    for i, col in enumerate(results_df.columns):
        label = processed_map.get(col, "")
        tick_labels.append(label if label else "")  
        if label:
            highlighted_positions.append(i)

    ax.set_xticklabels(tick_labels)


   # ax.tick_params(axis='x', color='black', direction='in', length=6, width=2)

    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i in highlighted_positions:
            # Style the tick marks
            tick.tick1line.set_color('black')
            tick.tick1line.set_markersize(24)
            tick.tick1line.set_linewidth(10)

            tick.tick2line.set_color('black')
            tick.tick2line.set_markersize(24)
            tick.tick2line.set_linewidth(10)

            tick.label1.set_fontweight('medium')
            
            # Add faint dashed vertical line at this tick position
            ax.axvline(x=i, color='gray', linestyle='--', linewidth=1.8, alpha=0.5)

        else:
            tick.label1.set_fontweight('medium')
            tick.tick1line.set_color('black')
            tick.tick1line.set_markersize(10)
            tick.tick1line.set_linewidth(1.5)

            tick.tick2line.set_color('black')
            tick.tick2line.set_markersize(10)
            tick.tick2line.set_linewidth(1.5)
            

    
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
    else:
        plt.show()
    plt.close(fig)
    
def plot_accuracy_distribution(accuracy_data, 
                               dist_params,
                               savepath=None,
                               group_sizes = None):#[1, 4, 4]
    """
    Crea un box plot per visualizzare la distribuzione di accuratezza di ogni esperimento.

    Args:
        accuracy_data (dict): Dizionario con struttura {nome_esperimento: {unit: accuracy_value, ...}, ...}.
        dist_params (dict): Dizionario con i parametri di plotting (colori, stili, ecc.).
        savepath (str|None): Percorso dove salvare il plot, se specificato.

    Returns:
        None
    """
    plot_data = []
    labels = []
    colors = []

    common_keys = set.intersection(*[set(subdict.keys()) for subdict in accuracy_data.values()])
    accuracy_data = {key: {subkey: value[subkey] for subkey in common_keys} for key, value in accuracy_data.items()}

   
    order = list(dist_params['plotting'].keys())
    def sort_key(item):
        lbl = item[0]
        for idx, k in enumerate(order):
            if lbl in k:
                return idx
        return len(order)
    for exp_label, unit_dict in sorted(
            accuracy_data.items(),
            key=sort_key
        ):
      
        pl_exp_lbl = [key for key in dist_params['plotting'].keys() if exp_label in key]
        if pl_exp_lbl: pl_exp_lbl = pl_exp_lbl[0]
        else: pl_exp_lbl = exp_label
        
        if pl_exp_lbl in dist_params['plotting']:
            plot_data.append(list(unit_dict.values()))
            labels.append(wrap_text(dist_params['plotting'][pl_exp_lbl]['label'] if 'label' in dist_params['plotting'][pl_exp_lbl] else exp_label))
            colors.append(dist_params['plotting'][pl_exp_lbl].get('color', 'C0'))
        elif pl_exp_lbl == 'nat_refs':
            plot_data.insert(0,list(unit_dict.values()))
            labels.insert(0,'Natural')
            colors.insert(0,'darkgray')


    group_sizes = [1]*len(plot_data) if group_sizes is None else group_sizes
    gap = 1  
    positions = []
    offset = 0
    for size in group_sizes:
        positions += list(range(offset, offset + size)) 
        offset += size + gap
    
    plt.figure(figsize=(18, 10))
    boxplt=plt.boxplot(
        plot_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(linewidth=1.7, edgecolor='black'),
        whiskerprops=dict(linewidth=1.7, color='black'),
        capprops=dict(linewidth=1.7, color='black'),
        medianprops=dict(linewidth=1.9, color='black'),
        showfliers=True
    )
    for patch, color in zip(boxplt['boxes'], colors):
        patch.set_facecolor(color)
    plt.tick_params(axis='both', direction='out', labelsize=26, length=18, width=4, pad=15)
    plt.tick_params(axis='x', rotation=45)
    plt.xticks(positions, labels, rotation=45, ha='right', fontsize=26)

    
    plt.xlabel('Representation space', fontsize=32, labelpad=15)
    plt.ylabel('Accuracy', fontsize=32, labelpad=15)
    ax = plt.gca()
    
    ax.spines['left'].set_position(('outward', 20))  
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_position(('outward', 20))  
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
   

    xt = ax.get_xticks()



    ax.spines['bottom'].set_bounds(xt[0], xt[-1])

    ax.set_ylim(0,1.001)
 
    for artist in ax.artists + ax.lines + ax.collections:
        artist.set_clip_on(False)

   
    for i, artist in enumerate(ax.artists):
        artist.set_edgecolor('black') 
        artist.set_linewidth(3.75)
        
    


    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#1b9e77', edgecolor='#1b9e77', label='Robust'),
        Patch(facecolor='#e57373', edgecolor='#e57373', label='Standard')
       
    ]
    plt.legend(handles=legend_handles, fontsize=32, frameon=False, loc= 'lower left')
    
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
        
def compute_accuracy(logits: np.ndarray,
                     true_class: int,
                     k: int = 1,
                     classes: list[int] | None = None) -> float:
    """
    logits: (N, C) array
    true_class: target class index
    k: top-k
    classes: if given, only these class-indices are eligible
    """
    if classes is not None:
        mask = np.ones(logits.shape[1], bool)
        mask[classes] = False
        logits = np.where(mask, -np.inf, logits)
    # get top-k indices for each sample
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    single_answers = topk == true_class
    return np.mean((single_answers).any(axis=1)), single_answers.astype(int).tolist()


def compute_distances(state_dict: States,
                      state_labes: list[str],
                      aggregator_stat: callable): 
    """
    Compute distances between different states in the given state dictionary.

    Parameters:
    state_dict (dict): A dictionary where keys are state names and values are dictionaries 
                       containing different spaces and their corresponding state representations.
    aggregator_stat (function, optional): A function to aggregate the computed distances.

    Returns:
    dict: A dictionary where keys are space names and values are the aggregated distances 
          computed for that space.
    """
    # Initialize an empty dictionary to store distances for each space
    space_distances = {}  
    for space in state_dict.keys():
        states_in_space = state_dict[space]
        # Compute the pairwise distances between states in the current space
        # Using pdist to compute the pairwise distances and squareform to convert it to a DataFrame
        if state_labes.count('ref')/len(state_labes) == 0.5:
            refs = states_in_space[[i for i, x in enumerate(state_labes) if x == 'ref'],:]
            invs = states_in_space[[i for i, x in enumerate(state_labes) if x != 'ref'],:]
            dists = np.array([euclidean(a, b) for a, b in zip(refs, invs)])
            betw_dists = squareform(pdist(invs, 'euclidean'))
            #build integrated 
            M = np.zeros((betw_dists.shape[0]+1, betw_dists.shape[1]+1))
            M[0, 0]    = 0
            M[0, 1:]   = dists
            M[1:, 0]   = dists
            M[1:, 1:]  = betw_dists
            distance_df = pd.DataFrame(M, index = state_labes[-M.shape[0]:], columns = state_labes[-M.shape[0]:]) 
        else:
            distance_df = pd.DataFrame(squareform(pdist(states_in_space, 'euclidean')),
                                                    index=state_labes, columns=state_labes)
        # Compute the aggregated distances using the provided aggregator function
        space_distances[space] = aggregator_stat(distance_df)
        
    # Return the dictionary containing aggregated distances for each space
    return space_distances  


def distance_analysis_SnS(repr_net: TorchNetworkSubject,
                          generator: DeePSiMGenerator,
                          experiment: Dict[str, Dict[str, Any]],
                          n2view: list[int]|int|None = None,
                          p1: str = 'end_p1_idxs',
                          save_name: str|None = None):
    
    n2view = [] if n2view is None else n2view
    acc = {}
    single_answers = {}
    results_vsref = {}
    results_betw = {}
    images2view = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    #for lbl, v in exp_dict.items():
    #get experiment data
    fp = experiment['path'][0]
    df = experiment['df']
    if p1 not in df.columns:
        p1 = p1.replace('_p1_idxs', '')
        df = df[df[p1+'_'+df['lower_ly'].unique()[0]].notna()]#drop rows with missing values
        is_GD = True
    else:
        is_GD = False
        df = df[df[p1].notna()]#drop rows with missing values
    lbl = experiment['label']
    if save_name:
        betw_fp = os.path.join(fp, f"{save_name}_dist_betw_{p1}.json")
        VSref_fp = os.path.join(fp, f"{save_name}_dist_VSref_{p1}.json")
        acc_fp = os.path.join(fp, f"{save_name}_acc_{p1}.json")
        if os.path.exists(betw_fp) and os.path.exists(VSref_fp) and os.path.exists(acc_fp) and not n2view:
            results_betw[lbl+'#betw'] = pd.read_json(betw_fp)
            results_vsref[lbl+'#VSref'] = pd.read_json(VSref_fp)
            acc[lbl] = read_json(acc_fp)
            print(f"Distance analysis found for {lbl}")  
            return results_vsref, results_betw, acc, images2view, single_answers 
    if is_GD==False:
        df['solution'] = df.apply(lambda r: r[p1][-1] if isinstance(r[p1], Iterable) else np.nan, axis=1)
        df['solution_code_coord'] = df.apply(
            lambda row: (np.where((np.array(row['all_p1s']) == row['solution']).all(axis=1))[0][0]
                        if (np.array(row['all_p1s']) == row['solution']).all(axis=1).any()
                        else np.nan),
            axis=1)  
    units = df['high_target'].unique()
    data_pkl = load_pickle(os.path.join(fp, 'data.pkl'))
    distances = {}
    acc[lbl] = {}
    single_answers[lbl] = {}
    acc['nat_refs'] = {}
    single_answers['nat_refs'] = {}
    if isinstance(n2view, int):
        n2view = [n for n in random.sample(units.tolist(), n2view)]
    for n in units: #for each target unit
        idxs = df[df['high_target'] == n].index.tolist()
        n = str(n).replace('(', '[').replace(')', ']')
        if n.isdigit(): n = f"[{n}]"
        #once the previous condition included a & (df['task'] == v['task'])
        #but one can assume that each df contains only one task
        #get the reference info
        
        def get_ref(ref_info, df):
            if isinstance(ref_info, str) and os.path.exists(ref_info):
                transform = T.Compose([
                    T.Resize((256, 256)),
                    T.ToTensor(),
                ])
                ref_img = (transform(Image.open(ref_info).convert("RGB")).unsqueeze(0).to('cuda'))
                return ref_img
            else:
                ref_ly = df.loc[idxs]['upper_ly'].unique()[0] #NOTE: not the best way to do this
                net = df.loc[idxs]['net_sbj'].unique()[0]
                net = net + '_r' if df.loc[idxs]['robust'].unique()[0] else net
                ref_info['layer'] = ref_ly
                ref_fp = ref_info.pop('ref_file')
                ref_info['code'] = 'code'
                ref_info = {'network': net, **ref_info}
                code_ref = ref_code_recovery(reference_file = load_pickle(REFERENCES), 
                                                keys = ref_info, ref_file_name = ref_fp)
                return code_ref
            
        ref_info_L = [data_pkl['reference_info'][i] for i in idxs]
        if  all(x == ref_info_L[0] for x in ref_info_L):
            code_ref = [get_ref(ref_info_L[0], df)]
        else:
            code_ref = []
            for ref_info in ref_info_L:
                code_ref.append(get_ref(ref_info, df))
                
        #organize the data for the generator and extract the corresponding labels
        if is_GD == False:
            codes_inv, inv_lbls = zip(*[[np.expand_dims(data_pkl['p1_codes'][i][df['solution_code_coord'].loc[i],:], axis=0), f'inv_{i}'] for i in idxs])
            codes = codes_inv if torch.is_tensor(code_ref[0]) else code_ref+list(codes_inv)
            images = generator(np.vstack(codes))
            if torch.is_tensor(code_ref[0]):
                images = torch.cat((torch.cat(code_ref, dim=0), images), dim=0)
        else:
            ref_im = generator(np.vstack(code_ref))
            inv_imgs, inv_lbls = zip(*[[torch.from_numpy(data_pkl['solution'][i]).to(DEVICE), f'inv_{i}'] for i in idxs])
            images = torch.cat([ref_im]+list(inv_imgs), dim = 0) 
        lbls = ['ref']*len(code_ref) + list(inv_lbls)
        states = repr_net(images)

        n_num = n.replace('[','').replace(']','')
        n_num = int(n_num) if n_num.isdigit() else 1000
        if n in n2view:
            images2view[lbl][n]['variants'] = {k: im for k, im in zip(lbls, images) if k != 'ref'}
            rname = 0 if isinstance(code_ref[0], torch.Tensor) else ref_info_L[0]['seed']
            images2view[lbl][n]['reference'] = {rname: images[0]}
            #print(images2view[lbl][n]['variants'][inv_lbls[0]].shape)
            #print(images2view[lbl][n]['reference'][ref_info['seed']].shape)
            
        distances[n] = compute_distances(state_dict = states, 
                                        state_labes = lbls,
                                        aggregator_stat = organize_distances_SnS)
        is_inv = [i for i, x in enumerate(lbls) if x != 'ref']
        if len(units[0].split(','))==1:#acc[lbl][n] = np.sum(np.argmax(states[repr_net.layer_names[-1]][is_inv,:], axis =1) == n_num)/len(is_inv)
            ac, single_ans= compute_accuracy(logits = states[repr_net.layer_names[-1]][is_inv,:],
                                                    true_class = n_num,
                                                    k = 1,
                                                    classes = [int(''.join(ch for ch in u if ch.isdigit())) if isinstance(u, str) else u for u in units])
            acc[lbl][n] = ac
            single_answers[lbl][n] = single_ans
            if len(lbls)- len(is_inv) > 1:
                is_ref = [i for i, x in enumerate(lbls) if x == 'ref']
                #acc['nat_refs'][n] = np.sum(np.argmax(states[repr_net.layer_names[-1]][is_ref,:], axis =1) == n_num)/len(is_ref)
                ac, single_ans = compute_accuracy(logits = states[repr_net.layer_names[-1]][is_ref,:],
                                                    true_class = n_num,
                                                    k = 1,
                                                    classes = [int(''.join(ch for ch in u if ch.isdigit())) if isinstance(u, str) else u for u in units])
                acc['nat_refs'][n] = ac
                single_answers['nat_refs'][n] = single_ans
            
            
    SnSri_distaces = pd.DataFrame.from_dict(distances, orient='index')
    results_betw[lbl+'#betw'] = SnSri_distaces.applymap(lambda x: x['betw_inv'] if isinstance(x, dict) and 'betw_inv' in x else None)
    results_vsref[lbl+'#VSref'] = SnSri_distaces.applymap(lambda x: x['with_ref'] if isinstance(x, dict) and 'with_ref' in x else None)
    if save_name:
        results_betw[lbl+'#betw'].to_json(betw_fp)
        results_vsref[lbl+'#VSref'].to_json(VSref_fp)
        save_json(acc[lbl], acc_fp)                
    
    return results_vsref, results_betw, acc, images2view, single_answers


def distance_analysis_XDREAM(repr_net: TorchNetworkSubject,
                             generator: DeePSiMGenerator,
                             path_to_refs: str = REFERENCES,
                             opt_net: str = 'resnet50',
                             opt_gen: str = 'fc7',
                             opt_trgt: str = '56_linear_01',
                             exp_name: str = 'XDREAM multiple inits',
                             n2view: list[str]|int|None = None,
                             save_name: str|None = None,
                             classes4classify = [1,19,43,118,179,316,470,579,654,759,805,968]):
    

    p2dist = os.path.join(os.path.dirname(path_to_refs), 'distances_XDREAM_nats')
    os.makedirs(p2dist, exist_ok=True)
    p2XDREAM_dist = os.path.join(p2dist, f"{save_name}_trgt_{opt_net}_{opt_gen}_{opt_trgt}.json")
    p2acc = os.path.join(p2dist, f"{save_name}_trgt_{opt_net}_{opt_gen}_{opt_trgt}_acc.json")
    results = {}
    n2view = [] if n2view is None else n2view
    images2view = defaultdict(lambda: defaultdict(dict))
    if os.path.exists(p2XDREAM_dist) and os.path.exists(p2acc) and not n2view:
        results[exp_name] = pd.read_json(p2XDREAM_dist)
        acc = read_json(p2acc)
        print(f"Distance analysis found for ref {opt_net}_{opt_gen}_{opt_trgt} with {save_name}")
    else:
        refs = load_pickle(path_to_refs)
        available_neurons = list(refs['reference'][opt_net][opt_gen][opt_trgt].keys())
        if isinstance(n2view, int):
            n2view = [int(n.strip('[]')) for n in random.sample(available_neurons, n2view)]
        #we will compute the distances between all pairs of references for each neuron
        distances = {}
        acc = {}
        single_answers = {}
        for n in available_neurons:
            n_refs = refs['reference'][opt_net][opt_gen][opt_trgt][n]
            rseeds, codes = zip(*[[k, v['code']] for k,v in n_refs.items()])
            codes = np.vstack(codes)
            images = generator(codes)
            #let's store the images for visualization
            if n in n2view: images2view[n]['variants'] = {k: im for k, im in zip(rseeds, images)} #int(n.strip('[]'))
            #print(images2view[n_int]['variants'][rseeds[0]].shape)
            states = repr_net(images)
            
            #We store the states associated to the various references of neuron n
            n_num = n.replace('[','').replace(']','')
            n_num = int(n_num) if n_num.isdigit() else 1000
            acc[n] = np.sum(np.argmax(states[repr_net.layer_names[-1]], axis =1) == n_num)/states[repr_net.layer_names[-1]].shape[0]
            ac, single_ans = compute_accuracy(logits = states[repr_net.layer_names[-1]],
                                        true_class = n_num,
                                        k = 1,
                                        #classes = [int(''.join(ch for ch in u if ch.isdigit())) for u in available_neurons])      
                                        classes = classes4classify)
            acc[n] = ac
            single_answers[n] = single_ans
            #for each neuron, we should compute the distance between all pairs of references
            #we will store the distances in a dictionary, whose keys are n
            distances[n] = compute_distances(state_dict = states,
                                                state_labes = rseeds,
                                                aggregator_stat = organize_distances_XDREAM)
        print(exp_name)
        results[exp_name] = pd.DataFrame.from_dict(distances, orient='index')
        if save_name: 
            results[exp_name].to_json(p2XDREAM_dist)
            save_json(acc, p2acc)

    return results, acc, images2view, single_answers


def distance_analysis_nat_imgs(repr_net: TorchNetworkSubject,
                               n_samples: int = 10,
                               rnd = False, #if True, we will sample random images from the dataset,
                                            #otherwise we will sample images from the same category
                               savepath: str|None = None,
                               save_name: str|None = None):
    
    """
    Analyzes the distances between representations of natural images in a neural network's feature space.
    This function computes pairwise distances between neural representations of images from the MiniImageNet dataset.
    It can sample images either randomly from the entire dataset or from the same category.
    Parameters
    ----------
    repr_net : TorchNetworkSubject
        The neural network model used to measure representations of the images.
    n_samples : int, default=10
        Number of images to sample per category or randomly.
    rnd : bool, default=False
        If True, samples random images from the dataset regardless of category.
        If False, samples images from the same category.
    savepath : str or None, default=None
        Directory path where the results will be saved. If None, results are not saved.
    save_name : str or None, default=None
        Base filename for saving results. If None, results are not saved.
    Returns
    -------
    dict
        Dictionary containing a pandas DataFrame with pairwise distances between image representations.
        The key corresponds to the sampling method used.
    """
    
    
    
    #load the natural images dataset
    nat_imgs = MiniImageNet(root = DATASET)
    batch_sz = int(len(nat_imgs)/len(nat_imgs.classes))
    nat_loader = DataLoader(nat_imgs, batch_size = batch_sz, shuffle = rnd, num_workers=8)
    
    #the labels are defined by the type of sampling
    if rnd: r_lbl = 'nat_images - random' 
    else: r_lbl = 'nat_images - same cat'
    
    #generate the file path for saving results
    if savepath and save_name:
        os.makedirs(savepath, exist_ok=True)
        fn = f"{save_name}_nat_sameC.json" if not rnd else f"{save_name}_nat_randC.json"
        fp = os.path.join(savepath, fn)
    #search for existing results    
    results = {}
    if os.path.exists(fp):
        #if results already exist, load them
        results[r_lbl] = pd.read_json(fp)
        print(f"Distance analysis found for {r_lbl}")
    else:
        #if results do not exist, compute them
        #initialize the dictionary to store distances
        repr_d = {}
        distances_nats = {}
        analyzed_cats = []
        #iterate over the dataset
        for i,d in enumerate(nat_loader):
            if not rnd:
                #if not random, we will sample images from the same category
                #get the category of the current batch
                assert torch.unique(d['labels']).numel() == 1, "The batch of nat images contains more than one unique label."
                inet_cat = int(torch.unique(d['labels']))
                #if the category has already been analyzed, skip it
                if inet_cat in analyzed_cats:
                    continue
                lbl = inet_cat
            else:
                #if random, we just assign a mock label
                lbl = f'rs{i}'
            #get the images and labels
            full_range = range(batch_sz)
            idxs_set1 = random.sample(full_range, n_samples)
            #get the states of the images in the representation network
            repr_d = repr_net(d['images'][idxs_set1].to(DEVICE))
            distances_nats[lbl] = compute_distances(state_dict = repr_d, 
                                                    state_labes = idxs_set1,
                                                    aggregator_stat= organize_distances_XDREAM)
            
            #if not random, let's store the inet cat for 
            if not rnd: analyzed_cats.append(inet_cat)
            
        #save your results    
        results[r_lbl] = pd.DataFrame.from_dict(distances_nats, orient='index')
        if savepath and save_name: results[r_lbl].to_json(fp)
        
    return results
    


def main():
    #parse the arguments of the distance analysis
    parser = argparse.ArgumentParser(description='Distance Analysis')
    parser.add_argument('--params', type=str, default='./dist_params.json', help='Path to the parameter file')
    parser.add_argument('--norm_var', type=str, default='nat_images - same cat', help='Variable to normalize the distances')
    args = parser.parse_args()
    prms = read_json(args.params)
    
    #set the ref network loading function
    if prms['ref_net']['robust'] == 'imagenet_l2_3_0.pt':
        net_load = madryLab_robust_load
        wp = os.path.join(CUSTOM_WEIGHTS, prms['ref_net']['net_name'], prms['ref_net']['robust'])
        net_nickname = prms['ref_net']['net_name']+'_l2'
    elif prms['ref_net']['robust'] == '':
        net_load = torch_load
        wp = ''
        net_nickname = prms['ref_net']['net_name']
    else:
        net_nickname = prms['ref_net']['net_name']+'_linf'
        net_load = robustBench_load
        wp = prms['ref_net']['robust']
        
    #load ref net
    repr_net = TorchNetworkSubject(
        network_name=prms['ref_net']['net_name'],
        t_net_loading = net_load,
        custom_weights_path = wp
    )
    probe = RecordingProbe(target = {ln : [] for ln in repr_net.layer_names})
    repr_net = TorchNetworkSubject(
        record_probe=probe,
        network_name=prms['ref_net']['net_name'],
        t_net_loading = net_load,
        custom_weights_path = wp
    )
    repr_net.eval()

    #we will a generator to generate images from codes
    generator = DeePSiMGenerator(
        root    = str(WEIGHTS),
        variant = str(prms['ref_net']['gen']) # type: ignore
    ).to(DEVICE)
    
    result_dict = {}
    acc_dict = {}
    sing_ans_dict = {}
    Image_dict = {}


    neurons2view = prms['plotting']['neurons2view']
    if neurons2view == [] and prms['plotting']['generate_collages']:
        available_neurons = []
        for _,v in prms['XDREAM'].items():
            refs = load_pickle(v['fp'])
            available_neurons += list(refs['reference'][v['net']][v['gen']][v['ly']].keys())
        neurons2view = list(set(available_neurons))
        neurons2view = [int(''.join(ch for ch in n if ch.isdigit())) for n in neurons2view]

    
    for k in prms.keys():
        if k == "SNS_exp":
            #load the metadata
            SNS_metadata = SnS_metadata.from_json("/home/ltausani/Documents/GitHub/ZXDREAM/metaexperiment/SnS_multiexp_dirs.json")
            #filter the metadata - TO BE DEBUGGED
            # for c, fltr in prms['SNS_exp']['filtering'].items():
            #     SNS_metadata = SNS_metadata.filter_dfs(column = c, dropna = True, value= fltr if fltr != [] else None)
            #get the experiments
            SNS_metaexp = SNS_metadata.get_experiments(queries = prms['SNS_exp']['query'])
            
            for k,v in SNS_metaexp.items():
                results_vsref, results_betw, acc, SnS_imgs, single_answ_SnS = distance_analysis_SnS(repr_net = repr_net,
                                  generator = generator,
                                  experiment = v,
                                  p1 = f"{prms['SNS_exp']['p1']}_p1_idxs",
                                  n2view = neurons2view if prms['plotting']['generate_collages'] else None,
                                  save_name = net_nickname)
                Image_dict.update(SnS_imgs)
                acc_dict.update(acc)
                sing_ans_dict.update(single_answ_SnS)
                result_dict.update(results_vsref)
                result_dict.update(results_betw)
            
        elif k == "XDREAM":
            for xdk,xdv in prms[k].items():
                xd_lbl = 'mXDREAM - '+xdk
                results, acc, XD_imgs, sa_XDREAM = distance_analysis_XDREAM(repr_net = repr_net,
                                        generator = generator,
                                        path_to_refs = xdv['fp'],
                                        opt_net = xdv['net'],
                                        opt_gen = xdv['gen'],
                                        opt_trgt = xdv['ly'],
                                        exp_name= xd_lbl,
                                        n2view = neurons2view if prms['plotting']['generate_collages'] else None,
                                        save_name = net_nickname)
                acc_dict[xd_lbl] = acc
                sing_ans_dict[xd_lbl] = sa_XDREAM
                result_dict.update(results)
                Image_dict[xd_lbl] = XD_imgs
                print(xd_lbl)

        elif k == "nats":
            for t in prms[k]['types']:
                results = distance_analysis_nat_imgs(repr_net = repr_net,
                                                    n_samples = prms[k]['n_samples'],
                                                    rnd = True if t == 'rand' else False,
                                                    savepath = os.path.join(os.path.dirname(prms["XDREAM"]["vanilla"]['fp']), 'distances_XDREAM_nats'),
                                                    save_name = net_nickname)
                result_dict.update(results)
        else:
            continue

    analysis_dir = os.path.join(os.getcwd(), 'Distance_analysis', prms['exp_name'])
    os.makedirs(analysis_dir, exist_ok=True)
    save_json(prms, os.path.join(analysis_dir, 'params.json'))
    results_avg = {}; results_sem = {}
    for k,v in result_dict.items():
        results_avg[k] = aggregate_df(df=v, f_aggr_betw_cells = partial(np.mean, axis=0))
        results_sem[k] = aggregate_df(df=v, f_aggr_betw_cells = lambda x: np.std(x, axis=0)/np.sqrt(x.shape[0]))

    results_df_avg = pd.DataFrame.from_dict(results_avg, orient='index')
    results_df_avg_norm = results_df_avg.div(results_df_avg.loc[args.norm_var])
    results_df_sem = pd.DataFrame.from_dict(results_sem, orient='index')
    results_df_sem_norm = results_df_sem.div(results_df_avg.loc[args.norm_var])

    results_df_avg.to_csv(os.path.join(analysis_dir, 'results.csv'))
    results_df_avg_norm.to_csv(os.path.join(analysis_dir, 'results_normalized.csv'))
    
    distance_plot(results_df_avg,
                  results_df_sem,
                  savepath = os.path.join(analysis_dir, 'distance_plot.png'),
                  plotting_params = prms['plotting'])
    distance_plot(results_df_avg_norm, 
                  results_df_sem_norm,
                  savepath = os.path.join(analysis_dir, 'distance_plot_normalized.svg'),
                  y_lbl='Euclidean distance relative to to same category variation',
                  plotting_params = prms['plotting']
                  )
    plot_accuracy_distribution(accuracy_data = acc_dict,
                               dist_params = prms,
                               savepath = os.path.join(analysis_dir, 'accuracy_boxplot.png'))
    
    if prms['plotting']['generate_collages']:
        img_dict = {}
        d = defaultdict(list)
        units = [list(v.keys()) for v in Image_dict.values()]
        neurons2view = list(reduce(set.intersection, map(set, units)))
        to_pil = T.ToPILImage()
        for n in neurons2view:
            try:
                for k in Image_dict.keys():
                    variants = list(Image_dict[k][n]['variants'].values())
                    ref = list(Image_dict[k][n]['reference'].values())
                    if len(ref) == 0:
                        ref = [torch.ones(variants[0].shape)]
                    # adapt labels
                    plt_k = [pl_k for pl_k in prms['plotting'].keys() if k == pl_k or k == '#'.join(pl_k.split('#')[:-1])]
                    plt_k = plt_k[0] if len(plt_k) == 1 else k
                    if plt_k in prms['plotting']:
                        if 'label' in prms['plotting'][plt_k]:
                            plt_k = prms['plotting'][plt_k]['label']
                        #save images just if the have to be plotted 
                        d[plt_k] = ref + variants

                        #save single images
                        if prms['plotting']['save_single_imgs']:
                            out_dir = os.path.join(analysis_dir,'single_imgs',f"unit {n}", plt_k)
                            os.makedirs(out_dir, exist_ok=True)
                            for i, img_tensor in enumerate(variants):
                                pil_img = to_pil(img_tensor.cpu())
                                pil_img.save(os.path.join(out_dir, f"unit_{n}_{plt_k}_{i}.png"))

                d = dict(d)
                print(d.keys())
                #print(d['m-XDREAM'][0].shape)
                padded_imgs = pad_tensor_lists(tensor_lists=d)
                img_dict[n] = vertical_stack_images(padded_imgs,y_dist=50, font_size=40, margin=10)
            except:
                continue
        #save the images
        gen_imgs_dir = os.path.join(analysis_dir, 'gen_imgs')
        os.makedirs(gen_imgs_dir, exist_ok=True)
        {os.path.join(gen_imgs_dir, f"{k}.png"): v.save(os.path.join(gen_imgs_dir, f"{k}.png")) 
        for k, v in img_dict.items()}
            
    
    with open(os.path.join(analysis_dir, 'accuracy.json'), 'w') as f:
        json.dump(acc_dict, f, indent=4)

    with open(os.path.join(analysis_dir, 'single_answers.json'), 'w') as f:
        json.dump(sing_ans_dict, f, indent=4)

if __name__ == "__main__":
    main()