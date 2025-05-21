import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import torch
from collections import defaultdict
from src.snslib.core.utils.io_ import load_pickle, read_json, save_json
from functools import partial, reduce
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from src.snslib.metaexperiment.plots import pad_tensor_lists, vertical_stack_images



try:
    from src.snslib.experiment.utils.args import WEIGHTS, CUSTOM_WEIGHTS
except ImportError:
    print("Warning: experiment.utils.args not found. WEIGHTS and CUSTOM_WEIGHTS might be undefined.")
    WEIGHTS = "path/to/weights" # Default or placeholder
    CUSTOM_WEIGHTS = "path/to/custom_weights" # Default or placeholder



from src.snslib.core.generator import DeePSiMGenerator
from src.snslib.core.subject import TorchNetworkSubject
from src.snslib.core.utils.probe import RecordingProbe
from src.snslib.core.utils.torch_net_load_functs import torch_load, madryLab_robust_load, robustBench_load
from src.snslib.metaexperiment.metaexp import SnS_metadata
from src.snslib.metaexperiment.distance_analysis import distance_analysis_SnS, distance_analysis_XDREAM, distance_analysis_nat_imgs, distance_plot as original_distance_plot, plot_accuracy_distribution as original_plot_accuracy_distribution
from src.snslib.metaexperiment.metaexp_functs import nat_percentiles as original_nat_percentiles
from src.snslib.core.utils.misc import aggregate_df


# --- Utilities for Activation Percentiles Boxplot ---

def calculate_activation_percentiles(exps):
    """
    Calculates natural statistics percentiles for given experiments.
    Adapted from nat_percentiles in metaplot.ipynb (cell 11).
    """
    if not exps:
        print("Warning: No experiments provided to calculate_activation_percentiles.")
        return {}
        
    # The original nat_percentiles function from metaexperiment.metaexp_functs is directly usable.
    # This wrapper ensures it's called correctly.
    # It expects `exps` to be a dictionary {label: data_dict}.
    # `data_dict` should contain a DataFrame `df`.
    # `original_nat_percentiles` processes this structure.
    results = original_nat_percentiles(exps=exps, end_type='end', y_norm=True, x_norm=False)
    return results


def plot_activation_percentiles_boxplot(
    general_stats_std,          # e.g., nat_stats_std_inv_pix (for 'Same cat.', 'Other cats.')
    general_stats_rob,          # e.g., nat_stats_r_inv_pix
    sns_experiment_data,        # List of tuples: (label, nat_stats_std_dict, nat_stats_rob_dict)
    save_dir=None,
    filename="activation_percentiles_boxplot.png"
):
    """
    Generates a flexible boxplot of activation percentiles.

    Args:
        general_stats_std (dict): Statistics dict for standard model, used for 'Same cat.' and 'Other cats.'.
                                  Typically, this would be the pixel-space invariance stats for standard.
        general_stats_rob (dict): Statistics dict for robust model, corresponding to general_stats_std.
        sns_experiment_data (list): A list of tuples. Each tuple should be:
                                    (label_for_plot (str), 
                                     nat_stats_std_for_this_exp (dict), 
                                     nat_stats_rob_for_this_exp (dict))
                                    Example: [("Inv Pixel space", nat_stats_std_inv_pix, nat_stats_r_inv_pix),
                                              ("Adv Pixel space", nat_stats_adv_std, nat_stats_adv_rob)]
        save_dir (str, optional): Directory to save the plot. Defaults to None.
        filename (str, optional): Filename for the saved plot. Defaults to "activation_percentiles_boxplot.png".
    """
    

    rename_map_for_labels = {
        'Same_cat': 'Same cat.',
        'Other_cats': 'Other cats.',
    }
    data_series1_list = []  # For Standard/Vanilla model data
    data_series2_list = []  # For Robust model data
    plot_labels_list = []   # For x-axis labels

    # 1. Handle general categories (e.g., 'Same_cat', 'Other_cats')
    # These are derived from general_stats_std and general_stats_rob
    if not general_stats_std or not general_stats_rob:
        print("Warning: general_stats_std or general_stats_rob is empty. Skipping general categories.")
    else:
        # Ensure the keys exist before trying to access them
        source_keys_initial = list(general_stats_std.keys())
        general_category_keys = [k for k in source_keys_initial if k in ['Same_cat', 'Other_cats']]

        for key in general_category_keys:
            if key in general_stats_std and key in general_stats_rob:
                # Convert numpy arrays to lists to avoid ValueError with 'if d:' in min/max calculation
                data1 = general_stats_std[key]
                data2 = general_stats_rob[key]
                data_series1_list.append(list(data1) if hasattr(data1, 'tolist') else data1)
                data_series2_list.append(list(data2) if hasattr(data2, 'tolist') else data2)
                plot_labels_list.append(rename_map_for_labels.get(key, key))
            else:
                print(f"Warning: Key '{key}' not found in both general_stats_std and general_stats_rob. Skipping for general categories.")

    # 2. Add data from specific SnS experiments provided in sns_experiment_data
    for label, stats_std, stats_rob in sns_experiment_data:
        natural_images_key = 'Natural images'
        
        # Plot 'Natural images' data first if available for this experiment item
        if natural_images_key in stats_std and natural_images_key in stats_rob:
            data1_nat = stats_std[natural_images_key]
            data2_nat = stats_rob[natural_images_key]
            data_series1_list.append(list(data1_nat) if hasattr(data1_nat, 'tolist') else data1_nat)
            data_series2_list.append(list(data2_nat) if hasattr(data2_nat, 'tolist') else data2_nat)
            plot_labels_list.append(f"{label} \n (Natural Images)")
        elif natural_images_key in stats_std or natural_images_key in stats_rob: # If 'Natural images' is in one dict but not both
            print(f"Warning: '{natural_images_key}' key for '{label}' not found in both standard and robust stats. Skipping '{natural_images_key}' data for this entry.")
        # If 'Natural images' is in neither, no warning.

        # Then, plot 'SnS' data (target data) if available
        if 'SnS' in stats_std and 'SnS' in stats_rob:
            data1_sns = stats_std['SnS']
            data2_sns = stats_rob['SnS']
            data_series1_list.append(list(data1_sns) if hasattr(data1_sns, 'tolist') else data1_sns)
            data_series2_list.append(list(data2_sns) if hasattr(data2_sns, 'tolist') else data2_sns)
            plot_labels_list.append(label)
        elif 'SnS' in stats_std or 'SnS' in stats_rob: # If 'SnS' is in one dict but not both
            print(f"Warning: 'SnS' key for '{label}' not found in both standard and robust stats. Skipping 'SnS' data for this entry.")
        # If 'SnS' is in neither, no warning for 'SnS' specifically.


    if not data_series1_list or not data_series2_list: # Check if any data was actually prepared for plotting
        # This implies that if one list is empty, the other should also be, given the paired additions.
        print("Error: No data to plot for activation percentiles. Aborting plot.")
        return

    data_series1 = data_series1_list
    data_series2 = data_series2_list
    keys_for_plot = plot_labels_list

    plt.figure(figsize=(max(10, len(keys_for_plot) * 2.0),5)) 
    num_groups = len(keys_for_plot)
    group_spacing_factor = 3.0
    if num_groups > 4: group_spacing_factor = 2.5
    if num_groups > 6: group_spacing_factor = 2.0
        
    group_centers = np.arange(num_groups) * group_spacing_factor
    offset_within_group = 0.8
    positions1 = group_centers
    positions2 = group_centers + offset_within_group

    # Define default colors and gradient shades for the boxplots
    vanilla_shades = ["#f1b4b4", "#e57373", "#d93232"] 
    robust_shades = ["#a1d99b", "#74c476", "#41ab5d"] # Greenish shades for robust
    
    default_vanilla_color = "#e57373" # Base color for vanilla/standard
    default_robust_color = "#74c476"  # Base color for robust

    box1 = plt.boxplot(data_series1, positions=positions1, widths=0.6, patch_artist=True,
                       medianprops=dict(color="k"), capprops=dict(color="k"),
                       whiskerprops=dict(color="k"), flierprops=dict(markeredgecolor="k", alpha=0.3))
    box2 = plt.boxplot(data_series2, positions=positions2, widths=0.6, patch_artist=True,
                       medianprops=dict(color="k"), capprops=dict(color="k"),
                       whiskerprops=dict(color="k"), flierprops=dict(markeredgecolor="k", alpha=0.3))

    num_boxes_in_series = len(box1['boxes'])
    num_gradient_boxes = 0 # This means the last 3 categories plotted will get the gradient

    # Color Vanilla boxes
    for i, patch in enumerate(box1['boxes']):
        patch.set_edgecolor('k')
        idx_from_end = (num_boxes_in_series - 1) - i
        if 0 <= idx_from_end < num_gradient_boxes: # Apply gradient to last N categories
            shade_palette_index = (num_gradient_boxes - 1) - idx_from_end
            if shade_palette_index < len(vanilla_shades):
                 patch.set_facecolor(vanilla_shades[shade_palette_index])
            else: # Fallback if not enough shades
                patch.set_facecolor(default_vanilla_color)
        else: # Default color for other categories
            patch.set_facecolor(default_vanilla_color)

    # Color Robust boxes
    for i, patch in enumerate(box2['boxes']):
        patch.set_edgecolor('k')
        idx_from_end = (num_boxes_in_series - 1) - i
        if 0 <= idx_from_end < num_gradient_boxes:
            shade_palette_index = (num_gradient_boxes - 1) - idx_from_end
            if shade_palette_index < len(robust_shades):
                patch.set_facecolor(robust_shades[shade_palette_index])
            else: # Fallback
                patch.set_facecolor(default_robust_color)
        else:
            patch.set_facecolor(default_robust_color)

    tick_positions = group_centers + offset_within_group / 2
    plt.xticks(tick_positions, keys_for_plot, rotation=45, ha='right', fontsize=12)
    plt.ylabel("Normalized activation", fontsize=14)
    plt.legend([box1["boxes"][0], box2["boxes"][0]], ["Standard", "Robust"], loc="lower left", frameon=False, fontsize=12)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.tick_params(axis='both', direction='out', labelsize=12, length=10, width=2, pad=10)
    
    xt = ax.get_xticks()
    yt = ax.get_yticks()
    if len(xt) > 0: ax.spines['bottom'].set_bounds(xt[0], xt[-1])
    
    # Ensure data lists are not empty before attempting min/max on potentially empty sublists
    # The `if d` in generator expression handles empty sublists `d`.
    # `data_series1` itself is checked earlier to not be empty.
    min_data_val = min(min(d) for d_list in [data_series1, data_series2] for d in d_list if d) if any(d for d_list in [data_series1, data_series2] for d in d_list) else 0
    max_data_val = max(max(d) for d_list in [data_series1, data_series2] for d in d_list if d) if any(d for d_list in [data_series1, data_series2] for d in d_list) else 1.1
    
    y_lower_bound = min(0, min_data_val - 0.1*(max_data_val - min_data_val))
    y_upper_bound = max(1.1, max_data_val + 0.1*(max_data_val - min_data_val))

    ax.spines['left'].set_bounds(-4, 2)


    plt.tight_layout()
    if save_dir:
        # This assumes os is imported globally
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Saved activation percentile boxplot to {os.path.join(save_dir, filename)}")
    plt.show()

# --- Utilities for SVC Analysis ---

def process_experiment_data_for_svc(exp_dict, p1, rec_ly, generator, repr_net, DEVICE):
    """
    Processes experiment data to extract representations for SVC.
    Adapted from playground_meta.ipynb (cell 4).
    """
    from typing import Iterable # Moved import here
    reprs = defaultdict(list)
    for lbl, v_data in exp_dict.items():
        df = v_data['df'].copy() # Use .copy() to avoid SettingWithCopyWarning
        df['solution'] = df.apply(lambda r: r[p1][-1] if isinstance(r[p1], Iterable) and len(r[p1]) > 0 else np.nan, axis=1)
        df.dropna(subset=['solution'], inplace=True) # Drop rows where solution is NaN

        df['solution_code_coord'] = df.apply(
            lambda row: (np.where((np.array(row['all_p1s']) == row['solution']).all(axis=1))[0][0]
                        if isinstance(row['all_p1s'], Iterable) and (np.array(row['all_p1s']) == row['solution']).all(axis=1).any()
                        else np.nan),
            axis=1)
        df.dropna(subset=['solution_code_coord'], inplace=True) # Drop rows where solution_code_coord is NaN
        df['solution_code_coord'] = df['solution_code_coord'].astype(int) # Ensure it's int for indexing

        data_pkl_path = os.path.join(v_data['path'][0], 'data.pkl')
        if not os.path.exists(data_pkl_path):
            print(f"Warning: data.pkl not found at {data_pkl_path} for experiment {lbl}. Skipping.")
            continue
        data_pkl = load_pickle(data_pkl_path)

        units = df['high_target'].unique()
        print(f"Processing SVC data for experiment: {lbl}")
        for n_unit in units:
            idxs = df[df['high_target'] == n_unit].index.tolist()
            
            # Filter idxs based on valid 'solution_code_coord'
            valid_idxs_data = []
            for i in idxs:
                if i in df.index and pd.notna(df.loc[i, 'solution_code_coord']):
                    solution_coord = df.loc[i, 'solution_code_coord']
                    if 0 <= solution_coord < len(data_pkl['p1_codes'][i]):
                         valid_idxs_data.append(
                             (np.expand_dims(data_pkl['p1_codes'][i][solution_coord,:], axis=0), lbl)
                         )
                    else:
                        print(f"Warning: solution_code_coord {solution_coord} out of bounds for data_pkl['p1_codes'][{i}] (len: {len(data_pkl['p1_codes'][i])}). Skipping index {i}.")
            
            if not valid_idxs_data:
                print(f"Warning: No valid codes found for unit {n_unit} in experiment {lbl}. Skipping.")
                continue

            codes_inv, inv_lbls = zip(*valid_idxs_data)
            codes = np.vstack(list(codes_inv))
            
            images = generator(codes.astype(np.float32)) # Ensure codes are float32
            with np.errstate(invalid='ignore'): # Suppress potential numpy warnings in repr_net
                 states = repr_net(images.to(DEVICE)) # Move images to device
            
            # Ensure states[rec_ly] are numpy arrays on CPU for concatenation
            if isinstance(states, dict) and rec_ly in states:
                current_states_np = states[rec_ly].detach().cpu().numpy() if hasattr(states[rec_ly], 'detach') else np.array(states[rec_ly])
                reprs[str(n_unit).replace("(", "").replace(")", "")].append(({rec_ly: current_states_np}, inv_lbls))
            else:
                print(f"Warning: rec_ly {rec_ly} not found in states or states is not a dict for unit {n_unit}, exp {lbl}.")

    return reprs


def unify_representations(reprs_dict):
    """
    Unifies representations collected from multiple experiments/units.
    Adapted from playground_meta.ipynb (cell 5).
    """
    unified_dict = {}
    for k, list_of_tuples in reprs_dict.items():
        if not list_of_tuples: continue # Skip empty lists
        all_keys = set().union(*[t[0].keys() for t in list_of_tuples])
        
        # Ensure all arrays are 2D before concatenation
        merged_sub_dict = {}
        for key_in_all_keys in all_keys:
            concatenated_arrays = []
            for t in list_of_tuples:
                if key_in_all_keys in t[0]:
                    arr = t[0][key_in_all_keys]
                    if arr.ndim == 1: arr = np.expand_dims(arr, axis=0) # Make 1D array 2D
                    if arr.size > 0 : concatenated_arrays.append(arr) # Only append non-empty
            if concatenated_arrays:
                 merged_sub_dict[key_in_all_keys] = np.concatenate(concatenated_arrays)

        merged_strings = sum([list(t[1]) for t in list_of_tuples], [])
        if merged_sub_dict: # Only add if there's data
            unified_dict[k] = (merged_sub_dict, merged_strings)

    if not unified_dict: return ({}, []) # Handle empty unified_dict

    all_dict_keys = set().union(*[t[0].keys() for t in unified_dict.values()])
    
    final_dict = {}
    for k_final in all_dict_keys:
        arrays_to_concat = []
        for t_final in unified_dict.values():
            if k_final in t_final[0]:
                arr_final = t_final[0][k_final]
                if arr_final.ndim == 1: arr_final = np.expand_dims(arr_final, axis=0)
                if arr_final.size > 0: arrays_to_concat.append(arr_final)
        if arrays_to_concat:
            final_dict[k_final] = np.concatenate(arrays_to_concat)

    final_strings = sum([t[1] for t in unified_dict.values()], [])
    return (final_dict, final_strings)


def calculate_classwise_scores_svc(data, labels, npcs_values):
    """
    Calculates class-wise SVC scores after PCA.
    Adapted from playground_meta.ipynb (cell 6).
    """
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    unique_labels = sorted(list(np.unique(labels)))
    num_classes = len(unique_labels)

    # Map string labels to integers if they are not already
    if isinstance(labels[0], str):
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[l] for l in labels])
    else: # assume numerical
        int_labels = np.array(labels)


    for npcs in npcs_values:
        if data.shape[1] < npcs : # If number of features is less than npcs
            print(f"Warning: Number of features ({data.shape[1]}) is less than npcs ({npcs}). Using all features.")
            current_npcs = data.shape[1]
        else:
            current_npcs = npcs
        
        if current_npcs == 0: # Skip if no features
            print(f"Warning: Zero features available for PCA. Skipping npcs={npcs}.")
            results.append({
                'npcs': npcs, 'mean': [np.nan] * num_classes,
                'std': [np.nan] * num_classes, 'var_tot': np.nan
            })
            continue

        pca = PCA(n_components=min(current_npcs, data.shape[0])) # n_components cannot exceed n_samples
        X_red = pca.fit_transform(data)

        per_class = {c: [] for c in range(num_classes)}
        for train, test in skf.split(X_red, int_labels):
            if len(np.unique(int_labels[train])) < num_classes:
                print(f"Warning: Not all classes present in a training fold for npcs={npcs}. Skipping fold.")
                # This can lead to NaNs if all folds are skipped for a class
                for c_idx in range(num_classes): per_class[c_idx].append(np.nan)
                continue

            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            clf.fit(X_red[train], int_labels[train])
            preds = clf.predict(X_red[test])
            
            cm = confusion_matrix(int_labels[test], preds, labels=list(range(num_classes)))
            with np.errstate(divide='ignore', invalid='ignore'): # Handle division by zero
                accs = cm.diagonal() / cm.sum(axis=1)
            
            for c_idx in range(num_classes):
                per_class[c_idx].append(accs[c_idx] if np.isfinite(accs[c_idx]) else np.nan)
        
        results.append({
            'npcs': npcs,
            'mean': [np.nanmean(per_class[c]) if per_class[c] else np.nan for c in range(num_classes)],
            'std': [np.nanstd(per_class[c]) if per_class[c] else np.nan for c in range(num_classes)],
            'var_tot': sum(pca.explained_variance_ratio_)
        })
    return results


def plot_svc_accuracy(van_results, rob_results, exp_keys_source, display_names_map,
                      save_dir=None, filename='pca_svm_acc.png'):
    """
    Plots SVC accuracy results.
    Adapted from playground_meta.ipynb (cell 7).
    `exp_keys_source` is used to derive layer labels, typically from one of the experiment dicts.
    """
    fig, ax1 = plt.subplots(figsize=(12, 9)) # Adjusted size
    markers = ['o', 's', '^', 'D', 'v', '*'] # Added more markers

    styles = [
        (van_results, 'Standard', '#e57373'), # Reddish for standard
        (rob_results, 'Robust',   '#1b9e77')  # Greenish for robust
    ]

    all_npcs = sorted(list(set(d['npcs'] for d_list in [van_results, rob_results] for d in d_list)))

    num_classes = 0
    if van_results and 'mean' in van_results[0] and isinstance(van_results[0]['mean'], list):
        num_classes = len(van_results[0]['mean'])
    elif rob_results and 'mean' in rob_results[0] and isinstance(rob_results[0]['mean'], list):
        num_classes = len(rob_results[0]['mean'])
    
    if num_classes == 0:
        print("Error: Could not determine number of classes for SVC plot.")
        return

    
    id_to_disp = {}
    if exp_keys_source:
        # Sort keys to ensure consistent mapping if order matters
        # E.g., sort by the layer code part of the key
        sorted_exp_keys = sorted(list(exp_keys_source), key=lambda k: k.split('#')[3] if len(k.split('#')) > 3 else k)

        for cls_idx, exp_key_full in enumerate(sorted_exp_keys):
            if cls_idx >= num_classes: break # only map for existing classes
            parts = exp_key_full.split('#')
            if len(parts) > 3:
                layer_code = parts[3]
                id_to_disp[cls_idx] = display_names_map.get(layer_code, layer_code) 
            else: 
                id_to_disp[cls_idx] = f"Class {cls_idx}"


    for res_list, label_root, color in styles:
        for cls in range(num_classes): 
            if not res_list: continue
            npcs_vals  = [d['npcs'] for d in res_list if d and 'npcs' in d]
            means = [d['mean'][cls] if d and 'mean' in d and len(d['mean']) > cls else np.nan for d in res_list]
            stds  = [d['std' ][cls] if d and 'std' in d and len(d['std']) > cls else np.nan for d in res_list]
            
       
            valid_indices = [i for i, m in enumerate(means) if pd.notna(m)]
            if not valid_indices: continue

            plot_npcs = [npcs_vals[i] for i in valid_indices]
            plot_means = [means[i] for i in valid_indices]
            plot_stds = [stds[i] for i in valid_indices]


            ax1.plot(
                plot_npcs, plot_means,
                color=color, marker=markers[cls % len(markers)], linestyle='-',
                linewidth=3, markersize=12, 
                markeredgecolor='white', markeredgewidth=1.5,
                label=f'{label_root} - {id_to_disp.get(cls, f"Layer {cls}")}',
                clip_on=False
            )
            lo, hi = np.array(plot_means) - np.array(plot_stds), np.array(plot_means) + np.array(plot_stds)
        
            if not np.allclose(lo, hi, equal_nan=True):
                 ax1.fill_between(plot_npcs, lo, hi, color=color, alpha=0.15, clip_on=False)


    ax1.set_xlabel('Number of Principal Components', fontsize=16)
    ax1.set_ylabel('Discrimination Accuracy', fontsize=16)
    ax1.spines['left'].set_position(('outward', 15))
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.set_xticks(all_npcs)
    ax1.tick_params(axis='both', direction='out', labelsize=14, length=10, width=2, pad=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.3, 1.01)
    if all_npcs: ax1.set_xlim(min(all_npcs), max(all_npcs))

    legend_handles = []
    plotted_labels = set()
    for res_list, label_root, color in styles:
        for cls in range(num_classes):
            label_text = f'{label_root} - {id_to_disp.get(cls, f"Layer {cls}")}'
            if label_text not in plotted_labels:
                legend_handles.append(
                    Line2D([0], [0], color=color, marker=markers[cls % len(markers)], linestyle='-',
                           linewidth=2, markersize=10, markeredgecolor='white',
                           label=label_text, clip_on=False)
                )
                plotted_labels.add(label_text)
    
    if legend_handles:
        ax1.legend(handles=legend_handles, loc='lower right', fontsize=12, frameon=False)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Saved SVC accuracy plot to {os.path.join(save_dir, filename)}")
    plt.show()


# --- Utilities for Distance Analysis ---

def run_full_distance_analysis(prms, sns_metadata_path, DEVICE, base_output_dir):
    """
    Orchestrates the full distance analysis pipeline.
    Adapted from distance_analysis.ipynb (cell 1).
    """
    # Setup ref net and generator based on prms
    ref_net_prms = prms['ref_net']
    if ref_net_prms['robust'] == 'imagenet_l2_3_0.pt':
        net_load_func = madryLab_robust_load
        wp = os.path.join(CUSTOM_WEIGHTS, ref_net_prms['net_name'], ref_net_prms['robust'])
        net_nickname = ref_net_prms['net_name'] + '_l2'
    elif ref_net_prms['robust'] == '':
        net_load_func = torch_load
        wp = ''
        net_nickname = ref_net_prms['net_name']
    else:
        net_nickname = ref_net_prms['net_name'] + '_linf'
        net_load_func = robustBench_load
        wp = ref_net_prms['robust']

    # Temporary subject to get layer names
    _repr_net_temp = TorchNetworkSubject(
        network_name=ref_net_prms['net_name'],
        t_net_loading=net_load_func,
        custom_weights_path=wp
    )
    probe = RecordingProbe(target={ln: [] for ln in _repr_net_temp.layer_names})
    repr_net = TorchNetworkSubject(
        record_probe=probe,
        network_name=ref_net_prms['net_name'],
        t_net_loading=net_load_func,
        custom_weights_path=wp
    ).eval().to(DEVICE)

    generator = DeePSiMGenerator(
        root=str(WEIGHTS),
        variant=str(ref_net_prms['gen'])
    ).to(DEVICE)

    result_dict, acc_dict, sing_ans_dict, image_dict_collages = {}, {}, {}, {}
    
    neurons2view = prms['plotting']['neurons2view']
    if not neurons2view and prms['plotting']['generate_collages']:
        available_neurons = []
        for _, v_xdream in prms.get('XDREAM', {}).items():
            refs = load_pickle(v_xdream['fp'])
            available_neurons.extend(list(refs.get('reference', {}).get(v_xdream['net'], {}).get(v_xdream['gen'], {}).get(v_xdream['ly'], {}).keys()))
        neurons2view = list(set(available_neurons))

    # Process SNS_exp
    if "SNS_exp" in prms:
        sns_metadata_obj = SnS_metadata.from_json(sns_metadata_path)
        sns_metaexp = sns_metadata_obj.get_experiments(queries=prms['SNS_exp']['query'])
        for k_sns, v_sns in sns_metaexp.items():
            res_vsref, res_betw, acc, sns_imgs, sa_sns = distance_analysis_SnS(
                repr_net=repr_net, generator=generator, experiment=v_sns,
                p1=f"{prms['SNS_exp']['p1']}_p1_idxs",
                n2view=neurons2view if prms['plotting']['generate_collages'] else None,
                save_name=net_nickname
            )
            image_dict_collages.update(sns_imgs)
            acc_dict.update(acc)
            sing_ans_dict.update(sa_sns)
            result_dict.update(res_vsref)
            result_dict.update(res_betw)

    # Process XDREAM
    if "XDREAM" in prms:
        for xdk, xdv in prms['XDREAM'].items():
            xd_lbl = f'mXDREAM - {xdk}'
            res_xd, acc_xd, xd_imgs, sa_xd = distance_analysis_XDREAM(
                repr_net=repr_net, generator=generator, path_to_refs=xdv['fp'],
                opt_net=xdv['net'], opt_gen=xdv['gen'], opt_trgt=xdv['ly'],
                exp_name=xd_lbl,
                n2view=neurons2view if prms['plotting']['generate_collages'] else None,
                save_name=net_nickname
            )
            acc_dict[xd_lbl] = acc_xd
            sing_ans_dict[xd_lbl] = sa_xd
            result_dict.update(res_xd)
            image_dict_collages[xd_lbl] = xd_imgs # Storing XDREAM images under its label

    # Process nats
    if "nats" in prms:
        for t_nats in prms['nats']['types']:
            res_nats = distance_analysis_nat_imgs(
                repr_net=repr_net, n_samples=prms['nats']['n_samples'],
                rnd=(t_nats == 'rand'),
                savepath=os.path.join(os.path.dirname(prms["XDREAM"]["vanilla"]['fp']), 'distances_XDREAM_nats') if "XDREAM" in prms and "vanilla" in prms["XDREAM"] else "./distances_nats",
                save_name=net_nickname
            )
            result_dict.update(res_nats)

    analysis_dir = os.path.join(base_output_dir, "distance_analysis", prms['exp_name'])
    os.makedirs(analysis_dir, exist_ok=True)
    save_json(prms, os.path.join(analysis_dir, 'params.json'))

    results_avg = {k: aggregate_df(df=v, f_aggr_betw_cells=partial(np.mean, axis=0)) for k, v in result_dict.items()}
    results_sem = {k: aggregate_df(df=v, f_aggr_betw_cells=lambda x: np.std(x, axis=0) / np.sqrt(x.shape[0] if x.shape[0] > 0 else 1)) for k, v in result_dict.items()}
    
    results_df_avg = pd.DataFrame.from_dict(results_avg, orient='index')
    results_df_sem = pd.DataFrame.from_dict(results_sem, orient='index')
    
    norm_var = 'nat_images - same cat' 
    if norm_var in results_df_avg.index:
        results_df_avg_norm = results_df_avg.div(results_df_avg.loc[norm_var])
        results_df_sem_norm = results_df_sem.div(results_df_avg.loc[norm_var]) 
    else:
        print(f"Warning: Normalization variable '{norm_var}' not found in results_df_avg. Skipping normalization.")
        results_df_avg_norm = pd.DataFrame() 
        results_df_sem_norm = pd.DataFrame()

    results_df_avg.to_csv(os.path.join(analysis_dir, 'results_distances_avg.csv'))
    if not results_df_avg_norm.empty:
        results_df_avg_norm.to_csv(os.path.join(analysis_dir, 'results_distances_avg_normalized.csv'))
    
    # Save accuracy and single answers
    with open(os.path.join(analysis_dir, 'accuracy_summary.json'), 'w') as f: json.dump(acc_dict, f, indent=4)
    with open(os.path.join(analysis_dir, 'single_answers_summary.json'), 'w') as f: json.dump(sing_ans_dict, f, indent=4)

    # Plotting
    original_distance_plot(results_df_avg, results_df_sem,
                           savepath=os.path.join(analysis_dir, 'distance_plot.png'),
                           plotting_params=prms['plotting'])
    if not results_df_avg_norm.empty:
        original_distance_plot(results_df_avg_norm, results_df_sem_norm,
                               savepath=os.path.join(analysis_dir, 'distance_plot_normalized.png'),
                               y_lbl='Euclidean distance relative to same category variation',
                               plotting_params=prms['plotting'])
    
    original_plot_accuracy_distribution(accuracy_data=acc_dict, dist_params=prms,
                                        savepath=os.path.join(analysis_dir, 'accuracy_boxplot.png'))

    # Collage generation
    if prms['plotting']['generate_collages']:
        img_dict = {}
        d = defaultdict(list)
        units = [list(v.keys()) for v in image_dict_collages.values()]
        neurons2view = list(reduce(set.intersection, map(set, units)))
        to_pil = T.ToPILImage()
        for n in neurons2view:
            try:
                for k in image_dict_collages.keys():
                    variants = list(image_dict_collages[k][n]['variants'].values())
                    ref = list(image_dict_collages[k][n]['reference'].values())
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
                
                padded_imgs = pad_tensor_lists(tensor_lists=d)
                img_dict[n] = vertical_stack_images(padded_imgs,y_dist=50, font_size=40, margin=10)
            except:
                continue
        #save the images
        gen_imgs_dir = os.path.join(analysis_dir, 'gen_imgs')
        os.makedirs(gen_imgs_dir, exist_ok=True)
        {os.path.join(gen_imgs_dir, f"{k}.png"): v.save(os.path.join(gen_imgs_dir, f"{k}.png")) 
        for k, v in img_dict.items()}

    return {
        "results_df_avg": results_df_avg, "results_df_sem": results_df_sem,
        "results_df_avg_norm": results_df_avg_norm, "results_df_sem_norm": results_df_sem_norm,
        "acc_dict": acc_dict, "analysis_dir": analysis_dir
    }


# --- Utilities for Multi-Net Accuracy Analysis ---

def load_and_process_multinet_accuracies(search_pattern, dist_params_plotting_source_path):
    """
    Loads accuracy data from multiple JSON files and processes it.
    Adapted from multiNets.ipynb (cell 1, data loading part).
    """
    import glob
    plot_data = {}
    accuracy_files = glob.glob(search_pattern)

    if not accuracy_files:
        print(f"No accuracy.json files found with pattern: {search_pattern}")
        return pd.DataFrame(), pd.DataFrame(), {}, []

    # Load plotting_params from one of the dist_params.json files
    try:
        with open(dist_params_plotting_source_path, 'r') as f:
            plotting_params_content = json.load(f)
        plotting_params = plotting_params_content.get("plotting", {})
    except Exception as e:
        print(f"Warning: Could not load plotting params from {dist_params_plotting_source_path}: {e}. Using empty.")
        plotting_params = {}
        
    for f_path in accuracy_files:
        try:
            with open(f_path, 'r') as f_acc: data = json.load(f_acc)
            # Try to find params.json in the same directory as accuracy.json
            params_path = os.path.join(os.path.dirname(f_path), "params.json")
            if not os.path.exists(params_path):
                 print(f"Warning: params.json not found at {params_path} for {f_path}. Skipping this file.")
                 continue
            with open(params_path, 'r') as f_prms: prms_model = json.load(f_prms)
        except Exception as e:
            print(f"Error loading or parsing {f_path} or its params.json: {e}")
            continue

        model_name_parts = prms_model.get('ref_net', {})
        model_base_name = model_name_parts.get('net_name', 'UnknownModel')
        is_robust = bool(model_name_parts.get('robust', ''))
        model_label = f"{model_base_name}{' (Robust)' if is_robust else ' (Standard)'}" # Added (Standard) for clarity
        
        plot_data[model_label] = {}
        
        # Find common subkeys (categories like "[118]") across all top-level keys in this accuracy file
        if not data or not all(isinstance(subdict, dict) for subdict in data.values()):
            print(f"Warning: Data format issue in {f_path}. Skipping.")
            continue
        
        try:
            common_subkeys = set.intersection(*(set(subdict.keys()) for subdict in data.values()))
        except TypeError: # If data.values() is empty or not iterable of dicts
            print(f"Warning: Could not determine common subkeys in {f_path}. Skipping.")
            continue

        for case_name_original, category_accuracies in data.items():
            label = case_name_original
         
            found_label_in_plotting_params = False
            for pk_plot, pv_plot in plotting_params.items():
                if isinstance(pv_plot, dict) and case_name_original in pk_plot: 
                    label = pv_plot.get('label', case_name_original)
                    found_label_in_plotting_params = True
                    break
            if not found_label_in_plotting_params and case_name_original == "nat_refs":
                label = "Natural"
            
            if common_subkeys and isinstance(category_accuracies, dict):
                 plot_data[model_label][label] = np.mean([category_accuracies[k_sub] for k_sub in common_subkeys if k_sub in category_accuracies])
            elif not common_subkeys : 
                 plot_data[model_label][label] = np.nan 

    df = pd.DataFrame(plot_data)
    df_orig = df.copy()

    
    color_map = {'alexnet': np.array([0.267004, 0.004874, 0.329415, 1.]),
                 'convnext_base': np.array([0.190631, 0.407061, 0.556089, 1.]),
                 'resnet101': np.array([0.20803, 0.718701, 0.472873, 1.]),
                 'resnet18': np.array([0.565498, 0.84243, 0.262877, 1.]),
                 'resnet50': np.array([0.993248, 0.906157, 0.143936, 1.]),
                 'regnet_x_16gf': np.array([0.127568, 0.566949, 0.550556, 1.]),
                 'vgg16': np.array([0.267968, 0.223549, 0.512008, 1.])}
    

    all_base_names = sorted(list(set(c.split(' ')[0] for c in df.columns)))
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(all_base_names)))
    for i, base_name in enumerate(all_base_names):
        if base_name not in color_map:
            color_map[base_name] = viridis_colors[i]


    ordered_labels = [
        "Natural", "Spacing1", "Robust MEI", "Robust Pixel space",
        "Robust Layer3_conv1", "Robust Layer4_conv7",
        "Spacing2", "Standard MEI", "Standard Pixel space",
        "Standard Layer3_conv1", "Standard Layer4_conv7",
    ]

    df = df.reindex(ordered_labels)


    return df, df_orig, color_map, ordered_labels


def plot_multinet_accuracy_trends(df, ordered_labels, color_map, save_dir=None, filename="multinet_accuracy_trends.png"):
    """
    Plots accuracy trends for multiple networks.
    Adapted from multiNets.ipynb (cell 1, plotting part).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)
    
    
    # Identify robust and standard columns based on naming convention
    robust_cols = [c for c in df.columns if "(Robust)" in c]
    standard_cols = [c for c in df.columns if "(Standard)" in c or ("(Robust)" not in c and df[c].notna().any())] # include if not explicitly robust and has data


    for ax, tag_filter, title_suffix in zip(axes, ["(Robust)", "(Standard)"], ["Robust models", "Standard models"]):
        cols_to_plot = robust_cols if tag_filter == "(Robust)" else standard_cols
        
        if not cols_to_plot: # Skip if no columns for this subplot
            ax.set_title(title_suffix); ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        for c_col in cols_to_plot:
            base_model_name = c_col.split(' ')[0] # e.g., "resnet50" from "resnet50 (Robust)"
            ax.plot(df.index, df[c_col], color=color_map.get(base_model_name, 'gray'), marker='o', label=c_col, linewidth=2, markersize=8)
        
        plot_ticks = [i for i, lbl in enumerate(ordered_labels) if not lbl.startswith("Spacing")]
        plot_tick_labels = [lbl for lbl in ordered_labels if not lbl.startswith("Spacing")]
        
        ax.set_xticks(plot_ticks)
        ax.set_xticklabels(plot_tick_labels, rotation=45, ha='right', fontsize=12)
        ax.set_title(title_suffix, fontsize=16)
        ax.legend(fontsize=10, title="Model", loc="best")
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel("Mean Accuracy", fontsize=14)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Saved multi-net accuracy plot to {os.path.join(save_dir, filename)}")
    plt.show()


def perform_correlation_analysis_with_human(human_data_path, model_df_orig, color_map,
                                            save_dir=None, filename="model_human_correlations.png"):
    """
    Performs correlation analysis between model accuracies and human data.
    Adapted from multiNets.ipynb (cell 2).
    """
    try:
        df_human = pd.read_csv(human_data_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Human data file not found at {human_data_path}. Skipping correlation analysis.")
        return None, None

    df_human.index = df_human.index.str.replace("_group", "").str.replace("_", " ").str.replace(" conv", "_conv").str.replace("Space", "space")
    df_human.index = df_human.index.str.replace('natural', 'Natural', case=False) # Match capitalization
    df_human.rename(columns={'participant_accuracy': 'Avg_human'}, inplace=True)

    # Align indices for merging. Use common index labels.
    common_indices = model_df_orig.index.intersection(df_human.index)
    if common_indices.empty:
        print("Warning: No common indices between model data and human data. Correlation cannot be computed.")
        print(f"Model indices: {model_df_orig.index.tolist()}")
        print(f"Human indices: {df_human.index.tolist()}")
        return None, None
        
    merged_df = pd.concat([df_human.loc[common_indices], model_df_orig.loc[common_indices]], axis=1)
    merged_df.dropna(subset=['Avg_human'], inplace=True) # Drop rows if human data is NaN
    merged_df = merged_df.dropna(axis=1, how='all') # Drop model columns if all values are NaN after merge

    model_cols = [c for c in merged_df.columns if c != "Avg_human"]
    if not model_cols:
        print("Warning: No model columns left after merging and cleaning. Skipping correlation.")
        return None, None


    results = {col: pearsonr(merged_df["Avg_human"].dropna(), merged_df[col].dropna()) 
               for col in model_cols if merged_df["Avg_human"].notna().sum() > 1 and merged_df[col].notna().sum() > 1}
    
    # Filter out any results where correlation could not be computed (e.g. due to all NaNs after dropna within pearsonr call)
    valid_results = {k:v for k,v in results.items() if not (np.isnan(v[0]) or np.isnan(v[1]))}
    if not valid_results:
        print("Warning: No valid correlations computed. Skipping plot.")
        return None, None

    sorted_results = sorted(valid_results.items(), key=lambda x: x[1][0], reverse=True)
    correlations = {model: stats[0] for model, stats in sorted_results}
    p_values = {model: stats[1] for model, stats in sorted_results}

    bar_colors = [color_map.get(col.split(' ')[0], "gray") for col in correlations.keys()]

    plt.figure(figsize=(12, 7)) # Adjusted size
    plt.bar(correlations.keys(), correlations.values(), color=bar_colors)
    plt.ylabel("Pearson correlation with Avg_human", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title("Correlation of Models with Human Average Performance", fontsize=16)
    plt.ylim(min(0, min(correlations.values()) - 0.1) if correlations else 0, 1.0) # Adjusted ylim
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Saved model-human correlation plot to {os.path.join(save_dir, filename)}")
    plt.show()

    print("Correlations (r-value, p-value):")
    for model, r_val in correlations.items():
        print(f"  {model}: r = {r_val:.3f}, p = {p_values[model]:.3g}")

    return correlations, p_values