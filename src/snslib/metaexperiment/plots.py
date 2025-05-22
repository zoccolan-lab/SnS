
# --- PLOTTING ---
import os
from typing import Dict, List, Literal, TypeVar
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

import numpy as np
import pandas as pd


import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont

from src.experiments.Stretch_and_Squeeze.plots import avg_spline
from datetime import datetime


MODEL_CMAP = {
    'resnet50': plt.cm.get_cmap('Reds'),
    'convnextB': plt.cm.get_cmap('Purples'),
}

TRAIN_TYPE_LINESTYLE = {
    'vanilla': '-',
    'robust_l2': ':',
    'robust_linf': '--',
}

CONSTRAINTS = ['B10', 'B25', 'B50', 'B75', 'B100', 'Bnan','es_1', 'es_2']

def create_alpha_mapping(elements: list, m:int=0.2, M: int = 1) -> dict:
    """Map list elements to evenly spaced values between 0.2 and 1."""
    values = np.linspace(m, M, len(elements))
    return dict(zip(elements, values))

def nit_linewidth(nit: int) -> float:
    """Calculate the linewidth based on the number of iterations."""
    return 1 + nit / 100



# Step 1: Define the groups to plot
groups_to_plot = [
    {'task': 'invariance', 'train_type': 'vanilla', 'cmap':     plt.cm.get_cmap('Blues')},
    {'task': 'invariance', 'train_type': 'robust_l2', 'cmap':   plt.cm.get_cmap('Purples')},
    {'task': 'invariance', 'train_type': 'robust_linf', 'cmap': plt.cm.get_cmap('Greens')}
]

def natural_sort_key(s):
    """Extract numeric part from string for sorting"""
    if s == 'NaN':
        return float('inf')
    numbers = re.findall(r'\d+', str(s))
    return float(numbers[0]) if numbers else float('inf')


#TO BE IMPROVED:

def metaplot_lines(grouped_stats_df: pd.DataFrame, groups_to_plot: dict[dict]):
    constraints = grouped_stats_df['constraint'].unique()
    desired_order = constraints[np.argsort([natural_sort_key(x) for x in constraints])]

    for group in groups_to_plot:
        query = ' & '.join([f"{k} == '{v}'" for k,v in group.items() if k in grouped_stats_df.columns])
        #query = f"task == '{group['task']}' & train_type == '{group['train_type']}'"
        subset_df = grouped_stats_df.query(query)
        # Convert 'constraint' to a categorical type with the specified order
        subset_df['constraint'] = pd.Categorical(subset_df['constraint'], categories=desired_order, ordered=True)
        # Sort the DataFrame
        subset_df = subset_df.sort_values('constraint').reset_index(drop=True)
                
        x = subset_df['dist_low'].apply(lambda x:x['mean'])
        y = subset_df['dist_up_perc'].apply(lambda x:x['mean'])
        
        alphas = np.linspace(0.25, 1, len(y))
        colors = [group['cmap'](t) for t in alphas] 
        
        for _x, _y, c in zip(subset_df['dist_low'], subset_df['dist_up_perc'], colors):
            print(_x)
            plt.errorbar(
                x=abs(_x['mean']),
                y=abs(_y['mean']),
                xerr=_x['std'],
                yerr=_y['std'],
                marker='none',
                color=c,
                linestyle='None',
                zorder=10,
            )
        
        plt.scatter(
            x = x,
            y = y,
            label='_nolegend_',
            marker='o',
            linestyle='None',
            fc=colors,
            ec='black',
            s=50,
            zorder=10,
        )
        # Create a proxy artist with the final color for the legend
        plt.scatter([], [], label=f"{group['train_type']}", marker='o',
                    fc=colors[-1], ec='black', s=50)
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if len(segments) > 0:
            # Subdivide each segment into 100 steps
            subdivided_segments = []
            for seg in segments:
                x_vals = np.linspace(seg[0][0], seg[1][0], 101)
                y_vals = np.linspace(seg[0][1], seg[1][1], 101)
                subdivided = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
                subdivided_segments.append(np.concatenate([subdivided[:-1], subdivided[1:]], axis=1))
            subdivided_segments = np.concatenate(subdivided_segments, axis=0)

            # Create a LineCollection with progressive alpha
            total_segments = len(subdivided_segments)
            alphas = np.linspace(0.25, 1, total_segments)
            colors = [group['cmap'](t) for t in alphas]  # Blue color with varying alpha
            lc = LineCollection(subdivided_segments, colors=colors, linewidth=2)

            # Add the LineCollection to the plot
            plt.gca().add_collection(lc)

    plt.xlim(0, 388)
    plt.ylim(0, 100)
    # Step 3: Customize the plot
    plt.xlabel('\u0394 pixel')
    plt.ylabel('\u0394 neuron activation (%)')
    plt.title('Varying constraints to invariance experiments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.grid(False)

    plt.show()

#usage: metaplot_lines(grouped_stats_df, groups_to_plot)

def plot_grouped_splines(grouped_splines: dict, comparison_key: str = 'adversarial', 
                        fig_size: tuple = (12, 6)) -> None:
    """Plot splines for different model groups with consistent styling."""
    plt.figure(figsize=fig_size)

    for group_name, group_data in grouped_splines.items():
        nn_type, tt, it, c = group_name.split('#')
        color_map = MODEL_CMAP[nn_type]
        linestyle = TRAIN_TYPE_LINESTYLE[tt]
        linewidth = nit_linewidth(int(it))
        color_intensity = create_alpha_mapping(CONSTRAINTS)[c]
        color = color_map(color_intensity)
        if comparison_key in group_data:

            # Get spline data
            bounds = group_data[comparison_key]['xbounds']
            spline = group_data[comparison_key]['spline']
            # Evaluate spline
            x_eval = np.linspace(bounds[0], bounds[1], 1000)
            y_eval = spline(x_eval)

            plt.plot(x_eval, y_eval, label=group_name.replace('#', ' '), 
                    color=color, linestyle=linestyle, linewidth=linewidth)

    # Format plot
    plt.xlabel('\u0394 Pixel')
    plt.ylabel('\u0394 Neuron activation (%)')
    plt.ylim(0, 120)
    plt.legend(bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', 
              ncol=3)
    
    
#Multiexp image visualization
def concat_ref_vars(img_list, x_space: int = 20, ncol =2) -> Image.Image:
    """Concatenate images in a grid with the first column separated from the rest."""
    grid = to_pil_image(make_grid(img_list, nrow=ncol, padding=0))
    w, h = grid.size
    col_w = w // ncol
    
    new_img = Image.new(grid.mode, (w + x_space, h), (255, 255, 255))
    new_img.paste(grid.crop((0, 0, col_w, h)), (0, 0))
    new_img.paste(grid.crop((col_w, 0, w, h)), (col_w + x_space, 0))
    
    return new_img


T = TypeVar('T', List[List[torch.Tensor]], Dict[str, List[torch.Tensor]])

def pad_tensor_lists(tensor_lists: T) -> T:
    """Pad shorter tensor lists with empty tensors to match longest list."""
    if isinstance(tensor_lists, dict):
        is_dict = True
        keys, tensor_lists = list(tensor_lists.keys()), list(tensor_lists.values())
        print(len(tensor_lists))
    
    max_len = max(len(t_list) for t_list in tensor_lists)
    empty = torch.ones(tensor_lists[0][0].size())
    padded = [t_list + [empty] * (max_len - len(t_list)) for t_list in tensor_lists]
    print(empty.shape, max_len, padded[0][0].shape)
    
    return {k: concat_ref_vars([v.to('cuda') for v in pl], ncol =len(padded[0]), x_space=200) 
            for k, pl in zip(keys, padded)} if is_dict else padded
    
    
def vertical_stack_images(image_dict, y_dist, font_size=20, margin=10):
    """Stack images vertically with text labels from dict keys."""
    keys, imgs = zip(*image_dict.items())
    dims = [img.size for img in imgs]
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size) \
           or ImageFont.load_default()
    
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    text_width = max(draw.textlength(str(k), font=font) for k in keys) + 2 * margin
    
    total_size = (
        round(max(w for w, _ in dims) + text_width),
        round(sum(h for _, h in dims) + y_dist * (len(imgs) - 1))
    )
    
    result = Image.new('RGB', total_size, (255, 255, 255))
    draw = ImageDraw.Draw(result)
    
    y = 0
    for img, key in zip(imgs, keys):
        draw.text((margin, y + (img.height - font_size) // 2), str(key), font=font, fill='black')
        result.paste(img, (int(text_width), y))
        y += img.height + y_dist
    
    return result


from typing import Any



def SnS_scatterplot(
    exps: Dict[str, Dict[str, Any]],
    end_type: str = 'end',
    savepath: str | None = None,
    cmap: str = 'jet',
    y_norm: bool = True,
    x_norm: bool = False,  # solo per 00_input_01
    plot_type: Literal['scatter', 'centroid', 'splines'] = 'centroid',
    common_units: bool = False,
    canonical: bool = True,
) -> None:
    """
    Plot baseline distances/activations and overlay transformation centroids.
    Requires each exps[k] to include:
      - 'df': DataFrame with distance & activation series
      - 'label': str
      - 'repr_net', 'generator', 'experiment': inputs for distance_analysis_transformations
    """
    from src.snslib.metaexperiment.canonical import distance_analysis_transformations, build_subject_and_generator
    fig, ax = plt.subplots(figsize=(16, 16))
    # assign colors
    colors = cm.get_cmap(cmap, len(exps))
    for i, (k, v) in enumerate(exps.items()):
        if 'color' not in v:
            v['color'] = colors(i)

    # restrict to common units if requested
    if common_units:
        unit_sets = [
            set(v['df']['high_target'][v['df'][f'{end_type}_{v["df"]["lower_ly"].unique()[0]}'].notna()])
            for v in exps.values()
        ]
        common = set.intersection(*unit_sets)
        print(f"{len(common)} common units found")
        for v in exps.values():
            v['df'] = v['df'][v['df']['high_target'].isin(common)]
            
    transform_cache: Dict[tuple, Any] = {}
    TRANSFORM_COLORS = {
        'rotation': '#9932CC',  # darkorchid
        'translation': '#1E90FF',  # dodger blue
        'scaling': '#2E8B57',  # sea green
    }
    natural_data_plotted = False
    all_params_for_ttype = {k: set() for k in TRANSFORM_COLORS.keys()} # Initialize to collect params

    # plot baseline and transformations
    for label, v in exps.items():
        df = v['df'].copy()
        lower_ly = df['lower_ly'].iat[0]
        upper_ly = df['upper_ly'].iat[0]
        # drop nan rows
        df = df[df[f'{end_type}_{lower_ly}'].notna()]

        # baseline series
        x_series = df[f'{end_type}_{lower_ly}'].tolist()
        y_series = df[f'{end_type}_{upper_ly}'].tolist()
        ref_baseline = df['ref_activ'].tolist()

        # normalize y
        if y_norm:
            y_series = [np.asarray(arr) / ref_baseline[i] * 100
                        for i, arr in enumerate(y_series)]
        # normalize x if needed
        if x_norm and lower_ly == '00_input_01':
            max_pix = df['max_pix_dist'].tolist()
            x_series = [np.asarray(arr) / max_pix[i] * 100
                        for i, arr in enumerate(x_series)]

        # extract last point per unit
        x_last = np.array([abs(arr[-1]) for arr in x_series])
        y_last = np.array([abs(arr[-1]) for arr in y_series])

        # baseline plotting
        current_marker = v.get('marker', 'o') # Get marker, default to 'o'
        if plot_type == 'centroid':
            ax.errorbar(
                x_last.mean(), y_last.mean(),
                xerr=x_last.std(), yerr=y_last.std(),
                fmt=current_marker, color=v['color'], label=v['label'], markersize=25, capsize=5, markeredgecolor='white'
            )
        elif plot_type == 'scatter':
            ax.scatter(x_last, y_last, c=v['color'], marker=current_marker, alpha=0.1, s=10)
            ax.scatter(x_last.mean(), y_last.mean(), c=v['color'], marker='*', s=225, label=v['label'], ec='white')
        elif plot_type == 'splines':
            pass

        # plot natural threshold line
        # compute threshold 
        if y_norm:
            nat_series = (df['ref_activ'] - df['nat_thresh']) / df['ref_activ'] * 100
        else:
            nat_series = df['nat_thresh']
        # first value per unit
        nat_first = df.groupby('high_target').apply(lambda g: nat_series.loc[g.index].iloc[0])
        ax.axhline(nat_first.mean(), color=v['color'], linestyle='--', linewidth=2.5)

    
        if canonical:
            # caching key: (net_name, upper_ly, robust_flag, lower_ly)
            parts = label.split('#')  # adjust if your labels differ
            group_key = tuple(parts[:4])
            if group_key not in transform_cache:
                repr_net, generator = build_subject_and_generator(label)
                transform_cache[group_key] = distance_analysis_transformations(
                    repr_net=repr_net,
                    generator=generator,
                    experiment=v,
                    process_natural_images=False
                )
            trans = transform_cache[group_key]

            # optional pixel-dist map
            mpd_map = df.set_index('high_target')['max_pix_dist'].to_dict() if x_norm and lower_ly=='00_input_01' else {}

            for ttype, tres in trans.items():
                if ttype in all_params_for_ttype:
                    all_params_for_ttype[ttype].update(tres['params'])
                params = tres['params']
                units = list(tres['ref_distances'].keys())
                # find identity index
                id_param = {'rotation':0,'translation':0.0,'scaling':1.0}[ttype]
                if id_param not in params: continue
                id_idx = params.index(id_param)

                avg_ref_x, avg_ref_y = [], []
                std_ref_x, std_ref_y = [], [] # New: For standard deviations of reference
                
                has_natural_data = 'nat_distances' in tres and 'nat_activations' in tres

                if has_natural_data:
                    avg_nat_x, avg_nat_y = [], []
                    
                    std_nat_x, std_nat_y = [], [] # New: For standard deviations of natural
                
                for j in range(len(params)):
                    # --- Reference Data ---
                    ref_xr_j = [tres['ref_distances'][u][j] for u in units if u in tres['ref_distances'] and j < len(tres['ref_distances'][u])]
                    ref_yr_j = [tres['ref_activations'][u][j] for u in units if u in tres['ref_activations'] and j < len(tres['ref_activations'][u])]
                    
                    ref_data_valid_for_j = True
                    if not ref_xr_j or not ref_yr_j:
                        ref_data_valid_for_j = False
                    else:
                        # Normalization for reference
                        if y_norm:
                           
                            temp_yr_norm = []
                            original_indices_for_yr = [idx for idx, u in enumerate(units) if u in tres['ref_activations'] and j < len(tres['ref_activations'][u])]
                            
                            for original_idx, y_val in zip(original_indices_for_yr, ref_yr_j):
                                u = units[original_idx] # Get the unit corresponding to y_val
                                if u in tres['ref_activations'] and id_idx < len(tres['ref_activations'][u]):
                                    ref_val_at_id = tres['ref_activations'][u][id_idx]
                                    temp_yr_norm.append((ref_val_at_id - y_val) / ref_val_at_id * 100)
                                else: # Should not happen if ref_yr_j was built correctly and id_idx is valid for all those units
                                    temp_yr_norm.append(np.nan) 
                            ref_yr_j = [val for val in temp_yr_norm if not np.isnan(val)]


                        if x_norm and lower_ly=='00_input_01':
                            temp_xr_norm = []
                            original_indices_for_xr = [idx for idx, u in enumerate(units) if u in tres['ref_distances'] and j < len(tres['ref_distances'][u])]

                            for original_idx, x_val in zip(original_indices_for_xr, ref_xr_j):
                                u = units[original_idx]
                                if u in mpd_map:
                                    temp_xr_norm.append(x_val / mpd_map[u] * 100)
                                else: # Unit not in mpd_map
                                    temp_xr_norm.append(np.nan)
                            ref_xr_j = [val for val in temp_xr_norm if not np.isnan(val)]

                        if not ref_xr_j or not ref_yr_j: # Check if empty after normalization
                            ref_data_valid_for_j = False

                    if ref_data_valid_for_j:
                        avg_ref_x.append(np.mean(ref_xr_j))
                        avg_ref_y.append(np.mean(ref_yr_j))
                        std_ref_x.append(np.std(ref_xr_j))
                        std_ref_y.append(np.std(ref_yr_j))
                    else:
                        avg_ref_x.append(np.nan); avg_ref_y.append(np.nan)
                        std_ref_x.append(np.nan); std_ref_y.append(np.nan)

                    # --- Natural Data ---
                    if has_natural_data:
                        nat_xn_j = [tres['nat_distances'][u][j] for u in units if u in tres['nat_distances'] and j < len(tres['nat_distances'][u])]
                        nat_yn_j = [tres['nat_activations'][u][j] for u in units if u in tres['nat_activations'] and j < len(tres['nat_activations'][u])]

                        nat_data_valid_for_j = True
                        if not nat_xn_j or not nat_yn_j:
                            nat_data_valid_for_j = False
                        else:
                            if y_norm:
                                temp_yn_norm = []
                                original_indices_for_yn = [idx for idx, u in enumerate(units) if u in tres['nat_activations'] and j < len(tres['nat_activations'][u])]

                                for original_idx, y_val in zip(original_indices_for_yn, nat_yn_j):
                                    u = units[original_idx]
                                    if u in tres['ref_activations'] and id_idx < len(tres['ref_activations'][u]):
                                        ref_val_at_id = tres['ref_activations'][u][id_idx]
                                        temp_yn_norm.append((ref_val_at_id - y_val) / ref_val_at_id * 100)
                                    else:
                                        temp_yn_norm.append(np.nan)
                                nat_yn_j = [val for val in temp_yn_norm if not np.isnan(val)]
                            
                            if x_norm and lower_ly=='00_input_01':
                                temp_xn_norm = []
                                original_indices_for_xn = [idx for idx, u in enumerate(units) if u in tres['nat_distances'] and j < len(tres['nat_distances'][u])]
                                for original_idx, x_val in zip(original_indices_for_xn, nat_xn_j):
                                    u = units[original_idx]
                                    if u in mpd_map:
                                        temp_xn_norm.append(x_val / mpd_map[u] * 100)
                                    else:
                                        temp_xn_norm.append(np.nan)
                                nat_xn_j = [val for val in temp_xn_norm if not np.isnan(val)]

                            if not nat_xn_j or not nat_yn_j: # Check if empty after normalization
                                nat_data_valid_for_j = False
                        
                        if nat_data_valid_for_j:
                            avg_nat_x.append(np.mean(nat_xn_j))
                            avg_nat_y.append(np.mean(nat_yn_j))
                            std_nat_x.append(np.std(nat_xn_j))
                            std_nat_y.append(np.std(nat_yn_j))
                        else:
                            avg_nat_x.append(np.nan); avg_nat_y.append(np.nan)
                            std_nat_x.append(np.nan); std_nat_y.append(np.nan)


                c = TRANSFORM_COLORS[ttype]
                # order parameters
                if ttype == 'scaling':
                    order_p = np.argsort([abs(p - 1.0) for p in params])
                else:
                    order_p = np.argsort(params)

                # sort arrays
                sorted_ref_x = np.array(avg_ref_x)[order_p]
                sorted_ref_y = np.array(avg_ref_y)[order_p]
                sorted_std_x = np.array(std_ref_x)[order_p]
                sorted_std_y = np.array(std_ref_y)[order_p]

                # Filter NaNs for reference plotting
                valid_ref = ~np.isnan(sorted_ref_x) & ~np.isnan(sorted_ref_y)
                
                x_ref = sorted_ref_x[valid_ref]
                y_ref = sorted_ref_y[valid_ref]
                ex_ref = sorted_std_x[valid_ref]
                ey_ref = sorted_std_y[valid_ref]

                                # prepare gradient with stronger contrast
                base_rgba = to_rgba(TRANSFORM_COLORS[ttype])
                # pale = mix halfway with white
                pale = tuple(base_rgba[i] + (1 - base_rgba[i]) * 0.5 if i < 3 else base_rgba[3] for i in range(4))
                # strong = 50% darker than base
                strong = tuple(base_rgba[i] * 0.5 if i < 3 else base_rgba[3] for i in range(4))
                # interpolation t-values using sqrt for non-linear spacing
                tvals = np.sqrt(np.linspace(0, 1, len(avg_ref_x)))
                grads = [tuple(pale[j] + (strong[j] - pale[j]) * t for j in range(4)) for t in tvals]

                # sort and mask
                #order_p = np.argsort([abs(p - id_param) for p in params])
                x_ref = np.array(avg_ref_x)[order_p]
                y_ref = np.array(avg_ref_y)[order_p]
                ex_ref = np.array(std_ref_x)[order_p]
                ey_ref = np.array(std_ref_y)[order_p]
                valid = ~np.isnan(x_ref) & ~np.isnan(y_ref)
                x_ref, y_ref, ex_ref, ey_ref = x_ref[valid], y_ref[valid], ex_ref[valid], ey_ref[valid]
                grads = [tuple(pale[i] + (strong[i] - pale[i]) * t for i in range(4)) for t in tvals]


                # plot segments colored by first marker of each segment
                if len(x_ref) > 1:
                    pts = np.column_stack([x_ref, y_ref])
                    segs = [pts[i:i+2] for i in range(len(pts)-1)]
                    lc = LineCollection(segs, colors=grads[:-1], linewidths=4, zorder=10, alpha=0.9)
                    ax.add_collection(lc)

                # Plot points with errorbars
                for pt, c in enumerate(grads):
                    ax.errorbar(
                        x_ref[pt], y_ref[pt],
                        xerr=ex_ref[pt], yerr=ey_ref[pt],
                        fmt='o', color=c, markersize=25,
                        capsize=3, elinewidth=1,
                        markeredgecolor='white', zorder=11,clip_on=False
                    )
    t_handles = [Line2D([0],[0], color=TRANSFORM_COLORS[ttype_key], marker='o', linestyle='-', markeredgecolor='white', markersize=25)
                 for ttype_key in TRANSFORM_COLORS.keys()]

    # Create labels for transformation types with their parameters
    processed_t_labels = []
    exp_handles, exp_labels = ax.get_legend_handles_labels() 
    for ttype_key in TRANSFORM_COLORS.keys(): # Iterate in defined order
        base_label_map = {'scaling': 'Scaling', 'translation': 'Translation', 'rotation': 'Rotation'}
        base_name = base_label_map.get(ttype_key, ttype_key.capitalize())
        
        current_params_set = all_params_for_ttype.get(ttype_key, set())
        if current_params_set:
            # Special sorting for scaling to be around 1.0, others numerically
            if ttype_key == 'scaling':
                sorted_params_list = sorted(list(current_params_set), key=lambda p: abs(p - 1.0))
            else:
                sorted_params_list = sorted(list(current_params_set))

            param_strings = []
            for p_val in sorted_params_list:
                if isinstance(p_val, (float, int)): 
                    if ttype_key == 'rotation':
                        param_strings.append(f"{int(p_val)}\u00B0")
                    elif ttype_key == 'translation':
                        param_strings.append(f"{int(p_val*100)}%") 
                    elif ttype_key == 'scaling':
                        param_strings.append(f"{int(100-p_val*100)}%") 
                    else:
                        param_strings.append(f"{p_val}") 
                else:
                    param_strings.append(str(p_val))
            
            if len(param_strings) > 5: # Abbreviate if too many params
                param_display_str = f"{', '.join(param_strings[:2])}, ..., {', '.join(param_strings[-2:])}"
            else:
                param_display_str = ", ".join(param_strings)
            processed_t_labels.append(f"{base_name} ({param_display_str})") 
        else:
            processed_t_labels.append(base_name) # Fallback if no params were collected

    # Create handles and labels for Reference/Natural lines
    s_handles = [Line2D([0],[0], color='gray', linestyle='-', label='Reference')]
    if natural_data_plotted:
        s_handles.append(Line2D([0],[0], color='gray', linestyle='--', label='Natural'))
    s_labels = [h.get_label() for h in s_handles]
    
    # Combine all handles and labels in the desired order for a single column legend
    # Order: Experiments, then Transformations, then Reference/Natural
    final_handles = exp_handles + t_handles
    final_labels = exp_labels + processed_t_labels 
    
    # Ensure all legend markers have a consistent size
    for handle in final_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(15) # Consistent marker size for legend items
    
    # Create the legend
    if final_handles: # Only create legend if there are items to show
        combined_legend = ax.legend(
            final_handles,
            final_labels,
            ncol=1, # Single column
            loc='upper right', # Current location
            fontsize=27,       # Current fontsize
            frameon=False
        )
        ax.add_artist(combined_legend)

    # 2. Get handles and labels for the EXPERIMENT legend (from errorbar/scatter calls)
    # This must be done *after* plotting the experiments but *before* the next legend call




    # Position it below the plot
    # if exp_handles: # Only create legend if there are labeled elements
    #     ax.legend(
    #         exp_handles,
    #         exp_labels,
            
    #         loc='upper center',
    #         bbox_to_anchor=(0.5, -0.1), # Position below axes (adjust y < 0)
    #         ncol=min(3, len(exp_handles)), # Adjust columns dynamically or set fixed
    #         fontsize=32,
    #         frameon=False
    #     )
    #ax.set_title(f"Distance from reference - ending condition: {end_type} - Robust ResNet50", fontsize=20)
    
    ax.set_ylabel("Activation decrement relative to the most exciting image" + (" (%)" if y_norm else ""), fontsize=32)

    if lower_ly == '00_input_01':
        ax.set_xlim([-0.001, 350] if not x_norm else [0, 100])
        ax.axvline(130, color='black', linestyle='--', linewidth=2.5)
        ax.set_xlabel("Euclidean distance from the most exciting image (pixels)", fontsize=32)
    if y_norm:
        ax.set_ylim([-0.001, 140])       #150
        
    ax.spines['left'].set_position(('outward', 20))  
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_position(('outward', 20))  
    ax.spines['bottom'].set_linewidth(4)
    
    # Hide the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # Make ticks point outwards on both axes
    ax.tick_params(axis='both', direction='out', labelsize=26, length=18, width=4, pad=5)
  
    plt.tight_layout()  # Shrink the main plot to make room for legends on right
    
    # save if requested
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        idx = 0
       
        while os.path.exists(os.path.join(savepath, f"{plot_type}_{idx}.svg")):
            idx += 1
        plt.savefig(os.path.join(savepath, f"{plot_type}_{idx}.svg"), bbox_inches='tight')













def wrap_text(text, width=20):
    """
    Insert newline characters in a string, replacing the first space after each
    'width' characters.
    
    Args:
        text (str): The text to modify
        width (int): Target line length before wrapping
        
    Returns:
        str: Text with newlines inserted
    """
    if not text or width <= 0:
        return text
        
    result = ""
    last_break = 0
    
    i = 0
    while i < len(text):
        # If we've passed the width and found a space, replace it with newline
        if i - last_break >= width and text[i] == ' ':
            result += '\n'
            last_break = i + 1
        else:
            result += text[i]
        i += 1
            
    return result