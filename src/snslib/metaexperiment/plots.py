
# --- PLOTTING ---
import os
from typing import Dict, List, Literal, TypeVar
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd


import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont

from src.experiments.AdversarialAttack_BMM.plots import avg_spline
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
    from metaexperiment.canonical import distance_analysis_transformations, build_subject_and_generator
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
        'rotation': '#8C564B',
        'translation': '#1E88E5',
        'scaling': '#2CA02C',
    }

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
        if plot_type == 'centroid':
            ax.errorbar(
                x_last.mean(), y_last.mean(),
                xerr=x_last.std(), yerr=y_last.std(),
                fmt='o', color=v['color'], label=v['label'], markersize=10, capsize=5
            )
        elif plot_type == 'scatter':
            ax.scatter(x_last, y_last, c=v['color'], alpha=0.1, s=10)
            ax.scatter(x_last.mean(), y_last.mean(), c=v['color'], marker='*', s=200, label=v['label'])
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
        ax.axhline(nat_first.mean(), color=v['color'], linestyle='--')

    
        if canonical:
            # caching key: (net_name, upper_ly, robust_flag, lower_ly)
            parts = label.split('#')  # adjust if your labels differ
            group_key = tuple(parts[:4])
            if group_key not in transform_cache:
                repr_net, generator = build_subject_and_generator(label)
                transform_cache[group_key] = distance_analysis_transformations(
                    repr_net=repr_net,
                    generator=generator,
                    experiment=v
                )
            trans = transform_cache[group_key]

            # optional pixel-dist map
            mpd_map = df.set_index('high_target')['max_pix_dist'].to_dict() if x_norm and lower_ly=='00_input_01' else {}

            for ttype, tres in trans.items():
                params = tres['params']
                units = list(tres['ref_distances'].keys())
                # find identity index
                id_param = {'rotation':0,'translation':0.0,'scaling':1.0}[ttype]
                if id_param not in params: continue
                id_idx = params.index(id_param)

                avg_ref_x, avg_ref_y = [], []
                avg_nat_x, avg_nat_y = [], []
                for j in range(len(params)):
                    xr = [tres['ref_distances'][u][j] for u in units]
                    yr = [tres['ref_activations'][u][j] for u in units]
                    xn = [tres['nat_distances'][u][j] for u in units]
                    yn = [tres['nat_activations'][u][j] for u in units]
                    if y_norm:
                        yr = [(tres['ref_activations'][u][id_idx] -y) / tres['ref_activations'][u][id_idx] * 100 for u,y in zip(units, yr)]
                        yn = [(tres['ref_activations'][u][id_idx]-y) / tres['ref_activations'][u][id_idx] * 100 for u,y in zip(units, yn)]
                    if x_norm and lower_ly=='00_input_01':
                        xr = [x/ mpd_map[u]*100 for u,x in zip(units, xr)]
                        xn = [x/ mpd_map[u]*100 for u,x in zip(units, xn)]
                    avg_ref_x.append(np.mean(xr)); avg_ref_y.append(np.mean(yr))
                    avg_nat_x.append(np.mean(xn)); avg_nat_y.append(np.mean(yn))

                c = TRANSFORM_COLORS.get(ttype, 'k')
                if ttype == 'scaling':
                                    # Sort scaling parameters by distance from identity (1.0)
                    order_p = np.argsort([abs(p - 1.0) for p in params])
                else:
                    # Default sort for rotation and translation
                    order_p = np.argsort(params)                
                sorted_params = np.array(params)[order_p]
                sorted_ref_x_by_param = np.array(avg_ref_x)[order_p]
                sorted_ref_y_by_param = np.array(avg_ref_y)[order_p]
                sorted_nat_x_by_param = np.array(avg_nat_x)[order_p]
                sorted_nat_y_by_param = np.array(avg_nat_y)[order_p]

                # Plot reference points connected by parameter order
                ax.plot(sorted_ref_x_by_param, sorted_ref_y_by_param, marker='o', linestyle='-', color=c)
                # Add text annotations for reference points
                # for px, py, param in zip(sorted_ref_x_by_param, sorted_ref_y_by_param, sorted_params):
                #     ax.text(px + 0.5, py + 0.5, f"{param:.2f}", fontsize=8, color=c)

                # Plot natural points connected by parameter order
                ax.plot(sorted_nat_x_by_param, sorted_nat_y_by_param, marker='o', linestyle='--', color=c)
                # Add text annotations for natural points
                # for px, py, param in zip(sorted_nat_x_by_param, sorted_nat_y_by_param, sorted_params):
                #      # Slightly different offset for dashed line points to avoid overlap
                #     ax.text(px + 0.5, py - 0.5, f"{param:.2f}", fontsize=8, color=c, alpha=0.7)


    t_handles = [Line2D([0],[0], color=c, marker='o', linestyle='-')
                 for c in TRANSFORM_COLORS.values()]
    t_labels = list(TRANSFORM_COLORS.keys())
    s_handles = [
        Line2D([0],[0], color='gray', linestyle='-', label='Reference'),
        Line2D([0],[0], color='gray', linestyle='--', label='Natural')
    ]
    s_labels = [h.get_label() for h in s_handles]
    combined_handles = t_handles + s_handles
    combined_labels = t_labels + s_labels

    # Create and add the FIRST legend (Canonical Transformations) as an artist
    # Position it to the right of the plot
    combined_legend = ax.legend(
        combined_handles,
        combined_labels,
        title='Canonical Transformations',
        loc='upper left',
        bbox_to_anchor=(1.0001, 1), # Position outside axes (adjust x > 1)
        
    )
    ax.add_artist(combined_legend) # Add as an artist so it's not overwritten

    # 2. Get handles and labels for the EXPERIMENT legend (from errorbar/scatter calls)
    # This must be done *after* plotting the experiments but *before* the next legend call
    exp_handles, exp_labels = ax.get_legend_handles_labels()

    # Create the SECOND legend (Experiments) using the retrieved handles/labels
    # Position it below the plot
    if exp_handles: # Only create legend if there are labeled elements
        ax.legend(
            exp_handles,
            exp_labels,
            title='Experiments', # Optional title for the second legend
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1), # Position below axes (adjust y < 0)
            ncol=min(3, len(exp_handles)), # Adjust columns dynamically or set fixed
           
        )
    ax.set_title(f"Distance from reference - ending condition: {end_type} - Vanilla ResNet50", fontsize=20)
    ax.set_xlabel(
        f"Distance in {lower_ly}" + (" (% of max)" if x_norm else "")
    )
    ax.set_ylabel("Activation distance" + (" (% of reference)" if y_norm else ""))

    if lower_ly == '00_input_01':
        ax.set_xlim([0, 388] if not x_norm else [0, 100])
        ax.axvline(130, color='black', linestyle='--')
    if y_norm:
        ax.set_ylim([0, 130])       #150

  
    # plt.tight_layout(rect=[0, 0, 0.8, 1])  # Shrink the main plot to make room for legends on right
    
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