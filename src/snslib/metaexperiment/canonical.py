import os
import ast
from typing import Any, Dict, List

import numpy as np
from snslib.core.utils.probe import RecordingProbe
from snslib.core.utils.torch_net_load_functs import madryLab_robust_load, robustBench_load, torch_load
import torch
from PIL import Image
from scipy.spatial.distance import euclidean
from torchvision import transforms
import matplotlib.pyplot as plt

from snslib.experiment.utils.args import CUSTOM_WEIGHTS, NATURAL_RECORDINGS, REFERENCES, WEIGHTS
from snslib.metaexperiment.distance_analysis import DEVICE
from snslib.core.utils.io_ import load_pickle
from snslib.experiment.utils.misc import ref_code_recovery
from snslib.core.subject import TorchNetworkSubject
from snslib.core.generator import DeePSiMGenerator
from snslib.core.utils.misc import deep_get


def build_subject_and_generator(label: str):
    """
    Given an experiment label (e.g. 'resnet50#…#robust_l2#…'), return
    (repr_net, generator) ready for your distance‐analysis calls.
    """
    net_name, up_ly, robust_flag,low_ly, *rest = label.split('#')
    trgt_lys = [ly.replace('conv', 'conv2d') for ly in [up_ly, low_ly]]
    # pick loader + weights
    if robust_flag == 'robust_l2':
        net_load, weight_file = madryLab_robust_load, 'imagenet_l2_3_0.pt'
    elif robust_flag == 'robust_linf':
        net_load, weight_file = robustBench_load, 'imagenet_linf.pt'
    else:
        net_load, weight_file = torch_load, ''
    wp = os.path.join(CUSTOM_WEIGHTS, net_name, weight_file) if weight_file else ''

    # build and eval subject
    repr_net = TorchNetworkSubject(
        network_name       = net_name,
        t_net_loading      = net_load,
        custom_weights_path= wp,
        record_probe       = RecordingProbe({
            ln: [] for ln in TorchNetworkSubject(net_name).layer_names if ln in trgt_lys
        })
    )
    repr_net.eval()

    # build the generator
    generator = DeePSiMGenerator(root=str(WEIGHTS), variant='fc7').to(DEVICE)
    return repr_net, generator





def distance_analysis_transformations(
    repr_net: TorchNetworkSubject,
    generator: DeePSiMGenerator,
    experiment: Dict[str, Any],
    
) -> Dict[str, Any]:
    """
    For each reference unit in the experiment, apply rotations, translations, and scaling (with black padding)
    to the reference stimulus and compute Euclidean distances in pixel space and the corresponding neuron activations.
    

    Args:
        repr_net: The network subject.
        generator: The generator for creating stimuli.
        experiment: Dictionary containing experiment data.
        

    Returns:
        results: Dict[str, Dict[str, Dict[str, List[float]]]]
            A nested dictionary with keys 'rotation', 'translation', 'scaling', each containing:
                - 'params': list of transformation parameters
                - 'ref_distances': Dict[unit_key, List[float]]
                - 'ref_activations': Dict[unit_key, List[float]]

    """
    # Prepare experiment data
    fp = experiment['path'][0]
    df = experiment['df']

    # Identify unique target units
    units = df['high_target'].unique()
    

    # Load data pickle and parsing utilities
    data_pkl = load_pickle(os.path.join(fp, 'data.pkl'))
    Param_nat_stats = NATURAL_RECORDINGS
    # Define transformation parameters
    angles = [0,45 , 90, 135,180]
    percents = [0.0,0.05,0.1, 0.25, 0.5]
    scales = [0.2,0.4,0.6,0.8,1.0]

    # Prepare converters
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

 

    # Initialize results dictionary
    results: Dict[str, Any] = {
        'rotation': {
            'params': angles,
            'ref_distances': {}, 'ref_activations': {},
        },
        'translation': {
            'params': percents,
            'ref_distances': {}, 'ref_activations': {},
        },
        'scaling': {
            'params': scales,
            'ref_distances': {}, 'ref_activations': {},
        }
    }



    # Iterate over each target unit
    for unit_identifier in units:
        idxs = df[df['high_target'] == unit_identifier].index.tolist()
        
        ref_info_list = [data_pkl['reference_info'][i] for i in idxs]
        if not ref_info_list: continue # Should not happen if units come from df
        ref_info = ref_info_list[0]

        ref_ly = df.loc[idxs, 'upper_ly'].unique()[0]
        lower_ly = df.loc[idxs, 'lower_ly'].iloc[0]
        net_sbj = df.loc[idxs, 'net_sbj'].unique()[0]
        is_robust = df.loc[idxs, 'robust'].unique()[0]
        
        net_for_ref_recovery = net_sbj + '_r' if is_robust else net_sbj
        
        # Create a mutable copy for ref_code_recovery
        current_ref_info = ref_info.copy()
        current_ref_info['layer'] = ref_ly
        ref_fp = current_ref_info.pop('ref_file')
        current_ref_info['code'] = 'code' # Placeholder, actual code is recovered
        current_ref_info = {'network': net_for_ref_recovery, **current_ref_info}
        
        code_ref = ref_code_recovery(reference_file=load_pickle(REFERENCES), 
                                     keys=current_ref_info, ref_file_name=ref_fp)
        ref_stimulus = generator(codes=code_ref)

        ref_img = ref_stimulus[0].cpu()
        orig_inp = to_tensor(to_pil(ref_img)).unsqueeze(0).to(DEVICE)

        # --- Natural recordings metadata, ht, current_unit_key, and linear_idx (ALWAYS NEEDED) ---
        net_key_for_nat_recs  = net_sbj # Use the base name for nat_recs lookup as per original structure
        net_type_for_nat_recs = 'robust_l2' if is_robust else 'vanilla' # Adjust if your nat_recs keys differ

        nat_recs_data = load_pickle(Param_nat_stats)
        nat_layer = deep_get(nat_recs_data, [net_key_for_nat_recs, net_type_for_nat_recs, ref_ly])
        labels    = nat_layer['labels']

        # unit_identifier is the raw value from df['high_target']
        # ht is the parsed version (e.g., int, or tuple from string)
        ht = ast.literal_eval(str(unit_identifier)) if isinstance(unit_identifier, str) else unit_identifier
        
        current_unit_key: Any # Declare type for clarity
        if isinstance(ht, (list, tuple)) and len(ht) == 3: # e.g. (channel, h, w)
            linear_idx = next(i for i, triple in enumerate(zip(*labels)) if triple == tuple(ht))
            current_unit_key = tuple(ht) 
        else: # e.g. channel index
            # _key_for_label_lookup should be the actual value present in 'labels'
            _key_for_label_lookup = ht[0] if isinstance(ht, (list, tuple)) and len(ht) == 1 else ht
            linear_idx = int(np.where(labels == _key_for_label_lookup)[0][0])
            current_unit_key = _key_for_label_lookup
        # --- End of always needed part for linear_idx & current_unit_key ---

        orig_feat_states = repr_net(stimuli=orig_inp)
        feat = orig_feat_states[lower_ly]
        if isinstance(feat, torch.Tensor):
            orig_feat_vec = feat.view(-1).cpu().numpy()
        else:
            orig_feat_vec = np.asarray(feat).ravel()

        
        
        # Define eval_transform here as it closes over linear_idx, ref_ly, lower_ly etc.
        def eval_transform(pil_img_to_eval, base_feature_vector):
            inp = to_tensor(pil_img_to_eval).unsqueeze(0).to(DEVICE)
            states = repr_net(stimuli=inp)

            feat_transformed = states[lower_ly]
            if isinstance(feat_transformed, torch.Tensor):
                feat_vec_transformed = feat_transformed.view(-1).cpu().numpy()
            else:
                feat_vec_transformed = np.asarray(feat_transformed).ravel()
            feat_dist = euclidean(base_feature_vector, feat_vec_transformed)

            rep = states[ref_ly]
            if isinstance(rep, torch.Tensor):
                arr = rep.view(-1).cpu().numpy()
            else:
                arr = np.asarray(rep).ravel()
            act = float(arr[linear_idx])
            return feat_dist, act

        # Apply each transformation type
        # 1) Rotations
        for ang in angles:
            d_ref, a_ref = eval_transform(
                transforms.functional.rotate(to_pil(ref_img), ang),
                orig_feat_vec
            )
            results['rotation']['ref_distances'].setdefault(current_unit_key, []).append(d_ref)
            results['rotation']['ref_activations'].setdefault(current_unit_key, []).append(a_ref)
            
            

        # 2) Translations
        pil_ref_img = to_pil(ref_img) # Use a distinct variable name
        w,h = pil_ref_img.size
        for p in percents:
            if p==0: shifts = [(0,0)]
            else:    dx, dy = int(p*w), int(p*h); shifts = [( dx,0),(-dx,0),(0, dy),(0,-dy)]
            
            drs, ars = [],[]
            for tx,ty in shifts:
                transformed_pil_r = transforms.functional.affine(pil_ref_img, 0, (tx,ty),1.0,0)
                d_r, a_r = eval_transform(transformed_pil_r, orig_feat_vec)
                drs.append(d_r); ars.append(a_r)
            results['translation']['ref_distances'].setdefault(current_unit_key,[]).append(np.mean(drs))
            results['translation']['ref_activations'].setdefault(current_unit_key,[]).append(np.mean(ars))
            
            

        # 3) Scaling + black padding
        for s_factor in scales: # Renamed s to s_factor
            transformed_pil_ref_scaling = transforms.functional.affine(
                to_pil(ref_img), angle=0, translate=(0, 0), scale=s_factor, shear=0
            )
            d_ref, a_ref = eval_transform(transformed_pil_ref_scaling, orig_feat_vec)
            results['scaling']['ref_distances'].setdefault(current_unit_key, []).append(d_ref)
            results['scaling']['ref_activations'].setdefault(current_unit_key, []).append(a_ref)
            
            
    return results

