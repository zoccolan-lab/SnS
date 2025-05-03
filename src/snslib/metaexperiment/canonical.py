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

    Returns:
        results: Dict[str, Dict[str, Dict[str, List[float]]]]
            A nested dictionary with keys 'rotation', 'translation', 'scaling', each containing:
                - 'params': list of transformation parameters
                - 'ref_distances': Dict[unit, List[float]]
                - 'ref_activations': Dict[unit, List[float]]
                - 'nat_distances': Dict[unit, List[float]]
                - 'nat_activations': Dict[unit, List[float]]
    """
    # Prepare experiment data
    fp = experiment['path'][0]
    df = experiment['df']

    # Identify unique target units
    units = df['high_target'].unique()
    lower_ly = df['lower_ly'].iloc[0]

    # Load data pickle and parsing utilities
    data_pkl = load_pickle(os.path.join(fp, 'data.pkl'))
    Param_nat_stats = NATURAL_RECORDINGS
    # Define transformation parameters
    angles = [0,45 , 90, 135,180]
    percents = [0.0,0.05,0.1, 0.25, 0.5]
    # translations = [(-80, 0), (80, 0), (0, -80), (0, 80),(0,0)]  # (x, y) pixel shifts
    scales = [0.2,0.4,0.6,0.8,1.0]                  # scaling factors

    # Prepare converters
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    # Load stimulus paths and natural recordings
    SP_PATH = '/home/ltausani/Desktop/Zout/NeuralRecording/neurec_rnet50_vaniGlia_rout_and_lbls/neurec_rnet50_vaniGlia_rout_and_lbls-0/stim_paths.npy'
    stim_paths = np.load(SP_PATH, allow_pickle=True)
    nat_recs = load_pickle(NATURAL_RECORDINGS)

    # Initialize results dictionary
    results: Dict[str, Any] = {
        'rotation': {
            'params': angles,
            'ref_distances': {}, 'ref_activations': {},
            'nat_distances': {}, 'nat_activations': {}
        },
        'translation': {
            'params': percents,
            'ref_distances': {}, 'ref_activations': {},
            'nat_distances': {}, 'nat_activations': {}
        },
        'scaling': {
            'params': scales,
            'ref_distances': {}, 'ref_activations': {},
            'nat_distances': {}, 'nat_activations': {}
        }
    }

    # Iterate over each target unit
    for n in units: #for each target unit
        idxs = df[df['high_target'] == n].index.tolist()
        n = str(n).replace('(', '[').replace(')', ']')
        if n.isdigit(): n = f"[{n}]"
        
        ref_info = [data_pkl['reference_info'][i] for i in idxs][0]
        ref_ly = df.loc[idxs]['upper_ly'].unique()[0] #NOTE: not the best way to do this
        net = df.loc[idxs]['net_sbj'].unique()[0]
        net = net + '_r' if df.loc[idxs]['robust'].unique()[0] else net
        ref_info['layer'] = ref_ly
        ref_fp = ref_info.pop('ref_file')
        ref_info['code'] = 'code'
        ref_info = {'network': net, **ref_info}
        code_ref = ref_code_recovery(reference_file = load_pickle(REFERENCES), 
                                        keys = ref_info, ref_file_name = ref_fp)
        ref_stimulus = generator(codes=code_ref)

        # Prepare original input tensor
        ref_img = ref_stimulus[0].cpu()
        orig_inp = to_tensor(to_pil(ref_img)).unsqueeze(0).to(DEVICE)

        ref_ly   = df.loc[idxs, 'upper_ly'].unique()[0]
        net_key  = df.loc[idxs, 'net_sbj'].unique()[0]
        net_type = 'robust_l2' if df.loc[idxs, 'robust'].unique()[0] else 'vanilla'

        # load the natural recordings for this layer
        nat_recs  = load_pickle(Param_nat_stats)
        nat_layer = deep_get(nat_recs, [net_key, net_type, ref_ly])
        labels    = nat_layer['labels']

        # parse the high_target and find the linear index in `labels`
        raw_ht = df.loc[idxs, 'high_target'].iloc[0]
        ht = ast.literal_eval(raw_ht) if isinstance(raw_ht, str) else raw_ht
        if isinstance(ht, (list, tuple)) and len(ht) == 3:
            linear_idx = next(i for i, triple in enumerate(zip(*labels)) if triple == tuple(ht))
        else:
            key = ht[0] if isinstance(ht, (list, tuple)) else ht
            linear_idx = int(np.where(labels == key)[0][0])

        # pick the maximally activating natural image
        row      = nat_layer['data'][linear_idx]
        max_idx  = int(np.argmax(row))
        max_path = stim_paths[max_idx]
      # ensure 3‐channel RGB for the network’s preprocessing
        nat_pil  = Image.open(max_path).convert('RGB')
        #check by TAU
        resize_transform = transforms.Resize((256, 256))
        nat_pil = resize_transform(nat_pil) # Resize the image
        orig_nat = to_tensor(nat_pil).unsqueeze(0).to(DEVICE)


        # 1) get the original reference image tensor
        orig_feat_states = repr_net(stimuli=orig_inp)
        feat = orig_feat_states[lower_ly]
        if isinstance(feat, torch.Tensor):
            orig_feat_vec = feat.view(-1).cpu().numpy()
        else:
            # NumPy array → flatten with ravel()
            orig_feat_vec = np.asarray(feat).ravel()


        nat_states = repr_net(stimuli=orig_nat)
        feat_nat   = nat_states[lower_ly]
        if isinstance(feat_nat, torch.Tensor):
            orig_nat_vec = feat_nat.view(-1).cpu().numpy()
        else:
            orig_nat_vec = np.asarray(feat_nat).ravel()


        # Helper to compute distance & activation
        # def eval_transform(pil_img, orig_tensor):
        #     inp = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
        #     dist = euclidean(
        #         orig_tensor.view(-1).cpu().numpy(),
        #         inp.view(-1).cpu().numpy()
        #     )
        #     states = repr_net(stimuli=inp)
        #     rep = states[ref_ly]
        #     arr = rep.view(-1).cpu().numpy() if isinstance(rep, torch.Tensor) \
        #         else np.asarray(rep).ravel()
        #     act = float(arr[linear_idx])
        #     return dist, act
        
        
        def eval_transform(pil_img, orig_vec):
            # Convert PIL → tensor → add batch dim → device
            inp = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
            states = repr_net(stimuli=inp)

            # 1) FEATURE‐SPACE distance at lower_ly
            feat = states[lower_ly]
            if isinstance(feat, torch.Tensor):
                feat_vec = feat.view(-1).cpu().numpy()
            else:
                feat_vec = np.asarray(feat).ravel()
            feat_dist = euclidean(orig_vec, feat_vec)

            # 2) UPPER‐LY activation for plotting on the y-axis
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
            d_nat, a_nat = eval_transform(
                transforms.functional.rotate(nat_pil, ang),
                orig_nat_vec
            )
            results['rotation']['ref_distances'].setdefault(key, []).append(d_ref)
            results['rotation']['ref_activations'].setdefault(key, []).append(a_ref)
            results['rotation']['nat_distances'].setdefault(key, []).append(d_nat)
            results['rotation']['nat_activations'].setdefault(key, []).append(a_nat)

        # 2) Translations
        ref_pil = to_pil(ref_img); w,h = ref_pil.size
        for p in percents:
            # define shifts
            if p==0:
                shifts = [(0,0)]
            else:
                dx, dy = int(p*w), int(p*h)
                shifts = [( dx,0),(-dx,0),(0, dy),(0,-dy)]
            drs, ars, dns, ans = [],[],[],[]
            for tx,ty in shifts:
                pil_r = transforms.functional.affine(ref_pil, 0, (tx,ty),1.0,0)
                pil_n = transforms.functional.affine(nat_pil, 0, (tx,ty),1.0,0)
                d_r, a_r = eval_transform(pil_r, orig_feat_vec)
                d_n, a_n = eval_transform(pil_n, orig_nat_vec)
                drs.append(d_r); ars.append(a_r)
                dns.append(d_n); ans.append(a_n)
            results['translation']['ref_distances'].setdefault(key,[]).append(np.mean(drs))
            results['translation']['ref_activations'].setdefault(key,[]).append(np.mean(ars))
            results['translation']['nat_distances'].setdefault(key,[]).append(np.mean(dns))
            results['translation']['nat_activations'].setdefault(key,[]).append(np.mean(ans))

        # 3) Scaling + black padding
        for s in scales:
            pil_ref = transforms.functional.affine(
                to_pil(ref_img), angle=0, translate=(0, 0), scale=s, shear=0
            )
            pil_nat = transforms.functional.affine(
                nat_pil, angle=0, translate=(0, 0), scale=s, shear=0
            )
            d_ref, a_ref = eval_transform(pil_ref,orig_feat_vec)
            d_nat, a_nat = eval_transform(pil_nat,orig_nat_vec)
            results['scaling']['ref_distances'].setdefault(key, []).append(d_ref)
            results['scaling']['ref_activations'].setdefault(key, []).append(a_ref)
            results['scaling']['nat_distances'].setdefault(key, []).append(d_nat)
            results['scaling']['nat_activations'].setdefault(key, []).append(a_nat)

    return results
