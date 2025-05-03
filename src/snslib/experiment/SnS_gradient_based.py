import os
from typing import Any, Dict, Tuple, Type, cast
import numpy as np
from numpy.typing import NDArray
import torch
import torchjd
from snslib.experiment.utils.args import ExperimentArgParams
from snslib.experiment.utils.misc import ref_code_recovery
from snslib.experiment.utils.parsing import parse_net_loading, parse_recording, parse_reference_info, parse_signature
from snslib.metaexperiment.distance_analysis import DEVICE
from snslib.metaexperiment.metaexp_functs import NAT_STAT_AGGREGATOR, get_df_summary
from snslib.core.generator import DeePSiMGenerator
from snslib.core.subject import InSilicoSubject, TorchNetworkSubject
from snslib.core.utils.io_ import load_pickle
from snslib.core.utils.logger import Logger, LoguruLogger, SilentLogger
from snslib.core.utils.message import GDSnSMessage
from snslib.core.utils.misc import deep_get
from snslib.core.utils.parameters import ArgParams, ParamConfig
from snslib.core.utils.types import States, Stimuli
from snslib.core.utils.probe               import RecordingProbe_Grad
from torchjd.aggregation import UPGrad
import torch.optim as optim
import torchvision.transforms as transforms
from snslib.core.experiment import Experiment, GradientBasedExperimentState, MultiExperiment

class SnS_gradient_based(Experiment):
    
    EXPERIMENT_TITLE = "SnS_gradient_based"
    
    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject)
    
    # --- CONFIG ---

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'SnS_gradient_based':

        # Extract parameter from configuration and cast
        PARAM_weights     = str  (conf[ExperimentArgParams.GenWeights      .value])
        PARAM_variant     = str  (conf[ExperimentArgParams.GenVariant      .value])
        
        PARAM_net_name    = str  (conf[ExperimentArgParams.NetworkName     .value])
        PARAM_rec_layers  = str  (conf[ExperimentArgParams.RecordingLayers .value])
        PARAM_customW_path = str  (conf[ExperimentArgParams.CustomWeightsPath     .value])
        PARAM_customW_var  = str  (conf[ExperimentArgParams.CustomWeightsVariant  .value])
        PARAM_scr_sign    = str  (conf[ExperimentArgParams.ScoringSignature.value])
        PARAM_exp_name    = str  (conf[          ArgParams.ExperimentName  .value])
        PARAM_iter        = int  (conf[          ArgParams.NumIterations   .value])
        PARAM_rnd_seed    = int  (conf[          ArgParams.RandomSeed      .value])
        PARAM_ref         = str  (conf[ExperimentArgParams.Reference       .value])
        PARAM_ref_info    = str  (conf[ExperimentArgParams.ReferenceInfo   .value])
        PARAM_net_loading = str  (conf[ExperimentArgParams.WeightLoadFunction.value])
        PARAM_learning_rate = float(conf[ExperimentArgParams.LearningRate.value])
        
        Param_nat_stats      =str(conf[ExperimentArgParams.Nat_recs.value])    
        Param_nat_stats_aggr = str(conf[ExperimentArgParams.Nrec_aggregate.value])
        
        if PARAM_customW_var == 'imagenet_l2_3_0.pt': 
            net_type = 'robust_l2'
            path2CustomW = os.path.join(PARAM_customW_path, PARAM_net_name, PARAM_customW_var)
        elif PARAM_customW_var == '':
            net_type = 'vanilla' 
            path2CustomW = PARAM_customW_var
        else:
            net_type ='robust_linf'
            path2CustomW = PARAM_customW_var

        # Set numpy random seed
        
        np.random.seed(PARAM_rnd_seed)
        torch.random.manual_seed(PARAM_rnd_seed)
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root    = str(PARAM_weights),
            variant = str(PARAM_variant) # type: ignore
        ).to(DEVICE)
        
        # --- SUBJECT ---
        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(
            network_name=str(PARAM_net_name)
        ).layer_info
        
        record_target = parse_recording(input_str=PARAM_rec_layers, net_info=layer_info)
        #Get net loading function from parsing
        net_loading = parse_net_loading(input_str = PARAM_net_loading)
        # Subject with attached recording probe
        #probe = RecordingProbe(target = record_target) # type: ignore
        probe = RecordingProbe_Grad(target = record_target) # type: ignore
        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name,
            t_net_loading = net_loading,
            custom_weights_path = path2CustomW
        )

        # Set the network in evaluation mode
        sbj_net.eval()
        
        # get the  reference
        reference_file      = load_pickle(PARAM_ref)
        gen_var, layer, neuron, seed = parse_reference_info(PARAM_ref_info)
        ref_info = {'gen_var': gen_var, 'layer': layer, 'neuron': neuron, 'seed': seed,
                    'ref_file': PARAM_ref}

        # Extract code from reference file
        r = '_r' if path2CustomW else ''
        net_key = PARAM_net_name+r
        layer_name = list(layer_info.keys())[layer]

        ref_code = ref_code_recovery(reference_file = reference_file, 
                    keys = {'network': net_key, 
                            'gen_var': gen_var, 
                            'layer': layer_name,
                            'neuron': neuron,
                            'seed': seed,
                            'code':'code'}, 
                    ref_file_name = PARAM_ref)
                
        # Generate the code and the state, unbatching it
        ref_stimulus : Stimuli = generator(codes=ref_code)
        ref_states_b : States  = sbj_net(stimuli=ref_stimulus)
        print(f"Reference performance: {ref_states_b[list(record_target.keys())[-1]]}")

            
        #extract activation with best natural 
        if Param_nat_stats:
            try:#try to see if the threshold is a scalar
                nat_thresh = float(Param_nat_stats_aggr)
            except:
                nat_aggr = NAT_STAT_AGGREGATOR[Param_nat_stats_aggr]
                #get the natural recording of the layer of interest
                nat_recs = load_pickle(Param_nat_stats)
                nat_ly_rec = deep_get(dictionary= nat_recs, keys = [PARAM_net_name, net_type, layer_name])
                #get the index of the target neuron

                if len(sbj_net.target[layer_name])>1:
                    linear_idx = next((i for i, triple in enumerate(zip(*nat_ly_rec['labels'])) if triple == sbj_net.target[layer_name]), None)
                else:
                    linear_idx = np.where(nat_ly_rec['labels'] == sbj_net.target[layer_name][0])[0]
                #select the target unit activation distribution and aggregate it to get the threshold
                nat_ly_rec = np.expand_dims(nat_ly_rec['data'][linear_idx], axis=0) if nat_ly_rec['data'][linear_idx].ndim < 2 else nat_ly_rec['data'][linear_idx]
                nat_thresh = nat_aggr(nat_ly_rec)
            print(f"nat threshold {nat_thresh} found for cells {sbj_net.target[layer_name]}")
        else:
            nat_thresh = 0

        signature = parse_signature(
            input_str=str(PARAM_scr_sign),
            net_info=layer_info,
        )
    
        # --- DATA ---
        
        data = {
            'ref_image'     : ref_stimulus,
            'ref_states'    : ref_states_b,
            'ref_info'      : ref_info,
            'params'        : conf,
            'nat_thresh'    : nat_thresh,
            'rnd_seed'      : PARAM_rnd_seed,
            
        }

        # --- EXPERIMENT INSTANCE ---
        
        experiment = cls(
            subject        = sbj_net,
            iteration      = PARAM_iter,
            learning_rate  = PARAM_learning_rate,
            data           = data, 
            name           = PARAM_exp_name,
            signature      = signature,
            
        )

        return experiment
    
    
    # --- INITIALIZATION ---

    def __init__(
        self,    
        subject        : InSilicoSubject,
        iteration      : int,
        learning_rate  : float,
        signature      : Dict[str, float],
        data           : Dict[str, Any] = dict(),
        name           : str            = 'GD Invariance'
    ) -> None:

        #self._readout_score = defaultdict(list)

        super().__init__(name=name)
        # Save attributes from data
        self._iteration      = iteration
        self._learning_rate  = learning_rate
        self._subject        = subject
        self._signature       = signature
        
        self._targets        = list(self._subject.target.keys())
        self._ref_image      = cast(NDArray, data['ref_image'])
        self.im_side         = self._ref_image.shape[-1]
        self._ref_states     = cast(States, data['ref_states'])
        self._nat_thresh     = cast(float,   data['nat_thresh'])
        self.params          = cast(ParamConfig, data['params']) if 'params' in data else {}
        self.rnd_seed        = cast(int, data['rnd_seed'])
        self.ref_info        = cast(Dict[str, Any], data['ref_info'])
        self._curr_iter      = 0
        print(f"Signature: {self._signature}")

        msg = super()._init()
        
        # NOTE: We need to create a new specif message for Zdream
        self.msg = GDSnSMessage(
            start_time = msg.start_time,
            #rec_units  = self.subject.target,
        )
        print(self.msg.rec_units)
    # --- DATA FLOW ---
    def run(self) -> None:
        """
        Run the experiment.
        """
        aggregator = UPGrad()
        initial_image = torch.rand(1, 3, self.im_side , self.im_side , device=DEVICE, requires_grad=True)
        optimizer = optim.Adam([initial_image], lr=self._learning_rate)
        loss_fn = lambda x, y: torch.norm(x - y, p=2)  # Euclidean distance
        
        # -- OPTIMIZATION LOOP --
        norm_loss1 = 1
        norm_loss2 = 1
        for step in range(self._iteration):
            self._curr_iter += 1
            optimizer.zero_grad()

            # Apply normalization *inside* the loop before feeding to the model
            # This ensures gradients are computed w.r.t the unnormalized image we optimize
            img_normalized = initial_image.clamp(0, 1)
            
            # if step % 10 == 0 and step > 0:
            #     # Applica blur per smoothare
            with torch.no_grad():
                blur = transforms.GaussianBlur(kernel_size=25, sigma=(1.0, 2.0))
                initial_image.data = blur(initial_image.data)

            # Forward pass
            output = self._subject(img_normalized, with_grad = True)
            # Compute loss
            loss1 = self._signature[self._targets[0]] * loss_fn(output[self._targets[0]], self._ref_states[self._targets[0]])/norm_loss1
            loss2 = self._signature[self._targets[1]] *loss_fn(output[self._targets[1]], self._ref_states[self._targets[1]])/norm_loss2
                
            if step == 0:
                norm_loss1 = abs(loss1.detach())
                norm_loss2 = abs(loss2.detach())
                
            torchjd.backward([loss1, loss2], aggregator)
            print(f"Step {step}: Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            optimizer.step()
            
            
            for trgt,l,nl in zip(self._targets, [loss1, loss2], [norm_loss1, norm_loss2]):
                if trgt not in self.msg.loss_history:
                    self.msg.loss_history[trgt] = []
                self.msg.loss_history[trgt].append(l.detach()*nl)
                
            self.msg.image_history.append(img_normalized.detach().cpu().numpy())
            if output[self._targets[1]].detach().cpu().numpy() >= self._nat_thresh:
                print(f"Found a solution at step {step}")
                break
                
        # Dump
        state = GradientBasedExperimentState.from_msg(msg=self.msg)
        state.dump(
            out_dir=os.path.join(self.dir, 'state'),
            logger=self._logger)
        return self.msg
                
    def _run(self):
        """
        Run the experiment.
        """
        self.run()
        
        
class SnSGDMultiExperiment(MultiExperiment):
    
    def _init(self):
        
        super()._init()
        
        self._data['desc'] = 'SnS GD multi experiment'
        
        self._data['net_sbj']        = []
        self._data['robust']         = []
        self._data['lower_ly']       = []
        self._data['upper_ly']       = []
        self._data['low_target']     = []
        self._data['high_target']    = []
        self._data['task_signature'] = []
        self._data['rnd_seed']       = []
        self._data['reference_info'] = []
        self._data['reference_activ']= []
        self._data['solution']       = []
        self._data['layer_scores']   = []
        self._data['num_iter']       = []
        self._data['nat_thresh']     = []

    def _progress(
        self, 
        exp  : SnS_gradient_based, 
        conf : ParamConfig,
        i    : int,
        msg : GDSnSMessage,
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        # get the network type (robust or not)
        self._data['net_sbj'].append(exp.subject._name)
        self._data['robust'].append(exp.subject.robust)
        # Collect the needed data from the experiment
        #get the layers involved and their target neurons
        low_key, high_key = list(exp.subject.target.keys()) 
        self._data['lower_ly'].append(low_key)
        self._data['upper_ly'].append(high_key)
        self._data['low_target'].append(exp.subject.target[low_key])
        self._data['high_target'].append(exp.subject.target[high_key])
        #get the signature, which is characteristic of the task (invariance or adversarial attack)
        self._data['task_signature'].append(exp._signature)
            
        self._data['rnd_seed'].append(exp.rnd_seed)
        self._data['reference_info'].append(exp.ref_info)
        self._data['reference_activ'].append(exp._ref_states[high_key])
        self._data['solution'].append(exp.msg.image_history[-1])
        self._data['layer_scores'].append({k:np.stack([x.detach().cpu().numpy() for x in v]) for k,v in exp.msg.loss_history.items()})
        self._data['num_iter'].append(exp._curr_iter)
        self._data['nat_thresh'].append(float(exp._nat_thresh))

    def _finish(self):
        #TODO: summary of the experiment as .xlsx file?
        super()._finish() 
        #df = get_df_summary(self._data, savepath = self.target_dir)
        
    @property
    def _logger_type(self) -> Type[Logger]: return LoguruLogger