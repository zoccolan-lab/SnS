from collections import defaultdict
from os import path
import os
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy  as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas       import DataFrame
from PIL          import Image
from torchvision.transforms.functional import to_pil_image

from snslib.core.experiment                import ZdreamExperiment
from snslib.core.generator                 import Generator, DeePSiMGenerator
from snslib.core.optimizer                 import CMAESOptimizer, GeneticOptimizer, Optimizer
from snslib.core.scorer                    import ActivityScorer, Scorer
from snslib.core.subject                   import InSilicoSubject, TorchNetworkSubject
from snslib.core.utils.dataset             import ExperimentDataset, MiniImageNet, NaturalStimuliLoader, RandomImageDataset
from snslib.core.utils.logger              import DisplayScreen, Logger, LoguruLogger
from snslib.core.utils.message             import ZdreamMessage
from snslib.core.utils.misc                import concatenate_images, device
from snslib.core.utils.parameters          import ArgParams, ParamConfig
from snslib.core.utils.probe               import RecordingProbe
from snslib.core.utils.types               import Codes, ScoringUnits, Stimuli, Fitness, States
from  experiments.MaximizeActivity.plot import multiexp_lineplot, plot_optimizing_units, plot_scores, plot_scores_by_label, save_best_stimulus_per_variant, save_stimuli_samples
from snslib.experiment.utils.args            import ExperimentArgParams
from snslib.experiment.utils.parsing         import parse_boolean_string, parse_net_loading, parse_recording, parse_scoring
from snslib.experiment.utils.misc            import BaseZdreamMultiExperiment, make_dir

class MaximizeActivityExperiment(ZdreamExperiment):

    EXPERIMENT_TITLE = "MaximizeActivity"

    # Screen titles for the display
    NAT_IMG_SCREEN = 'Best Natural Image'
    GEN_IMG_SCREEN = 'Best Synthetic Image'
    
    # --- OVERRIDES ---
    
    # We override the properties to cast them to the specific components to activate their methods

    @property
    def scorer(self)  -> ActivityScorer:      return cast(ActivityScorer,      self._scorer ) 

    @property
    def subject(self) -> TorchNetworkSubject: return cast(TorchNetworkSubject, self._subject) 
    
    # --- CONFIG ---

    @classmethod
    def _from_config(cls, conf : ParamConfig) -> 'MaximizeActivityExperiment':
        '''
        Create a MaximizeActivityExperiment instance from a configuration dictionary.

        :param conf: Dictionary-like configuration file.
        :type conf: Dict[str, Any]
        :return: MaximizeActivityExperiment instance with hyperparameters set from configuration.
        :rtype: MaximizeActivity
        '''
        
        # Extract parameter from configuration and cast
        
        PARAM_weights    = str  (conf[ExperimentArgParams.GenWeights     .value])
        PARAM_variant    = str  (conf[ExperimentArgParams.GenVariant     .value])
        PARAM_template   = str  (conf[ExperimentArgParams.Template       .value])
        PARAM_dataset    = str  (conf[ExperimentArgParams.Dataset        .value])
        PARAM_shuffle    = bool (conf[ExperimentArgParams.Shuffle        .value])
        PARAM_batch      = int  (conf[ExperimentArgParams.BatchSize      .value])
        PARAM_net_name   = str  (conf[ExperimentArgParams.NetworkName    .value])
        PARAM_rec_layers = str  (conf[ExperimentArgParams.RecordingLayers.value])
        PARAM_scr_layers = str  (conf[ExperimentArgParams.ScoringLayers  .value])
        PARAM_unit_red   = str  (conf[ExperimentArgParams.UnitsReduction .value])
        PARAM_layer_red  = str  (conf[ExperimentArgParams.LayerReduction .value])
        PARAM_pop_size   = int  (conf[ExperimentArgParams.PopulationSize .value])
        PARAM_sigma0     = float(conf[ExperimentArgParams.Sigma0         .value])
        PARAM_exp_name   = str  (conf[          ArgParams.ExperimentName .value])
        PARAM_iter       = int  (conf[          ArgParams.NumIterations  .value])
        PARAM_rnd_seed   = int  (conf[          ArgParams.RandomSeed     .value])
        PARAM_render     = bool (conf[          ArgParams.Render         .value])
        PARAM_ref_code   = str  (conf[ExperimentArgParams.Reference .value])
        PARAM_customW_path = str  (conf[ExperimentArgParams.CustomWeightsPath     .value])
        PARAM_customW_var  = str  (conf[ExperimentArgParams.CustomWeightsVariant  .value])
        PARAM_optim_type  = str  (conf[ExperimentArgParams.OptimType       .value])
        PARAM_net_loading = str  (conf[ExperimentArgParams.WeightLoadFunction.value])


        PARAM_close_screen = conf.get(ArgParams.CloseScreen.value, True)
        # Set numpy random seed
        PARAM_ref_code = os.path.join(PARAM_ref_code, f'reference_code_{PARAM_net_name}{"robust" if PARAM_customW_var else ""}.npy')
        np.random.seed(PARAM_rnd_seed)
        if PARAM_customW_var == 'imagenet_l2_3_0.pt': 
            path2CustomW = os.path.join(PARAM_customW_path, PARAM_net_name, PARAM_customW_var)
        else:
            path2CustomW = PARAM_customW_var
        # --- NATURAL IMAGE LOADER ---

        # Parse template and derive if to use natural images
        template = parse_boolean_string(boolean_str=PARAM_template) 
        use_nat  = template.count(False) > 0
        
        # Create dataset and loader
        dataset  = MiniImageNet(root=PARAM_dataset) #RandomImageDataset(n_img=100, img_size=(224, 224,))# 
        
        nat_img_loader   = NaturalStimuliLoader(
            dataset=dataset,
            template=template,
            shuffle=PARAM_shuffle,
            batch_size=PARAM_batch
        )
        
        # --- GENERATOR ---

        generator = DeePSiMGenerator(
            root    = str(PARAM_weights),
            variant = str(PARAM_variant) # type: ignore
        ).to(device)
        
        # --- SUBJECT ---

        # Create a on-the-fly network subject to extract all network layer names
        layer_info: Dict[str, Tuple[int, ...]] = TorchNetworkSubject(
            network_name=str(PARAM_net_name)
        ).layer_info

        # Probe
        record_target = parse_recording(input_str=PARAM_rec_layers, net_info=layer_info)
        probe = RecordingProbe(target = record_target) # type: ignore
        #Get net loading function from parsing
        net_loading = parse_net_loading(input_str = PARAM_net_loading)
        # Subject with attached recording probe
        sbj_net = TorchNetworkSubject(
            record_probe=probe,
            network_name=PARAM_net_name,
            t_net_loading = net_loading,
            custom_weights_path = path2CustomW
        )
        
        # Set the network in evaluation mode
        sbj_net.eval()

        # --- SCORER ---

        # Parse target neurons
        scoring_units = parse_scoring(
            input_str=str(PARAM_scr_layers), 
            net_info=layer_info,
            rec_info=record_target
        )

        scorer = ActivityScorer(
            scoring_units=scoring_units,
            units_reduction=str(PARAM_unit_red),  # type: ignore
            layer_reduction=str(PARAM_layer_red), # type: ignore
        )

        # --- OPTIMIZER ---
        if PARAM_optim_type == 'cmaes':
            optim = CMAESOptimizer(
                codes_shape = generator.input_dim,
                rnd_seed    = PARAM_rnd_seed,
                pop_size    = PARAM_pop_size,
                sigma0      = PARAM_sigma0
                )
            
        elif PARAM_optim_type == 'genetic':
            optim = GeneticOptimizer(
                codes_shape  = generator.input_dim,
                rnd_seed     = PARAM_rnd_seed,
                pop_size     = PARAM_pop_size,
                rnd_scale    = 1,
                mut_size     = 0.3,
                mut_rate     = 0.3,
                allow_clones = True,
                n_parents    = 4
            )
        else:
            raise ValueError(f'Optimizer type {PARAM_optim_type} not recognized')
        #  --- LOGGER --- 

        conf[ArgParams.ExperimentTitle.value] = MaximizeActivityExperiment.EXPERIMENT_TITLE
        # logger = LoguruLogger.from_conf(conf=conf) NOT IN ISCHIAGUALASTIA BABY :)
        logger = LoguruLogger(to_file=False)
        
        # In the case render option is enabled we add display screens
        if bool(PARAM_render):

            # In the case of multi-experiment run, the shared screens
            # are set in `Args.DisplayScreens` entry
            if ArgParams.DisplayScreens.value in conf:
                for screen in cast(Iterable, conf[ArgParams.DisplayScreens.value]):
                    logger.add_screen(screen=screen)

            # If the key is not set it is the case of a single experiment
            # and we create screens instance
            else:

                # Add screen for synthetic images
                logger.add_screen(
                    screen=DisplayScreen(title=cls.GEN_IMG_SCREEN, display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE)
                )

                # Add screen fro natural images if used
                if use_nat:
                    logger.add_screen(
                        screen=DisplayScreen(title=cls.NAT_IMG_SCREEN, display_size=DisplayScreen.DEFAULT_DISPLAY_SIZE)
                    )

        # --- DATA ---
        
        data = {
            "dataset"      : dataset if use_nat else None,
            'render'       : PARAM_render,
            'close_screen' : PARAM_close_screen,
            'use_nat'      : use_nat
        }

        # --- EXPERIMENT INSTANCE ---
        
        experiment = cls(
            generator      = generator,
            scorer         = scorer,
            optimizer      = optim,
            subject        = sbj_net,
            logger         = logger,
            nat_img_loader = nat_img_loader,
            iteration      = PARAM_iter,
            data           = data, 
            name           = PARAM_exp_name,
        )

        return experiment
    
    # --- INITIALIZATION ---

    def __init__(
        self,    
        generator      : Generator,
        subject        : InSilicoSubject,
        scorer         : Scorer,
        optimizer      : Optimizer,
        iteration      : int,
        logger         : Logger,
        nat_img_loader : NaturalStimuliLoader,
        data           : Dict[str, Any] = dict(),
        name           : str            = 'maximize_activity',
    ) -> None:
        ''' 
        Uses the same signature as the parent class ZdreamExperiment.
        It save additional information from the `data` attribute to be used during the experiment.
        '''

        super().__init__(
            generator      = generator,
            subject        = subject,
            scorer         = scorer,
            optimizer      = optimizer,
            iteration      = iteration,
            logger         = logger,
            nat_img_loader = nat_img_loader,
            data           = data,
            name           = name,
        )

        # Save attributes from data
        self._render        = cast(bool, data[str(ArgParams.Render)])
        self._close_screen  = cast(bool, data['close_screen'])
        self._use_natural   = cast(bool, data['use_nat'])
        # Save dataset if used
        if self._use_natural: self._dataset = cast(ExperimentDataset, data['dataset'])
            
    # --- DATA FLOW ---
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        Override the method to save the last set of stimuli and the labels if natural images are used.
        '''

        # We save the last set of stimuli in `self._stimuli`
        self._stimuli, msg = data
        
        # We save the last set of labels if natural images are used
        if self._use_natural: self._labels.extend(msg.labels)
        
        states, msg = super()._stimuli_to_states(data)

        return states, msg

    def _scores_to_codes(self, data: Tuple[Fitness, ZdreamMessage]) -> Tuple[Codes, ZdreamMessage]:
        ''' We use scores to save stimuli (both natural and synthetic) that achieved the highest score. '''

        sub_score, msg = data

        # We inspect if the new set of stimuli for natural images,
        # checking if they achieved an higher score than previous ones.
        # In the case we both store the new highest value and the associated stimuli
        if self._use_natural:

            max_, argmax = tuple(f_func(sub_score[~msg.mask]) for f_func in [np.amax, np.argmax])

            if max_ > self._best_nat_scr:
                self._best_nat_scr = max_
                self._best_nat_img = self._stimuli[torch.tensor(~msg.mask)][argmax]

        return super()._scores_to_codes((sub_score, msg))
    
    # --- RUN METHODS ---

    def _progress_info(self, i: int, msg : ZdreamMessage) -> str:
        ''' Add information about the best score and the average score.'''
        
        # Best synthetic scores
        stat_gen = msg.stats_gen
        
        best_gen     = cast(NDArray, stat_gen['best_score']).mean()
        curr_gen     = cast(NDArray, stat_gen['curr_score']).mean()
        
        best_gen_str = f'{" " if best_gen < 1 else ""}{best_gen:.1f}' # Pad for decimals
        curr_gen_str = f'{curr_gen:.1f}'
        
        desc = f' | best score: {best_gen_str} | avg score: {curr_gen_str}'
        
        # Best natural score
        if self._use_natural:
            
            stat_nat     = msg.stats_nat
            best_nat     = cast(NDArray, stat_nat['best_score']).mean()
            best_nat_str = f'{best_nat:.1f}'
            
            desc = f'{desc} | {best_nat_str}'

        progress_super = super()._progress_info(i=i, msg=msg)

        return f'{progress_super} {desc}'

    def _init(self) -> ZdreamMessage:
        ''' We initialize the experiment and add the data structure to save the best score and best image. '''

        msg = super()._init()

        # Data structures to save best natural images and labels presented
        # Note that we don't keep track for the synthetic image as it is already done by the optimizer
        if self._use_natural:
            self._best_nat_scr = float('-inf') 
            self._best_nat_img = torch.zeros(self.generator.output_dim, device = device)
            self._labels: List[int] = []
        
        # Gif
        self._gif: List[Image.Image] = []

        return msg

    def _progress(self, i: int, msg : ZdreamMessage):

        super()._progress(i, msg)

        # Best synthetic stimulus (and natural one)
        best_synthetic     = self.generator(codes=msg.best_code)
        best_synthetic_img = to_pil_image(best_synthetic[0])
        
        if self._use_natural: best_natural = self._best_nat_img
        
        # Update gif if different from the previous frame
        if not self._gif or self._gif[-1] != best_synthetic_img:
            self._gif.append(
                best_synthetic_img
            )          
        
        # Update screens
        if self._render:

            self._logger.update_screen(
                screen_name=self.GEN_IMG_SCREEN,
                image=best_synthetic_img
            )

            if self._use_natural:
                self._logger.update_screen(
                    screen_name=self.NAT_IMG_SCREEN,
                    image=to_pil_image(best_natural)
                )   

    def _finish(self, msg : ZdreamMessage):
        ''' 
        Save best stimuli and make plots
        '''

        msg = super()._finish(msg)
        
        
    
        

        # Close screens if set (or preserve for further experiments using the same screen)
        if self._close_screen: self._logger.close_all_screens()
            
        # 1. SAVE VISUAL STIMULI (SYNTHETIC AND NATURAL)

        # Create image folder
        img_dir = make_dir(path=path.join(self.dir, 'images'), logger=self._logger)

        # We retrieve the best code from the optimizer
        # and we use the generator to retrieve the best image
        best_gen = self.generator(codes=msg.best_code)[0]  # remove batch size
        
        to_save: List[Tuple[Image.Image, str]] = [(to_pil_image(best_gen), 'best synthetic')]
        # If used we retrieve the best natural image
        if self._use_natural:
            best_nat = self._best_nat_img
            to_save.append((to_pil_image(best_nat), 'best natural'))
            to_save.append((concatenate_images(img_list=[best_gen, best_nat]), 'best stimuli'))
        
        # Saving images
        for img, label in to_save:
            
            out_fp = path.join(img_dir, f'{label.replace(" ", "_")}.png')
            self._logger.info(f'> Saving {label} image to {out_fp}')
            img.save(out_fp)
        
     

        self._logger.info(msg='')
        
        # 2. SAVE PLOTS

        # Create plot folder
        plots_dir = make_dir(path=path.join(self.dir, 'plots'), logger=self._logger)
        self._logger.formatting = lambda x: f'> {x}'
        
        # Plot scores
        plot_scores(
            scores=(
                np.stack(msg.scores_gen_history),
                np.stack(msg.scores_nat_history) if self._use_natural else np.array([])
            ),
            stats=(
                msg.stats_gen,
                msg.stats_nat if self._use_natural else dict(),
            ),
            out_dir=plots_dir,
            logger=self._logger
        )

        # Plot scores by label
        if self._use_natural:
            
            plot_scores_by_label(
                scores=(
                    np.stack(msg.scores_gen_history),
                    np.stack(msg.scores_nat_history)
                ),
                lbls=self._labels,
                out_dir=plots_dir, 
                dataset=self._dataset,
                logger=self._logger
            )
        
        self._logger.reset_formatting()
        self._logger.info(msg='')
        
        return msg

class MaximizeActivityExperiment2(MaximizeActivityExperiment):
    '''
    This class extends the MaximizeActivityExperiment to save activation states
    '''
    
    def __init__(
        self,
        generator      : Generator, 
        subject        : InSilicoSubject, 
        scorer         : Scorer, 
        optimizer      : Optimizer, 
        iteration      : int, 
        logger         : Logger, 
        nat_img_loader : NaturalStimuliLoader, 
        data           : Dict[str, Any] = dict(), 
        name           : str = 'maximize_activity'
    ) -> None:
        
        super().__init__(
            generator=generator, 
            subject=subject, 
            scorer=scorer, 
            optimizer=optimizer, 
            iteration=iteration, 
            logger=logger, 
            nat_img_loader=nat_img_loader,
            data=data,
            name=name
        )
    
    def _stimuli_to_states(self, data: Tuple[Stimuli, ZdreamMessage]) -> Tuple[States, ZdreamMessage]:
        '''
        Override the method to save the activation states of the network.
        '''
        
        states, msg = super()._stimuli_to_states(data=data)

        # NOTE: This is used in a separated class as it can be memory demanding 
        msg.states_history.append(states)
        
        return states, msg


# --- MULTI EXPERIMENT ---

class NeuronScalingMultiExperiment(BaseZdreamMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'   ] = 'Scores at varying number of scoring neurons' # TODO
        
        # Scores with order - layer -> neuron -> scores
        self._data['score'] = defaultdict(lambda: defaultdict(list))

    def _progress(
        self, 
        exp  : MaximizeActivityExperiment, 
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        tot_units = sum([rec_units[0].shape[0] for rec_units in msg.rec_units.values()]) # type: ignore
        scr_layers = msg.scr_layers
        
        if len(scr_layers) > 1: raise ValueError(f'Expected to score from a single layer, but {len(scr_layers)} were found')
        scr_layer = scr_layers[0]
        
        self._data['score'][scr_layer][tot_units].append(msg.stats_gen['best_score'])

    def _finish(self):
        
        # Reorganize results from defaultdict to dict to store as .pickle
        self._data['score'] = {
            layer: {
                neuron: [scr for scr in score] 
                for neuron, score in scores.items()
            }
            for layer, scores in self._data['score'].items()
        }
        
        super()._finish()
        
        plot_dir = make_dir(path=path.join(self.target_dir, 'plots'), logger=self._logger)

        self._logger.info("Saving plots...")
        self._logger.formatting = lambda x: f'> {x}'
        
        plot_optimizing_units(
            data    = self._data['score'],
            out_dir = plot_dir,
            logger  = self._logger
        )
        
        self._logger.reset_formatting()
        self._logger.info("")

class GeneratorVariantMultiExperiment(BaseZdreamMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'   ] = 'Best stimuli for each variant version of a generator' # TODO
        
        # Stores the best code and score for each variant
        self._data['best_variant'] = defaultdict(lambda: defaultdict(tuple))

    def _progress(
        self, 
        exp  : MaximizeActivityExperiment, 
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        code      = msg.best_code
        score     = msg.stats_gen['best_score']
        scr_layer = str(conf[ExperimentArgParams.RecordingLayers.value])
        variant   = str(conf[ExperimentArgParams.GenVariant     .value])
        
        # Check if new scoring layer
        if scr_layer in self._data['best_variant']:
            
            # Check if new variant
            if variant in self._data['best_variant'][scr_layer]:
                
                # Extract best score
                _, best_score = self._data['best_variant'][scr_layer][variant]
                
                # Update if new best score
                if score > best_score:
                    self._data['best_variant'][scr_layer][variant] = (code, score)
            
            # If new variant
            else:
                self._data['best_variant'][scr_layer][variant] = (code, score)
        
        # If new scoring layer (and new variant)
        else:
            self._data['best_variant'][scr_layer][variant] = (code, score)


    def _finish(self):
        
        self._data['best_variant'] = {
            neuron: {variant: code_score for variant, code_score in scores.items()} 
            for neuron, scores in self._data['best_variant'].items()
        }
        
        super()._finish()
        
        plot_dir = make_dir(path=path.join(self.target_dir, 'plots'), logger=self._logger)

        self._logger.info("Saving plots...")
        self._logger.formatting = lambda x: f'> {x}'
        
        save_best_stimulus_per_variant(
            neurons_variant_codes=self._data['best_variant'],
            gen_weights=self._search_config[0][ExperimentArgParams.GenWeights.value], # type: ignore - literal
            out_dir=plot_dir,
            logger=self._logger
        )
        
        self._logger.reset_formatting()
        self._logger.info("")

class SamplesMaximizationMultiExperiment(BaseZdreamMultiExperiment):

    def _init(self):
        
        super()._init()
        
        self._data['desc'   ] = 'Scores for multiple optimization for the same neuron'
        
        self._data['samples'] = defaultdict(list)

    def _progress(
        self, 
        exp  : MaximizeActivityExperiment, 
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        rec_layer = conf[ExperimentArgParams.RecordingLayers.value]
        score     = msg.stats_gen['best_score']
        code      = msg.best_code
        
        self._data['samples'][rec_layer].append((score, code))
        
    def _finish(self):
        
        self._data['samples'] = {k: v for k, v in self._data['samples'].items()}
        
        super()._finish()
        
        plot_dir = make_dir(path=path.join(self.target_dir, 'plots'), logger=self._logger)

        self._logger.info("Saving stimuli samples...")
        self._logger.formatting = lambda x: f'> {x}'
        
        conf = self._search_config[0]
        
        generator = DeePSiMGenerator(
            root    = str(conf[ExperimentArgParams.GenWeights.value]),
            variant = str(conf[ExperimentArgParams.GenVariant.value])  # type: ignore - literal
        ).to(device)
        
        save_stimuli_samples(
            stimuli_scores = self._data['samples'],
            generator      = generator,
            out_dir        = plot_dir,
            logger         = self._logger
        )
        
        self._logger.reset_formatting()
        self._logger.info("")


class LayersCorrelationMultiExperiment(BaseZdreamMultiExperiment):

    def _init(self):

        super()._init()
        
        # Description
        self._data['desc'] = 'Layers correlations ...' 
        # Delta activation
        self._data['deltaA_rec'] = list()

        # Hyperparameter and recordings
        self._data['layer']     = list()
        self._data['score']     = list()
        self._data['score_nat'] = list()
        self._data['iter']      = list()
        self._data['neurons']   = list()

        # Correlations
        self._data['corr_opt_rec_mean'] = list()
        self._data['corr_opt_rec_var']  = list()
        self._data['corr_rec_rec_mean'] = list()
        self._data['corr_rec_rec_var']  = list()

    
    def _progress(
        self, 
        exp  : MaximizeActivityExperiment2, 
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)

        # Update score and parameters
        self._data['score']    .append(msg.stats_gen['best_score'])
        self._data['score_nat'].append(msg.stats_nat['best_score'] if exp._use_natural else np.nan)
        self._data['neurons']  .append(0) # TODO
        self._data['layer']    .append(msg.scr_layers)
        self._data['iter']     .append(exp.iteration)
        
        # Create new dictionary for fields
        deltaA_rec        = {}
        corr_opt_rec_mean = {}
        corr_opt_rec_var  = {}
        corr_rec_rec_mean = {}
        corr_rec_rec_var  = {}

        # Iterate over recorded layers to get statistics about non optimized sites
        non_scoring = self._get_non_scoring_units(exp=exp)
        
        states_history = {
            key: np.stack([state[key] for state in msg.states_history]) if msg.states_history else np.array([])
            for key in msg.rec_layers
        }
        
        for layer, activations in states_history.items():

            # Extract non score units for that layer
            non_scoring_units = non_scoring[layer]

            # If there are units recorder but not scored
            if non_scoring_units:

                # Get the average difference in activation between the last and the first optim iteration
                deltaA_rec[layer] = np.mean(activations[-1, :, non_scoring_units]) - \
                                    np.mean(activations[ 0, :, non_scoring_units])   

                # Compute the correlation between each avg recorded unit and the mean of the optimized sites. 
                # Store correlation mean and variance
                avg_rec  = np.mean(activations[:, :, non_scoring_units], axis=1).T
                corr_vec = np.corrcoef(msg.stats_gen['mean_gens'], avg_rec)[0, 1:]
                
                corr_opt_rec_mean[layer] = np.mean(corr_vec)
                corr_opt_rec_var [layer] = np.std(corr_vec)

                # Compute the correlation between each avg recorded units. 
                # Store correlation mean and variance
                corr_rec_rec_mat = np.corrcoef(avg_rec)
                upper_triangle   = np.triu_indices(corr_rec_rec_mat.shape[0], k=1)
                
                corr_rec_rec_mean[layer] = np.mean(corr_rec_rec_mat[upper_triangle])
                corr_rec_rec_var[layer]  = np.std (corr_rec_rec_mat[upper_triangle])

        # Update correlations
        self._data['deltaA_rec']       .append(deltaA_rec)
        self._data['corr_opt_rec_mean'].append(corr_opt_rec_mean)
        self._data['corr_opt_rec_var'] .append(corr_opt_rec_var)
        self._data['corr_rec_rec_mean'].append(corr_rec_rec_mean)
        self._data['corr_rec_rec_var'] .append(corr_rec_rec_var)

    def _finish(self):

        super()._finish()

        # Organize results as a dataframe
        df = self._create_df(multiexp_data=self._data)

        # Create output image folder
        plots_dir = make_dir(path=path.join(self.target_dir, 'plots'), logger=self._logger)

        self._logger.formatting = lambda x: f'> {x}'

        # Plot neuron score scaling
        sc_metrics = ['scores', 'scores_norm']
        
        for m in sc_metrics:
            
            multiexp_lineplot(
                out_df=df, 
                gr_vars=['layers', 'neurons'],
                out_dir=plots_dir,
                y_var=m,
                metrics=['mean', 'sem'],
                logger=self._logger
            )
        
        # Plot different information varying the response variable
        for c in df.columns:

            if 'rec' in c:

                multiexp_lineplot(
                    out_df=df, 
                    gr_vars= ['layers', 'neurons'],
                    out_dir=plots_dir,
                    y_var = c, 
                    metrics = ['mean', 'sem'],
                    logger=self._logger
                )
        
        self._logger.reset_formatting
        
    @staticmethod
    def _create_df(multiexp_data: Dict[str, Any]) -> DataFrame:
        '''
        Reorganize multi-experiment result in a Dataframe for plotting.

        NOTE: This static method allows for an offline computation provided
            multi-experiment results stored as .PICKLE
        '''
        
        def extract_layer(el):
            ''' Auxiliar function to extract layer names from a list element'''
            if len(el) > 1:
                raise ValueError(f'Expected to record from a single layer, multiple where found: {el}')
            return el[0]
        
        # Organize main experiment data
        data = {
            'layers'  : np.stack([extract_layer(el) for el in multiexp_data['layer']]),
            'neurons' : np.stack(multiexp_data['neurons'], dtype=np.int32),
            'iter'    : np.stack(multiexp_data['iter'], dtype=np.int32),
            'scores'  : np.concatenate(multiexp_data['score']),
            'scores_nat': np.concatenate(multiexp_data['score_nat'])     
        }

        data['scores_norm'] = data['scores']/data['scores_nat']
        
        # Extract recording-related data 

        # Unique_keys referred to a 
        unique_layers = list(set().union(*[d.keys() for d in multiexp_data['deltaA_rec']]))

        # Iterate over the keys involved in recordings
        rec_keys = ['deltaA_rec', 'corr_opt_rec_mean', 'corr_opt_rec_var', 'corr_rec_rec_mean', 'corr_rec_rec_var']

        for key in rec_keys:

            # Key prefix depends on th
            prefix = 'Δrec_' if key == 'deltaA_rec' else key

            # Initialize dictionary with Nans
            rec_dict = {
                f'{prefix}{layer}': np.full(len(multiexp_data[key]), np.nan, dtype=np.float32) 
                for layer in unique_layers
            }
            
            # Add result to specific experiment if layer name match
            for i, result in enumerate(multiexp_data[key]):
                for layer in unique_layers:
                    if layer in result.keys():
                        rec_dict[f'{prefix}{layer}'][i] = result[layer]
                        
            # Unify all data into a single dictionary
            data.update(rec_dict)
        
        # Transform data in dictionary form
        return pd.DataFrame(data)

    @staticmethod
    def _get_non_scoring_units(exp: MaximizeActivityExperiment) -> Dict[str, ScoringUnits]:
        '''
        Given an experiment returns a mapping between layer name and activations indexes
        for recorded but non-scored units
        '''

        non_scoring_units = {}

        for layer, rec_units in exp.subject.target.items():
        
            # If not all recorded extract their length
            if rec_units:
                recorded = rec_units[0].size  # length of the first array of the tuple
            
            # If all recorded is the total number of neurons
            else:
                recorded = np.prod(exp.subject.layer_info[layer])
            
            # Compute the non recorded units
            try:
                scoring = exp.scorer.scoring_units[layer]
            except KeyError:
                scoring = []
            
            not_scoring = set(range(recorded)).difference(scoring)

            non_scoring_units[layer] = list(not_scoring)

        return non_scoring_units
    

class NeuronReferenceMultiExperiment(BaseZdreamMultiExperiment):
    
    def _init(self):
        
        super()._init()
        
        self._data['desc'     ] = 'Creates a code reference for layer, neuron and random_seed'
        self._data['reference'] = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    def _progress(
        self, 
        exp  : MaximizeActivityExperiment, 
        conf : ParamConfig,
        msg  : ZdreamMessage, 
        i    : int
    ):

        super()._progress(exp=exp, conf=conf, i=i, msg=msg)
        
        gen_variant = str(conf[ExperimentArgParams.GenVariant.value])
        rec_units   = exp.subject.target
        r = '_r' if exp.subject.robust else ''
        neural_net  = exp.subject.name + r
        
        assert len(rec_units) == 1, f'Expected to record from a single layer, but {len(rec_units)} were found'
        
        layer = list(rec_units.keys())[0]
        units = rec_units[layer]
        #assert len(units) == 1, f'Expected to record a single neuron, but {len(units)} were found'
        
        neuron = str(list(map(int, units))) # NOTE: tuple indexing
        
        seed = int(conf[ArgParams.RandomSeed.value])
        
        score = msg.stats_gen['best_score']
        code  = msg.best_code
        
        self._data['reference'][neural_net][gen_variant][layer][neuron][seed] = {'code': code, 'fitness': score}
        
    def _finish(self):
        
        self._data['reference'] = {
            neural_net: {
                variant: {
                    layer: {
                        neuron: {
                            seed: data
                            for seed, data in seeds.items()
                        }
                        for neuron, seeds in neurons.items()
                    }
                    for layer, neurons in layers.items()
                }
                for variant, layers in variants.items()
            }
            for neural_net, variants in self._data['reference'].items()
        }
        
        super()._finish()


