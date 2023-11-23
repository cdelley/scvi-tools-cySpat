from typing import (
    Callable, 
    Iterable, 
    List, 
    Literal, 
    Optional, 
    Union, 
    Sequence, 
    Dict, 
    Any, 
    Tuple
)

import numpy as np
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)

from ._module import DiffusionModuleVB, NoiseModel

class DiffusionVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Implements a Variational Inference model for spatial single-cell experiments
    using diffusion-based spatial encoding.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    spot_encoding : Iterable
        Encoding of the spots in the spatial array.
    spot_xpos : Iterable
        X positions of the spots.
    spot_ypos : Iterable
        Y positions of the spots.
    layer_dims : List[int], default=[96, 10]
        Dimensions of the layers in the neural network.
    grid_oversampling : float, default=1.0
        Oversampling rate for the grid used in spatial encoding.
    method : Optional[Literal["VB", "MLE"]], default="MLE"
        Method for inference, either 'VB' (Variational Bayes) or 'MLE' (Maximum Likelihood Estimation).
    module_kwargs : dict
        Additional keyword arguments for the diffusion module.
    """
    
    #_module_cls = DiffusionVAE
    #_data_splitter_cls = DataSplitter
    #_training_plan_cls = AdversarialTrainingPlan
    #_train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        spot_encoding: Iterable,
        spot_xpos: Iterable,
        spot_ypos: Iterable,
        layer_dims: List[int] = [96,10],
        grid_oversampling: float = 1.0,
        method: Optional[Literal["VB", "MLE"]] = "MLE",
         **module_kwargs,
    ):
        super().__init__(adata)
        
        # params
        self.method = method
        self.spot_encoding = spot_encoding
        self.spot_xpos = spot_xpos
        self.spot_ypos = spot_ypos
        
        noise_dist = NoiseModel.get_average(adata.X)
        
        self.module = DiffusionModuleVB(
            n_input=self.summary_stats.n_vars,
            layer_dims = layer_dims,
            encodings = spot_encoding,
            encoding_x = spot_xpos,
            encoding_y = spot_ypos,
            grid_approximation = True,
            grid_oversampling = grid_oversampling,
            normalized_noise_dist = noise_dist,
            method = method,
            **module_kwargs,
        )
        self._model_summary_string = (
            "Diffusion Model with the following params: \nn_latent: {}"
        ).format(
            np.sum(layer_dims),
        )
        self.init_params_ = self._get_init_params(locals())
        
    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        Prepares an AnnData object for use with the DiffusionVI model.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix to be set up.
        batch_key : Optional[str]
            Key in adata.obs to be used as batch information.
        layer : Optional[str]
            Specifies which layer of the AnnData object to use.
        **kwargs
            Additional keyword arguments passed to the setup method.
        
        Returns
        -------
        Optional[AnnData]
            The prepared AnnData object, if applicable.
        """
        
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
        set_diff_const: Optional[float] = None,
        set_noise_epsilon: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the latent representation of cells.

        Parameters
        ----------
        adata : Optional[AnnData]
            Annotated data matrix. If None, uses the AnnData object used to initialize the model.
        indices : Optional[Sequence[int]]
            Indices of the cells to be used. If None, all cells are used.
        give_mean : bool, default=True
            If True, returns the mean of the distribution, otherwise samples from the distribution.
        mc_samples : int, default=5000
            Number of Monte Carlo samples for distributions without a closed-form mean.
        batch_size : Optional[int]
            Batch size for data loading. If None, uses the default scvi setting.
        return_dist : bool, default=False
            If True, returns the distribution's mean and variance. Ignores `give_mean` and `mc_samples`.
        set_diff_const : Optional[float]
            Sets a specific diffusion constant.
        set_noise_epsilon : Optional[float]
            Sets a specific noise epsilon.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            The latent representation of each cell, or a tuple of its mean and variance.
        """
        
        self._check_if_trained(warn=True)

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        latents = {
            'diffusion_constant': [],
            'noise_epsilon' : [],
            'xy_map' : [],
            'xy_err' : [],
            'xy_mle' : [],
            'logpostprob' : [],
        }
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            
            generative_input = self.module._get_generative_input(tensors, inference_outputs)
            scale = generative_input['scale']
            noise_dist = generative_input['lambda_noise']
            
            xy = inference_outputs['z']
            if self.method == "VB":
                diff_const = inference_outputs["diff_dist"].sample([mc_samples])
                noise_epsilon = inference_outputs["noise_epsilon_dist"].sample([mc_samples])
                
                mean_diff_const = diff_const.mean(dim=0)
                mean_noise_epsilon = noise_epsilon.mean(dim=0)
                    
            elif self.method == "MLE":
                diff_const = inference_outputs["diff_const"]
                mean_diff_const = diff_const
                noise_epsilon = inference_outputs["noise_epsilon"]
                mean_noise_epsilon = noise_epsilon
            
            if set_diff_const is not None:
                mean_diff_const = mean_diff_const * 0 + set_diff_const
            if set_noise_epsilon is not None:
                mean_noise_epsilon = mean_noise_epsilon * 0 + set_noise_epsilon
                
            # currently we always use the mean of the latent parameters and 
            # not fully propagate their uncertainity into cell position inference
            observed_counts = tensors[REGISTRY_KEYS.X_KEY]
            generative_outputs = self.module.generative(
                xy, 
                mean_diff_const,
                mean_noise_epsilon,
                scale,
                noise_dist,
                generative_input['precomputed_distances'],
            )
            expected_counts = generative_outputs["expected_counts"]
            expected_counts_dist = generative_outputs["expected_counts_dist"]

            # calculate log likelyhood of the tag counts to get the positions
            observed_counts = observed_counts.to(expected_counts.device)
            xy =  xy.to(expected_counts.device)
            llik, logpostprob = self.module.post_prob(observed_counts, expected_counts_dist)
            x_map, y_map, xy_err = self.module.position_and_variance(logpostprob, xy)
            x_mle, y_mle = self.module.MLE_position_estimate(logpostprob, xy)
            
            if give_mean:
                diff_const = mean_diff_const
                noise_epsilon = mean_noise_epsilon
            
            latents['diffusion_constant'].append(diff_const.cpu().numpy())
            latents['noise_epsilon'].append(noise_epsilon.cpu().numpy())
            latents['xy_map'].append(torch.stack([x_map, y_map]).cpu().numpy())
            latents['xy_err'].append(xy_err.cpu().numpy())
            latents['xy_mle'].append(torch.stack([x_mle, y_mle]).cpu().numpy())
            #latents['logpostprob'].append(logpostprob.cpu().numpy())
            
        latents['diffusion_constant'] = np.array(latents['diffusion_constant']).mean(axis=-1)
        latents['noise_epsilon'] = np.concatenate(latents['noise_epsilon'], axis=-1)
        latents['xy_map'] = np.concatenate(latents['xy_map'], axis=-1).T
        latents['xy_err'] = np.concatenate(latents['xy_err'], axis=-1)
        latents['xy_mle'] = np.concatenate(latents['xy_mle'], axis=-1).T
        #latents['logpostprob'] = np.concatenate(latents['logpostprob'], axis=0)
        return latents
    

    def get_simple_position_estimate(
        self, 
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Provides a simple position estimate by mapping the count vector against the tag encoding scheme.
        Each cell is assigned to the position of the array spot which is encoded by the two most abundant 
        tags on the cell. If the pair of two highest tags isn't encoded np.nan is assigned.

        Parameters
        ----------
        adata : Optional[AnnData]
            Annotated data matrix. If None, uses the AnnData object used to initialize the model.
        indices : Optional[Sequence[int]]
            Indices of the cells to be used. If None, all cells are used.
        batch_size : Optional[int]
            Batch size for data loading. If None, uses the default scvi setting.

        Returns
        -------
        np.ndarray
            2D Array of estimated positions. 
        """
        
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        len_spot_encoding = len(self.spot_encoding[0])
        spots = [set(t) for t in self.spot_encoding]
        
        pos_x = []
        pos_y = []
        assigned, failed = 0, 0
        for tensors in dataloader:
            observed_counts = tensors[REGISTRY_KEYS.X_KEY].numpy()
            for vec in observed_counts:
                srt_idx = np.argsort(vec)
                try:
                    pos = spots.index(set(srt_idx[-len_spot_encoding:]))
                    pos_x.append(self.spot_xpos[pos])
                    pos_y.append(self.spot_ypos[pos])
                    assigned += 1
                except ValueError:
                    pos_x.append(np.nan)
                    pos_y.append(np.nan)
                    failed += 1
        print(f'assigned {assigned} cells to position, {failed} cells failed')
        return np.array([pos_x, pos_y]).T
    
    @staticmethod
    def optimizer_creator(module_parts: Dict[str, Any]) -> Dict[str, torch.optim.Optimizer]:
        """
        this appears not working currently

        The goal here is to implement different training rates for the NN in the amortized inference
        of the local params and the shared global params.
        
        
        Creates optimizers with different training rates for various parts of the neural network.

        Parameters
        ----------
        module_parts : Dict[str, Any]
            Parts of the module for which to create optimizers.

        Returns
        -------
        Dict[str, torch.optim.Optimizer]
            Dictionary of optimizers for each module part.
        """
        learning_rates = {
            'noise_encoder': 0.00001,
            'diff_const': 0.1
        }

        optimizers = {}
        for part_name, part_params in module_parts.items():
            if part_name == 'noise_encoder':
                optimizer = torch.optim.Adam(part_params.parameters(), lr=learning_rates['noise_encoder'])
            elif part_name == 'diff_const':
                optimizer = torch.optim.Adam([part_params], lr=learning_rates['diff_const'])
            else:
                optimizer = torch.optim.Adam(part_params, lr=0.001)  # Default optimizer
            optimizers[part_name] = optimizer

        return optimizers
