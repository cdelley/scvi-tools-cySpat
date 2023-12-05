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

import torch
from torch import nn
from torch.distributions import Normal, Gamma, Beta, LogNormal, kl_divergence
import numpy as np


from scvi.module.base import BaseModuleClass, auto_move_data, LossOutput
from scvi import REGISTRY_KEYS

# Import FCLayers from the appropriate location
from scvi.nn import FCLayers
from ._base_components import flexFCLayers

import matplotlib.pyplot as plt

# If using collections or other specific modules, add them here
# import collections


#from scvi.module.base import (
#    BaseModuleClass,
#    BaseMinifiedModeModuleClass,
#    LossOutput,
#    PyroBaseModuleClass,
 #   auto_move_data,
#)



# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        hidden_dims: List[int] = [128],
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.return_dist = return_dist
        
        # z encoder takes as imput gene expression and covariates
        self.z_encoder = FCLayers(
            n_in=n_input,
            n_out=hidden_dims[-1],
            n_cat_list=n_cat_list,
            hidden_dims=hidden_dims[:-1],
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(hidden_dims[-1], n_output)
        self.var_encoder = nn.Linear(hidden_dims[-1], n_output)
        
        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = Encoder._identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent
    
    @staticmethod
    def _identity(x):
        return x


class DiffusionDecoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses the provided diffusion process as the decoder model.
    """

    def __init__(
        self,
        encodings: List,
        encoding_x: torch.Tensor,
        encoding_y: torch.Tensor,
        rel_tag_concentration: Union[float, torch.Tensor] = 1.,
        rel_spot_concentration: Union[float, torch.Tensor] = 1.,
        **kwargs,
    ):
        super().__init__()
#        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.encoding_x = encoding_x.clone().detach()
        self.encoding_y = encoding_y.clone().detach()
        
        # map encoding labels to array position
        self.encoding_map = dict.fromkeys(np.sort(np.ravel(encodings)))
        for i, k in enumerate(self.encoding_map.keys()):
            self.encoding_map[k] = [i for (i,t) in enumerate(encodings) if k in t]
        self.n_labels = len(self.encoding_map)
        self.n_spots = len(encodings)
        
        # goal is to have the possibility to encode unequal concentrations  
        # across tags and spots this is not yet implemented however
        if type(rel_spot_concentration) == int or type(rel_spot_concentration) == float:
            self.rel_spot_concentration = torch.ones(self.n_spots, dtype=torch.float32) * rel_spot_concentration
        else:
            self.rel_spot_concentration = rel_spot_concentration
            
    
    def forward(
        self,
        z: Union[None, torch.Tensor],
        diffusion_constant: float,
        precomputed_distances: Union[None, torch.Tensor] = None, 
        *cat_list: int,
    ):
        """
        Forward pass for the diffusion-based decoder.

        Parameters
        ----------
        z: Tensor
            Latent space representation with shape (batch_size, n_input)
        x_img: Tensor
            x-coordinates of the measurements with shape (batch_size,)
        y_img: Tensor
            y-coordinates of the measurements with shape (batch_size,)
        encodings: array-like
            Integer labels of the emitter spots
        
        Returns
        -------
        Tensor
            Generated count vector based on the diffusion process.
        """

        # Get the encoder coordinates from the latent space
        # Assuming the latent space has 2 dimensions for x and y coordinates
        # x_estimate = z[:, 0], y_estimate = z[:, 1] 
        
        # Calculate the emission based on the diffusion process
        rate_per_spot = self.diffusion_kernel(z, diffusion_constant, precomputed_distances)
        rate_per_tag = self.label_transfer_rate(rate_per_spot)
        
        return rate_per_tag
    
    def diffusion_kernel(
        self, 
        z, 
        diffusion_constant, 
        precomputed_distances = None, 
        underflow_nu = 1e-12
    ):
        """2D Diffusion kernel for the emission process. (Fick's law)"""

        # Calculate distances between emitter spots and measurement positions
        # shape is (batch_size, n_spots)
        if precomputed_distances == None:
            distances = self.rel_distances(z)
        else:
            distances = precomputed_distances.to(diffusion_constant.device)
        
        # calculate the normalization factor
        norm = 1 / (2 * torch.pi * diffusion_constant)
        
        # Apply diffusion kernel
        kernel = norm * torch.exp(-distances / (2 * diffusion_constant))
        
        # scale the diffusion kernel by concentration
        rate_per_spot = kernel# * self.rel_spot_concentration[None, :] 
        
        # prevent underflow
        rate_per_spot = rate_per_spot + underflow_nu
        
        return rate_per_spot
    
    def rel_distances(self, z):
        """calculates distances between array spots and test points"""
        distances = (
            (self.encoding_x[None, :] - z[:, 0][:, None]) ** 2 
            + (self.encoding_y[None, :] - z[:, 1][:, None]) ** 2
        )
        return distances
    
    def label_transfer_rate(self, scaled_kernel):
        """For each tag, sum up the tag diffusion rate contribution from different 
        array spots. This reduces a n_cell x n_spots matrix to a n_cell x n_tags
        matrix."""
        
        # reduce over labels
        # shape is (batch_size, n_labels)
        exposure = torch.zeros(
            (scaled_kernel.shape[0], self.n_labels), 
            dtype=torch.float32, 
            device=self.encoding_x.device
        )
        for label, pos in self.encoding_map.items():
            exposure[:, label] = scaled_kernel[:, pos].sum(axis=1)
        return exposure
             
class NoiseModel(nn.Module):
    """Very simple noise model for now that is simply a global background vector"""
    
    @staticmethod
    def get_average(tag_counts: List[float]):
        """Uses input matrix to determin the average profile over all
        droplets and normalizes the counts to one
        
        Inputs:
        -------
            tag_counts: n_drops x m_tags count matrix
            
        Returns:
        --------
            normalized 1 x m_tags vector
        """
        
        mean_count =  np.mean(tag_counts, axis=0) 
        normalized_mean_count = mean_count / tag_counts.shape[1]
        return normalized_mean_count

class DiffusionModuleVB(BaseModuleClass):
    """
    

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
        self,
        n_input: int,
        encodings: Iterable[int],
        encoding_x: Iterable[int],
        encoding_y: Iterable[int],
        n_cat_list: Iterable[int] = None,
        layer_dims: List[int] = [64, 12],
        grid_approximation: bool = True,
        grid_oversampling: int = 1,
        normalized_noise_dist: Iterable[float] = None,
        method: Optional[Literal["VB", "MLE"]] = "MLE",
        dist_eps: float = 1e-8,
        loss_agregation: Optional[Callable] = None,
        prior_diff_const: Optional[torch.distributions.Distribution] = Gamma(10, 20),
        prior_noise_const: Optional[torch.distributions.Distribution] = Beta(1.0, 1.0),
    ):
        super().__init__()
        
        # params
#        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.grid_approximation = grid_approximation
        self.method = method
        self.dist_eps = dist_eps
        self.encoding_x = torch.Tensor(encoding_x)
        self.encoding_y = torch.Tensor(encoding_y)
        
        # priors
        self.prior_diffusion = prior_diff_const
        self.prior_noise = prior_noise_const
        
        # encoder for all auxilary latent variables
        self.auxilary_encoder = flexFCLayers(
            n_in=n_input,
            n_out=layer_dims[-1], 
            hidden_dims=layer_dims[:-1],
            n_cat_list=n_cat_list, 
            dropout_rate=0.1,
            use_batch_norm=True,
            use_layer_norm=False,
            use_activation=True,
            bias=True,
            inject_covariates=True,
            activation_fn= nn.ReLU,            
        )
        
        # callables
        self.exp_transform = torch.exp
        self.sigmoid_transform = torch.nn.Sigmoid()
        self.loss_agregation = torch.mean if loss_agregation is None else loss_agregation
        
        if method == "VB":            
            self.diff_params = nn.Parameter(torch.rand(2), requires_grad=True)
            
            # encoder for amortized variational Bayes of local latent variables
            self.noise_encoder = nn.Linear(layer_dims[-1], 2)
            
        elif method == "MLE":
            # global diffusion constant
            self.diff_const = nn.Parameter(torch.rand(1), requires_grad=True)
                        
            # encoder for local latent variables
            self.noise_encoder = nn.Linear(layer_dims[-1], 1)
            
        else:
            raise ValueError("""Use a supported method for inference. Choices are: 'VB', 'MLE'
                             \n but was set to: {}""".format(method))
        
        # generative model
        self.diffusion_decoder = DiffusionDecoder(
            encodings = encodings,
            encoding_x = self.encoding_x,
            encoding_y = self.encoding_y,
        )
        
        if grid_approximation:
            self.xy_grid = self.make_grid(grid_oversampling, self.encoding_x, self.encoding_y)
            self.precomputed_distances = self.diffusion_decoder.rel_distances(self.xy_grid)
        
        # noise model
        if normalized_noise_dist is None:
            self.normalized_noise_dist = torch.ones(n_input) / n_input
        else:
            _sum_noise = torch.tensor(normalized_noise_dist).sum()
            self.normalized_noise_dist = torch.tensor(normalized_noise_dist) / _sum_noise
        
    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # fetch the raw counts, and add them to the dictionary
        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = dict(x=x)
        return input_dict
        
    @auto_move_data
    def inference(self, x):
        """ amortized inference of the latent parameters using a neuronal net"""
        
        # log for stability
        x_ = torch.log(1 + x)
        
        if self.grid_approximation:
            # position is not inferred but all possible positions sampled with a grid
            z = self.xy_grid
        else:
            # no other mode implemented
            pass        
        
        # get params from encoders
        aux_latents = self.auxilary_encoder(x)
        noise_params = self.noise_encoder(aux_latents)
        
        if self.method == "VB":
            diff_const_conc = self.diff_params[0]
            diff_const_rate = self.exp_transform(self.diff_params[1]) + self.dist_eps
            diff_params = torch.tensor([diff_const_conc, diff_const_rate])
            diff_dist = LogNormal(diff_const_conc, diff_const_rate)
            diff_const = diff_dist.rsample()

            
            noise_params = self.exp_transform(noise_params) + self.dist_eps
            noise_epsilon_dist = Beta(noise_params[:,0], noise_params[:,1])
            noise_epsilon = noise_epsilon_dist.rsample()
            
            out_dict = dict(
                noise_epsilon=noise_epsilon, 
                diff_const=diff_const, 
                z=z, 
                diff_params=diff_params,
                noise_params=noise_params,
                diff_dist=diff_dist,
                noise_epsilon_dist=noise_epsilon_dist,
            )
            
        elif self.method == "MLE":
            # global
            #diff_const = diff_params[0]
            diff_const = self.exp_transform(self.diff_const)
            noise_epsilon = self.sigmoid_transform(noise_params.ravel())
            
            out_dict = dict(
                noise_epsilon=noise_epsilon, 
                diff_const=diff_const, 
                z=z, 
            )
        
        return out_dict

    def _get_generative_input(self, tensors, inference_outputs):
        """Parse the dictionary to get appropriate args for generator"""
        z = inference_outputs["z"]
        noise_epsilon = inference_outputs["noise_epsilon"]
        diff_const = inference_outputs["diff_const"]
        x = tensors[REGISTRY_KEYS.X_KEY]

        # here we extract the number of UMIs per cell as a known quantity
        scale = torch.sum(x, dim=1, keepdim=True)
        
        # and we get the noise distribution
        lambda_noise = self.normalized_noise_dist
        
        if self.grid_approximation:
            precomputed_distances = self.precomputed_distances
        else:
            precomputed_distances = None
        
        input_dict = dict(
            z=z,
            scale=scale,
            noise_epsilon=noise_epsilon,
            diff_const=diff_const,
            lambda_noise=lambda_noise,
            precomputed_distances=precomputed_distances
        )
        return input_dict    
    
    @auto_move_data
    def generative(self, z, diff_const, noise_epsilon, scale, lambda_noise, precomputed_distances, **kwargs):
        """Runs the generative model."""
        # m_tests, p_tags
        lambda_signal = self.diffusion_decoder(
            z = z, 
            diffusion_constant = diff_const,
            precomputed_distances = precomputed_distances,
        )
        
        if self.grid_approximation:
            # tensor of shape n_cells, m_gridpoints, p_tags
            scale = scale.unsqueeze(1)
            noise_epsilon = noise_epsilon.unsqueeze(1).unsqueeze(2)
            lambda_signal = lambda_signal.unsqueeze(0)
            lambda_signal = lambda_signal.to(noise_epsilon.device)
            lambda_noise = lambda_noise.unsqueeze(0).unsqueeze(1)
            lambda_noise = lambda_noise.to(noise_epsilon.device)
            
        lambda_combined = lambda_signal * (1 - noise_epsilon) + lambda_noise * noise_epsilon
        
        # normalize the scaling factor according to the total rate
        #---------------------------------------------------------
        # Since the diffusion kernel rate vector does not sum to one we need
        # to renormalize the library scale here. Forcing it to sum to 1 would
        # be wrong, as the magnitude should not be constant and depend on  the
        # position and diffusion constant.
        #scale_norm = (
        #    lambda_signal.sum(axis=-1, keepdims=True) * (1 - noise_epsilon) 
        #    + noise_epsilon
        #)
        
        lambda_scaled = lambda_combined / lambda_combined.sum(axis=-1, keepdims=True) * scale
        #lambda_scaled = lambda_combined * scale / scale_norm
        
        # Here, we use a Poisson measurement model
        expected_counts_dist = torch.distributions.Poisson(lambda_scaled)
        expected_counts = expected_counts_dist.sample()
        
        return dict(
            lambda_signal=lambda_signal,
            lambda_noise=lambda_noise,
            lambda_combined=lambda_combined,
        #    scale_norm=scale_norm,
            lambda_scaled=lambda_scaled,
            expected_counts_dist=expected_counts_dist,
            expected_counts=expected_counts,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        
        # get required information
        observed_counts = tensors[REGISTRY_KEYS.X_KEY]
        expected_counts = generative_outputs["expected_counts"]
        expected_counts_dist = generative_outputs["expected_counts_dist"]
        diff_const = inference_outputs["diff_const"]
        noise_epsilon = inference_outputs["noise_epsilon"].cpu()
        
        # calculate log likelyhood of the tag counts
        llik, logpostprob = self.post_prob(observed_counts, expected_counts_dist)
        E_loglik_post_per_cell = self.E_loglik_post(llik, logpostprob)
        E_loglik_post = self.loss_agregation(E_loglik_post_per_cell)
        
        if self.method == "VB":
            # calculate KL-divergence latents
            diff_dist = inference_outputs["diff_dist"]
            noise_epsilon_dist = inference_outputs["noise_epsilon_dist"]
            
            kl_divergence_diffusion = kl_divergence(diff_dist, self.prior_diffusion)
            kl_divergence_noise = kl_divergence(noise_epsilon_dist, self.prior_noise)
            kl_divergence_complete = self.loss_agregation(kl_divergence_noise) + kl_divergence_diffusion
            elbo = E_loglik_post - kl_divergence_complete
            loss = -elbo
            loss_out = LossOutput(
                    loss=loss, 
                    reconstruction_loss=E_loglik_post_per_cell, 
                    kl_local=kl_divergence_noise, 
                    kl_global=kl_divergence_complete,
                    extra_metrics = {
                    'noise_epsilon':noise_epsilon.mean(),
                    'noise_params0':inference_outputs["noise_params"].mean(axis=0)[0],
                    'noise_params1':inference_outputs["noise_params"].mean(axis=0)[1],
                    'diff_const':diff_const,
                    'diff_params0':inference_outputs["diff_params"][0],
                    'diff_params1':inference_outputs["diff_params"][1]},
            )
            
        elif self.method == "MLE":
            # calculate log likelyhood of the latents
            log_prob_diff_const = self.loss_agregation(self.prior_diffusion.log_prob(diff_const))
            log_prob_noise_e = self.loss_agregation(self.prior_noise.log_prob(noise_epsilon))
        
            loss = -(E_loglik_post + log_prob_diff_const + log_prob_noise_e)
            loss_out = LossOutput(
                loss=loss, 
                reconstruction_loss=E_loglik_post_per_cell, 
                extra_metrics = {
                    'diff_const':diff_const,
                    'noise_epsilon':noise_epsilon.mean()},
            )
        
        return loss_out

    def post_prob(self, observed_counts, expected_counts_dist):
        """gives log likelihood for observed counts at given position"""
        # n_cells, 1, tags
        observed_counts = observed_counts.unsqueeze(1)
        # n_cells, flattened_xy, tags
        logpmf = expected_counts_dist.log_prob(observed_counts)
        # n_cells, flattened_xy
        loglik = logpmf.sum(axis=-1)
        # n_cells, flattened_xy
        logpostprob = loglik - torch.logsumexp(loglik, axis=1,keepdims=True)        
        return loglik, logpostprob
    
    def position_and_variance(self, logpostprob, xy_coords):
        """Since we are assuming an uniform prior distribution over all grid position,
        this returns the MAP cell position estimate for grid approach"""
        postprob = torch.exp(logpostprob)
        exp_pos_x = torch.sum(xy_coords[:,0]*postprob, axis=-1, keepdims=True)
        exp_pos_y = torch.sum(xy_coords[:,1]*postprob, axis=-1, keepdims=True)

        exp_var_x = torch.sum(((xy_coords[:,0] - exp_pos_x) ** 2) * postprob, axis=-1)
        exp_var_y = torch.sum(((xy_coords[:,1] - exp_pos_y) ** 2) * postprob, axis=-1)

        exp_err = torch.sqrt(exp_var_x+exp_var_y)
        return exp_pos_x.ravel(), exp_pos_y.ravel(), exp_err
    
    def MLE_position_estimate(self, logpostprob, xy_coords):
        """returns MLE cell position for grid approach"""
        mle_pos_x = xy_coords[logpostprob.argmax(axis=-1),0]
        mle_pos_y = xy_coords[logpostprob.argmax(axis=-1),1]
        return mle_pos_x, mle_pos_y

    def E_loglik_post(self, loglik, logpostprob):
        # marginal likelihood per cell k: $L(A|c_k)$
        E_loglik_post_per_cell = torch.sum(
            loglik * torch.exp(logpostprob),
            axis=1
        ) 
        # full marginal likelihood: $L(A) = \sum_k L(A|c_k)$
        E_loglik_post = torch.sum(E_loglik_post_per_cell)
        return E_loglik_post_per_cell
    
    def plot_prior_dist(self, show=False):
        """plot the prior distributions to check choice of hyperparameters"""
        
        gamma_dist = self.prior_diffusion.log_prob(torch.arange(10**-6,10,.01))
        beta_dist = self.prior_noise.log_prob(torch.arange(0,1,.01))
        
        fig, ax = plt.subplots(1, 2, figsize=(6,3))
        ax[0].plot(np.arange(0,10,.01), np.exp(gamma_dist))
        ax[0].set_title('diffusion prior ({})'.format(self.prior_diffusion))
        ax[1].plot(torch.arange(0,1,.01), np.exp(beta_dist))
        ax[1].set_title('noise prior ({})'.format(self.prior_noise))
        if show:
            plt.show()
        else:
            return ax
            
    def make_grid(self, grid_oversampling: float, encoding_x: Iterable, encoding_y: Iterable):
        """
        set up a sampling grid accross the domain of x and y.
        
        Input:
        ------
        grid_oversampling = numper of grid points to generate relative to
            the number of array points.
            
        Output:
        -------
        xy = meshgrid over domain of the spatial array with number of points
            equal to n_spots * grid_oversampling
        """
            
        x_spots = len(set(encoding_x.numpy()))
        y_spots = len(set(encoding_y.numpy()))
        x_test, y_test = torch.meshgrid(
            torch.linspace(
                torch.min(encoding_x), 
                torch.max(encoding_x), 
                int(x_spots * grid_oversampling), 
                dtype=torch.float
            ), 
            torch.linspace(
                torch.min(encoding_y), 
                torch.max(encoding_y), 
                int(y_spots * grid_oversampling), 
                dtype=torch.float
            ),
            indexing='xy'
        )
        self.x_test = torch.ravel(x_test)
        self.y_test = torch.ravel(y_test)
        xy = torch.stack([self.x_test,self.y_test]).T
        return xy

