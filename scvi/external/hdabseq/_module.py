from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderTOTALVI, EncoderTOTALVI, one_hot

from scvi.module._totalvae import TOTALVAE

torch.backends.cudnn.benchmark = True

class ABSEQ_VAE(TOTALVAE):
    """
    Reduction of the TotalVAE model for antibody only data

    Implements the totalVI model of :cite:p:`GayosoSteier21`.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_input_proteins
        Number of input proteins
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    protein_batch_mask
        Dictionary where each key is a batch code, and value is for each protein, whether it was observed or not.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    protein_background_prior_mean
        Array of proteins by batches, the prior initialization for the protein background mean (log scale)
    protein_background_prior_scale
        Array of proteins by batches, the prior initialization for the protein background scale (log scale)
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 12,
        n_layers_encoder: Tunable[int] = 2,
        n_layers_decoder: Tunable[int] = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: Tunable[float] = 0.2,
        dropout_rate_encoder: Tunable[float] = 0.2,
        gene_dispersion: Tunable[Literal["gene", "gene-batch", "gene-label"]] = "gene",
        protein_dispersion: Tunable[
            Literal["protein", "protein-batch", "protein-label"]
        ] = "protein",
        log_variational: bool = True,
        gene_likelihood: Tunable[Literal["zinb", "nb"]] = "nb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        **totalvae_kwargs
    ):
        super().__init__(
        n_input_genes = n_input_genes,
        n_input_proteins = n_input_proteins,
        n_batch = n_batch,
        n_labels = n_labels,
        n_hidden = n_hidden,
        n_latent = n_latent,
        n_layers_encoder = n_layers_encoder,
        n_layers_decoder = n_layers_decoder,
        n_continuous_cov = n_continuous_cov,
        n_cats_per_cov = n_cats_per_cov,
        dropout_rate_decoder = dropout_rate_decoder,
        dropout_rate_encoder = dropout_rate_encoder,
        gene_dispersion = gene_dispersion,
        protein_dispersion = protein_dispersion,
        log_variational = log_variational,
        gene_likelihood = gene_likelihood,
        latent_distribution = latent_distribution,
        protein_batch_mask = protein_batch_mask,
        encode_covariates = encode_covariates,
        protein_background_prior_mean = protein_background_prior_mean,
        protein_background_prior_scale = protein_background_prior_scale,
        use_size_factor_key = use_size_factor_key,
        use_observed_lib_size = use_observed_lib_size,
        library_log_means = library_log_means,
        library_log_vars = library_log_vars,
        use_batch_norm = use_batch_norm,
        use_layer_norm = use_layer_norm,
         **totalvae_kwargs,
        )


