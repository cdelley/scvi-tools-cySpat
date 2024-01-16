import logging
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData

from scvi import REGISTRY_KEYS
from scvi._types import Number
from scvi.data import AnnDataManager, fields
from scvi.data._utils import _check_nonnegative_integers
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_manager,
    _init_library_size,
)

from ._module import ABSEQ_VAE
from ._differential import _de_hdabseq

from scvi.model._totalvi import TOTALVI

from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)


class ABseqVI(TOTALVI): #RNASeqMixin, DEMixin
    """
    """

    _module_cls = ABSEQ_VAE

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        gene_dispersion: Literal[
            "gene", "gene-batch", "gene-label", "gene-cell"
        ] = "gene",
        protein_dispersion: Literal[
            "protein", "protein-batch", "protein-label"
        ] = "protein",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        empirical_protein_background_prior: Optional[bool] = None,
        override_missing_proteins: bool = False,
        **model_kwargs,
    ):
        super().__init__(
            adata,
            n_latent,
            gene_dispersion,
            protein_dispersion,
            gene_likelihood,
            latent_distribution,
            empirical_protein_background_prior,
            override_missing_proteins,
            **model_kwargs)

    def _expression_for_de(
        self,
        adata=None,
        indices=None,
        n_samples_overall=None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        scale_protein=False,
        batch_size: Optional[int] = None,
        sample_protein_mixing=False,
        include_protein_background=False,
        protein_prior_count=0.5,
    ):
        _, protein = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            n_samples_overall=n_samples_overall,
            transform_batch=transform_batch,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
        )
        protein += protein_prior_count

        return protein
            
    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        protein_prior_count: float = 0.1,
        scale_protein: bool = False,
        sample_protein_mixing: bool = False,
        include_protein_background: bool = False,
        balance_sample_covariate: bool = False,
        sample_covariate_index: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        r"""\.

        A stript down variant of TotalVI for Antibody and DNA experiments. The model components for 
        RNA have been removed, and the raw antibody counts are expected to reside in anndata.AnnData.X

        Parameters
        ----------
        protein_prior_count
            Prior count added to protein expression before LFC computation
        scale_protein
            Force protein values to sum to one in every single cell (post-hoc normalization)
        sample_protein_mixing
            Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
            that determines if expression is from foreground/background.
        include_protein_background
            Include the protein background component as part of the protein expression
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)            
        
        model_fn = partial(
            self._expression_for_de,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
            protein_prior_count=protein_prior_count,
            batch_size=batch_size,
        )
        
        col_names = self.protein_state_registry.column_names 
        
        result = _de_hdabseq(
            adata_manager,
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            hdab_seq_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            balance_sample_covariate = balance_sample_covariate,
            sample_covariate_index = sample_covariate_index,
            **kwargs,
        )

        return result
        
def hdab_seq_raw_counts_properties(
    adata_manager: AnnDataManager,
    idx1: Union[List[int], np.ndarray],
    idx2: Union[List[int], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Computes and returns some statistics on the raw counts of two sub-populations.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object setup with :class:`~scvi.model.TOTALVI`.
    idx1
        subset of indices describing the first population.
    idx2
        subset of indices describing the second population.

    Returns
    -------
    type
        Dict of ``np.ndarray`` containing, by pair (one for each sub-population),
        mean expression per gene, proportion of non-zero expression per gene, mean of normalized expression.
    """
    protein_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)

    nan = np.array([np.nan] * adata_manager.summary_stats.n_proteins)
    protein_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
    mean1_pro = np.asarray(protein_exp[idx1].mean(0))
    mean2_pro = np.asarray(protein_exp[idx2].mean(0))
    nonz1_pro = np.asarray((protein_exp[idx1] > 0).mean(0))
    nonz2_pro = np.asarray((protein_exp[idx2] > 0).mean(0))
    properties = {
        "raw_mean1": mean1_pro,
        "raw_mean2":mean2_pro,
        "non_zeros_proportion1": nonz1_pro,
        "non_zeros_proportion2": nonz2_pro,
    }

    return properties
        
