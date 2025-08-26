"""
MS2 spectral matching using BLINK algorithm.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from ..config.annotation_config import AnnotationConfig
import blink as blink


class MS2Matcher:
    """Handles MS2 spectral matching using BLINK."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def match_spectra(self, ms2_data: pd.DataFrame, 
                     envnet_data: Dict, spectrum_type: str) -> pd.DataFrame:
        """
        Match experimental MS2 spectra to ENVnet reference spectra.
        
        Args:
            ms2_data: Experimental MS2 data
            envnet_data: ENVnet reference data
            spectrum_type: 'deconvoluted' or 'original'
            
        Returns:
            pd.DataFrame: Spectral matching results
        """
        if spectrum_type == 'deconvoluted':
            ref_spectra = envnet_data['deconvoluted_spectra']
            mz_attr = 'deconvoluted_spectrum_mz_vals'
            intensity_attr = 'deconvoluted_spectrum_intensity_vals'
        elif spectrum_type == 'original':
            ref_spectra = envnet_data['original_spectra']
            mz_attr = 'original_spectrum_mz_vals'
            intensity_attr = 'original_spectrum_intensity_vals'
        else:
            raise ValueError("spectrum_type must be 'deconvoluted' or 'original'")
        
        return self._blink_score_and_filter(
            ms2_data, ref_spectra, envnet_data['reference_pmzs'],
            mz_attr, intensity_attr
        )
    
    def _blink_score_and_filter(self, ms2_data: pd.DataFrame,
                               ref_spectra: List[np.ndarray],
                               ref_pmzs: List[float],
                               mz_attribute: str,
                               intensity_attribute: str) -> pd.DataFrame:
        """
        Apply BLINK scoring and filtering to MS2 data.
        
        Args:
            ms2_data: Experimental MS2 data
            ref_spectra: Reference spectra list
            ref_pmzs: Reference precursor m/z values
            mz_attribute: Column name for experimental m/z values
            intensity_attribute: Column name for experimental intensity values
            
        Returns:
            pd.DataFrame: Scored and filtered spectral matches
        """
        # Reset index for sequential linking
        ms2_data.reset_index(drop=True, inplace=True)
        
        # Prepare experimental data
        exp_pmzs = ms2_data['precursor_mz'].tolist()
        exp_spectra = ms2_data.apply(
            lambda x: np.array([x[mz_attribute], x[intensity_attribute]]), 
            axis=1
        ).tolist()
        
        # Process in chunks
        exp_chunks = [exp_spectra[i:i + self.config.chunk_size] 
                     for i in range(0, len(exp_spectra), self.config.chunk_size)]
        pmz_chunks = [exp_pmzs[i:i + self.config.chunk_size] 
                     for i in range(0, len(exp_pmzs), self.config.chunk_size)]
        index_chunks = [list(range(i, i + self.config.chunk_size)) 
                       for i in range(0, len(exp_spectra), self.config.chunk_size)]
        
        # Score each chunk
        all_scores = []
        for i in tqdm(range(len(exp_chunks)), desc="Scoring spectra", unit='chunk'):
            chunk_scores = self._score_chunk(
                exp_chunks[i], pmz_chunks[i], index_chunks[i],
                ref_spectra, ref_pmzs
            )
            all_scores.append(chunk_scores)
        
        return pd.concat(all_scores, ignore_index=True)
    
    def _score_chunk(self, exp_chunk: List[np.ndarray], 
                    pmz_chunk: List[float], index_chunk: List[int],
                    ref_spectra: List[np.ndarray], 
                    ref_pmzs: List[float]) -> pd.DataFrame:
        """Score a chunk of spectra using BLINK."""
        # Discretize spectra
        discretized = blink.discretize_spectra(
            exp_chunk, ref_spectra, pmz_chunk, ref_pmzs,
            bin_width=self.config.bin_width,
            tolerance=self.config.mz_tol,
            intensity_power=self.config.intensity_power,
            trim_empty=False,
            remove_duplicates=False,
            network_score=False
        )
        
        # Score spectra
        scores = blink.score_sparse_spectra(discretized)
        scores = blink.filter_hits(
            scores,
            min_score=self.config.min_score,
            min_matches=self.config.min_matches,
            override_matches=self.config.override_matches
        )
        scores = blink.reformat_score_matrix(scores)
        scores = blink.make_output_df(scores)
        
        # Convert sparse to dense
        for col in scores.columns:
            scores[col] = scores[col].sparse.to_dense()
        
        # Convert query/ref columns to int
        for col in ['query', 'ref']:
            scores[col] = scores[col].astype(int)
        
        # Add precursor m/z information
        scores['precursor_mz_query'] = scores['query'].apply(lambda x: pmz_chunk[x])
        scores = pd.merge(
            scores, 
            pd.DataFrame({'precursor_mz_ref': ref_pmzs}).reset_index().rename(columns={'index': 'ref'}),
            on='ref', 
            how='left'
        )
        
        # Filter by precursor m/z difference
        scores['mz_diff'] = abs(scores['precursor_mz_query'] - scores['precursor_mz_ref'])
        scores = scores[scores['mz_diff'] < self.config.mz_tol]
        
        # Map back to original indices
        scores['ms2_data_index'] = scores['query'].apply(lambda x: index_chunk[x])
        scores.drop(columns=['query'], inplace=True)
        
        return scores