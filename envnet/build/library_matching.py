"""
Library matching functionality using BLINK scoring.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import multiprocessing

from ..config.build_config import BuildConfig
import blink


class LibraryMatcher:
    """Handles library matching against reference spectra using BLINK."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def score_spectra_chunk(self, df: pd.DataFrame, scoring_type: str = 'deconvoluted') -> pd.DataFrame:
        """Score a chunk of spectra against reference library."""
        df.reset_index(inplace=True, drop=True)
        
        # Prepare query data
        query_pmz = df['precursor_mz'].tolist()
        q_cols = ['precursor_mz', 'original_index']
        r_cols = ['original_p2d2_index', 'formula', 'precursor_mz', 'inchi_key', 'name', 'smiles']
        
        # Select spectra type
        if scoring_type == 'deconvoluted':
            query_spec = df.apply(
                lambda x: np.asarray([x['deconvoluted_spectrum_mz_vals'], 
                                    x['deconvoluted_spectrum_intensity_vals']]), axis=1
            ).tolist()
            ref_spec = self.config.ref_spec_nl
            ref_pmz = self.config.ref_pmz_nl
            full_ref = self.config.ref2
        elif scoring_type == 'original':
            query_spec = df.apply(
                lambda x: np.asarray([x['original_spectrum_mz_vals'], 
                                    x['original_spectrum_intensity_vals']]), axis=1
            ).tolist()
            ref_spec = self.config.ref_spec
            ref_pmz = self.config.ref_pmz
            full_ref = self.config.ref
        else:
            raise ValueError(f"Unknown scoring_type: {scoring_type}")
        
        # Discretize spectra
        d_specs = blink.discretize_spectra(
            query_spec, ref_spec, query_pmz, ref_pmz,
            intensity_power=self.config.intensity_power,
            bin_width=self.config.bin_width,
            tolerance=self.config.mz_tol,
            network_score=False
        )
        
        # Score and filter
        hits = self._score_and_filter(d_specs, full_ref, df, q_cols, r_cols)
        return hits
    
    def _score_and_filter(self, specs, r: pd.DataFrame, q: pd.DataFrame, 
                         q_cols: List[str], r_cols: List[str]) -> pd.DataFrame:
        """Score spectra and filter results."""
        scores = blink.score_sparse_spectra(specs)
        filtered_scores = blink.filter_hits(
            scores, 
            min_score=self.config.min_score,
            min_matches=self.config.min_matches,
            override_matches=self.config.override_matches
        )
        
        mz_df = blink.reformat_score_matrix(filtered_scores)
        mz_df = blink.make_output_df(mz_df)
        
        # Convert sparse to dense
        for c in mz_df.columns:
            mz_df[c] = mz_df[c].sparse.to_dense()
        
        # Merge with query and reference data
        mz_df = pd.merge(mz_df, q[q_cols], left_on='query', right_index=True)
        mz_df = pd.merge(mz_df, r[r_cols].add_suffix('_ref'), left_on='ref', right_index=True)
        
        # Apply filters
        mz_df = mz_df[abs(mz_df['precursor_mz'] - mz_df['precursor_mz_ref']) < self.config.mz_tol]
        mz_df = mz_df[mz_df['score'] > self.config.min_score]
        mz_df['matches'] = mz_df['matches'].astype(int)
        mz_df = mz_df[mz_df['matches'] >= self.config.min_matches]
        
        return mz_df
    
    def score_all_spectra(self, df: pd.DataFrame, scoring_type: str = 'deconvoluted', 
                         keep_top_hits: bool = False) -> pd.DataFrame:
        """Score all spectra against library, processing by filename."""
        groups = [gg for _, gg in df.groupby('filename')]
        groups = sorted(groups, key=lambda x: x.shape[0], reverse=True)
        
        results = []
        for i, gg in enumerate(groups):
            library_matches = self.score_spectra_chunk(gg, scoring_type=scoring_type)
            if library_matches is not None and len(library_matches) > 0:
                if keep_top_hits:
                    library_matches.sort_values('score', ascending=False, inplace=True)
                    library_matches.drop_duplicates('original_index', keep='first', inplace=True)
                
                print(f'Processing {scoring_type} {gg["basename"].iloc[0]} - found {library_matches.shape[0]} matches')
                results.append(library_matches)
        
        if not results:
            return pd.DataFrame()
        
        # Combine results
        all_matches = pd.concat(results, ignore_index=True)
        all_matches.rename(columns={
            'inchi_key_ref': 'inchi_key', 
            'name_ref': 'compound_name', 
            'smiles_ref': 'smiles'
        }, inplace=True)
        
        return all_matches