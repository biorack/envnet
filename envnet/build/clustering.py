"""
Spectral clustering to eliminate duplicate spectra.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..config.build_config import BuildConfig
from ..vendor.feature_tools import group_consecutive
from .formula_tools import get_formula_props
import blink


class SpectraClusterer:
    """Handles clustering of similar spectra to eliminate duplicates."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def cluster_duplicate_spectra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster duplicate spectra based on precursor m/z and MS2 similarity.
        
        Groups spectra by precursor m/z, then finds connected components
        based on spectral similarity within each group.
        """
        print('Chunking spectra into groups based on precursor m/z...')
        
        df.sort_values('precursor_mz', inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        # Group by precursor m/z tolerance
        df['precursor_mz_group'] = group_consecutive(
            df['precursor_mz'].values[:], 
            stepsize=self.config.mz_tol, 
            do_ppm=False
        )
        
        print(f'Found {df["precursor_mz_group"].nunique()} unique precursor m/z groups')
        
        # Process each group
        groups = [gg for _, gg in df.groupby('precursor_mz_group')]
        groups = sorted(groups, key=lambda x: x.shape[0], reverse=True)
        
        clustered_groups = []
        for i, group in enumerate(groups):
            cluster_labels = self._cluster_group(group)
            group['cluster_label'] = cluster_labels
            clustered_groups.append(group)
            
            print(f'Counter: {i}, Processing {group.shape[0]} entries for m/z range '
                  f'{group["precursor_mz"].min():.4f} to {group["precursor_mz"].max():.4f}, '
                  f'Found {group["cluster_label"].nunique()} unique spectra')
        
        return pd.concat(clustered_groups, ignore_index=True)
    
    def _cluster_group(self, df: pd.DataFrame) -> List[int]:
        """Cluster a single precursor m/z group."""
        if df.shape[0] == 1:
            return [0]
        
        # Find similar spectra pairs
        r, c = self._label_similar_spectra(df)
        
        if r is None or c is None:
            # No similar pairs found
            return list(range(df.shape[0]))
        
        # Get connected components
        component_labels = self._get_connected_components(r, c, df.shape[0])
        return component_labels.tolist()
    
    def _label_similar_spectra(self, df: pd.DataFrame, 
                              chunk_size: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find pairs of spectra with similar precursor m/z and high MS2 similarity."""
        df.reset_index(inplace=True, drop=True)
        
        # Prepare spectral data
        spec = df.apply(
            lambda row: np.array([row['deconvoluted_spectrum_mz_vals'],
                                row['deconvoluted_spectrum_intensity_vals']]), 
            axis=1
        ).tolist()
        precursors = df['precursor_mz'].tolist()
        precursor_array = df['precursor_mz'].values
        n = len(df)
        
        # Process in chunks to manage memory
        chunk_indices = np.array_split(np.arange(n), max(1, n // chunk_size))
        n_chunks = len(chunk_indices)
        
        all_r, all_c = [], []
        
        for i in range(n_chunks):
            chunk_i_idx = chunk_indices[i]
            spec_i = [spec[idx] for idx in chunk_i_idx]
            precursors_i = [precursors[idx] for idx in chunk_i_idx]
            precursor_array_i = precursor_array[chunk_i_idx]
            
            for j in range(i, n_chunks):
                chunk_j_idx = chunk_indices[j]
                spec_j = [spec[idx] for idx in chunk_j_idx]
                precursors_j = [precursors[idx] for idx in chunk_j_idx]
                precursor_array_j = precursor_array[chunk_j_idx]
                
                # Compute MS2 similarity
                d_specs = blink.discretize_spectra(
                    spec_i, spec_j, precursors_i, precursors_j,
                    intensity_power=self.config.intensity_power,
                    bin_width=self.config.bin_width,
                    tolerance=self.config.mz_tol,
                    network_score=False
                )
                scores = blink.score_sparse_spectra(d_specs)
                similarity_matrix = scores['mzi'].todense()
                
                # Compute precursor m/z differences
                pmz_diff = abs(np.subtract.outer(precursor_array_i, precursor_array_j))
                
                # Apply both conditions
                idx_ms2similarity = similarity_matrix > self.config.min_score
                idx_pmz_same = pmz_diff < self.config.mz_tol
                conditions = idx_pmz_same & idx_ms2similarity
                
                # Get matching pairs
                r_local, c_local = np.where(conditions)
                
                if len(r_local) > 0:
                    # Convert to global indices
                    r_global = chunk_i_idx[r_local]
                    c_global = chunk_j_idx[c_local]
                    
                    if i == j:
                        # Same chunk: only keep r < c to avoid duplicates
                        mask = r_global < c_global
                        r_global = r_global[mask]
                        c_global = c_global[mask]
                    
                    all_r.extend(r_global)
                    all_c.extend(c_global)
        
        if len(all_r) == 0:
            return None, None
        
        return np.array(all_r), np.array(all_c)
    
    def _get_connected_components(self, r: np.ndarray, c: np.ndarray, 
                                 n_nodes: int) -> np.ndarray:
        """Find connected components using scipy sparse matrices."""
        # Create sparse adjacency matrix
        data = np.ones(len(r), dtype=bool)
        
        # Make symmetric
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        data_sym = np.concatenate([data, data])
        
        # Create sparse matrix
        adj_matrix = csr_matrix((data_sym, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # Find connected components
        _, labels = connected_components(adj_matrix, directed=False)
        return labels
    
    def eliminate_redundant_spectra(self, all_spectra: pd.DataFrame,
                                library_matches_deconvoluted: pd.DataFrame,
                                library_matches_original: pd.DataFrame) -> pd.DataFrame:
        """
        Eliminate redundant spectra by keeping the best one from each cluster.
        """
        # Check if library match dataframes have expected columns
        expected_cols = ['score', 'matches', 'original_index', 'formula_ref', 'precursor_mz_ref', 
                        'inchi_key', 'compound_name', 'smiles']
        
        # Handle deconvoluted matches
        if len(library_matches_deconvoluted) > 0 and all(col in library_matches_deconvoluted.columns for col in expected_cols):
            top_matches_deconv = library_matches_deconvoluted[expected_cols].sort_values(
                'score', ascending=False).drop_duplicates('original_index', keep='first')
            top_matches_deconv['match_type'] = 'deconvoluted'
        else:
            # Create empty dataframe with expected structure
            top_matches_deconv = pd.DataFrame(columns=expected_cols + ['match_type'])
            print("Warning: No valid deconvoluted library matches found")
        
        # Handle original matches
        if len(library_matches_original) > 0 and all(col in library_matches_original.columns for col in expected_cols):
            top_matches_orig = library_matches_original[expected_cols].sort_values(
                'score', ascending=False).drop_duplicates('original_index', keep='first')
            top_matches_orig['match_type'] = 'original'
        else:
            # Create empty dataframe with expected structure
            top_matches_orig = pd.DataFrame(columns=expected_cols + ['match_type'])
            print("Warning: No valid original library matches found")
        
        # Combine matches (prefer deconvoluted)
        if len(top_matches_deconv) > 0 or len(top_matches_orig) > 0:
            top_matches = pd.concat([top_matches_deconv, top_matches_orig], axis=0)
            top_matches.sort_values(['original_index', 'score'], ascending=[True, False], inplace=True)
            top_matches.drop_duplicates(subset='original_index', keep='first', inplace=True)
            top_matches.reset_index(inplace=True, drop=True)
        else:
            # No matches at all - create empty dataframe
            top_matches = pd.DataFrame(columns=expected_cols + ['match_type'])
            print("Warning: No library matches found at all")
        
        # Merge with spectra data
        temp = pd.merge(all_spectra, top_matches, on='original_index', how='left')
        
        # Add duplicate cluster information
        g = temp.groupby(['cluster_label', 'precursor_mz_group'])['original_index'].count()
        g = g.reset_index(drop=False)
        g.rename(columns={'original_index': 'num_duplicates'}, inplace=True)
        g.index.name = 'duplicate_cluster_index'
        g.reset_index(inplace=True, drop=False)
        
        temp = pd.merge(temp, g, on=['cluster_label', 'precursor_mz_group'], how='left')
        temp.drop(columns=['cluster_label', 'precursor_mz_group'], inplace=True)
        
        # Filter and clean
        temp = temp[temp['num_duplicates'] >= self.config.min_cluster_size]
        
        # Select best representative from each duplicate cluster
        temp = self._select_cluster_representatives(temp)
        
        # Clean up columns and add formula properties
        temp = self._clean_and_enhance_data(temp)
        
        print(f'Final dataset: {temp.shape[0]} unique spectra')
        
        return temp
    
    def _select_cluster_representatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the best representative spectrum from each duplicate cluster."""
        
        def select_best_representative(group):
            """Select best spectrum from a duplicate cluster."""
            # Priority order:
            # 1. Has library match
            # 2. Highest peak area
            # 3. Highest peak height
            # 4. Most MS2 peaks
            
            # First, prefer spectra with library matches
            with_matches = group[pd.notna(group['score'])]
            if len(with_matches) > 0:
                # Among matched spectra, select highest scoring
                best_match = with_matches.loc[with_matches['score'].idxmax()]
                return best_match
            
            # If no library matches, select by peak area
            if 'peak_area' in group.columns:
                best_area = group.loc[group['peak_area'].idxmax()]
                return best_area
            
            # Fallback to first entry
            return group.iloc[0]
        
        # Group by duplicate cluster and select representatives
        representatives = df.groupby('duplicate_cluster_index').apply(select_best_representative)
        representatives.reset_index(drop=True, inplace=True)
        
        return representatives
    
    def _clean_and_enhance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and add formula properties."""
        # Rename columns for consistency
        column_mapping = {
            'formula_ref': 'library_formula',
            'precursor_mz_ref': 'library_precursor_mz'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Fill missing values
        df['compound_name'] = df['compound_name'].fillna('')
        df['inchi_key'] = df['inchi_key'].fillna('')
        df['smiles'] = df['smiles'].fillna('')
        df['library_formula'] = df['library_formula'].fillna('')
        
        # Add formula properties if we have predicted formulas
        if 'predicted_formula' in df.columns:
            formula_props = get_formula_props(df, 'predicted_formula')
            if len(formula_props) > 0:
                df = pd.merge(df, formula_props, left_on='predicted_formula', 
                            right_on='formula', how='left')
                df.drop(columns=['formula'], errors='ignore', inplace=True)
        
        # Add quality metrics
        df['has_library_match'] = pd.notna(df['score'])
        df['is_singleton'] = df['num_duplicates'] == 1
        
        # Sort by quality
        sort_cols = ['has_library_match', 'score', 'peak_area']
        sort_cols = [c for c in sort_cols if c in df.columns]
        if sort_cols:
            df.sort_values(sort_cols, ascending=[False, False, False], inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def get_clustering_statistics(self, clustered_df: pd.DataFrame) -> dict:
        """Get statistics about the clustering results."""
        total_spectra = len(clustered_df)
        
        # Cluster statistics
        cluster_sizes = clustered_df.groupby(['cluster_label', 'precursor_mz_group']).size()
        
        stats = {
            'total_spectra': total_spectra,
            'unique_clusters': len(cluster_sizes),
            'reduction_factor': total_spectra / len(cluster_sizes),
            'mean_cluster_size': cluster_sizes.mean(),
            'median_cluster_size': cluster_sizes.median(),
            'max_cluster_size': cluster_sizes.max(),
            'singleton_clusters': (cluster_sizes == 1).sum(),
            'large_clusters': (cluster_sizes >= 10).sum()
        }
        
        return stats