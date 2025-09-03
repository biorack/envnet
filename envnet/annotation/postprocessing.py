"""
Post-processing utilities for annotation results.
"""

import pandas as pd
from typing import Dict, Optional

from ..config.annotation_config import AnnotationConfig


class AnnotationPostprocessor:
    """Handles post-processing of annotation results."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def format_ms1_results(self, ms1_data: pd.DataFrame, 
                          file_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Format MS1 annotation results with metadata.
        
        Args:
            ms1_data: Raw MS1 feature data
            file_metadata: Experimental file metadata
            
        Returns:
            pd.DataFrame: Formatted MS1 results
        """
        # Check if environmental metadata was included in annotation
        if 'environmental_subclass' in file_metadata.columns and 'name' in file_metadata.columns:
            metadata_cols = ['h5', 'lcmsrun_observed', 'environmental_subclass', 'name']

            important_cols = [
                'original_index', 'lcmsrun_observed', 'annotation_method',
                'mz_centroid', 'rt_peak', 'peak_height', 'peak_area',
                'num_datapoints', 'environmental_subclass', 'name'
            ]
        else:
            metadata_cols = ['h5', 'lcmsrun_observed']

            important_cols = [
                'original_index', 'lcmsrun_observed', 'annotation_method',
                'mz_centroid', 'rt_peak', 'peak_height', 'peak_area',
                'num_datapoints']

        # Add file metadata
        ms1_results = pd.merge(
            ms1_data,
            file_metadata[metadata_cols],
            left_on='lcmsrun_observed',
            right_on='lcmsrun_observed',
            how='left'
        )
        # Add annotation type
        ms1_results['annotation_method'] = 'MS1_precursor_match'
        ms1_results['confidence_level'] = 'MS1_match'
        
        other_cols = [col for col in ms1_results.columns if col not in important_cols]
        ms1_results = ms1_results[important_cols + other_cols]
        
        return ms1_results
    
    def format_ms2_results(self, ms2_scores: pd.DataFrame, 
                          ms2_data: pd.DataFrame, 
                          spectrum_type: str = 'deconvoluted') -> pd.DataFrame:
        """
        Format MS2 annotation results.
        
        Args:
            ms2_scores: Spectral matching scores
            ms2_data: Original MS2 experimental data
            spectrum_type: Type of spectrum matched
            
        Returns:
            pd.DataFrame: Formatted MS2 results
        """
        # Merge scores with experimental data
        suffix = f'_{spectrum_type}_match'
        ms2_scores_suffixed = ms2_scores.add_suffix(suffix)
        index_col = f'ms2_data_index{suffix}'
        
        ms2_results = pd.merge(
            ms2_scores_suffixed,
            ms2_data,
            left_on=index_col,
            right_index=True,
            how='left'
        )
        
        # Add annotation metadata
        ms2_results['annotation_method'] = f'MS2_{spectrum_type}_spectral_match'
        ms2_results['spectrum_type'] = spectrum_type
        
        # Add confidence levels based on score
        score_col = f'score{suffix}'
        if score_col in ms2_results.columns:
            ms2_results['confidence_level'] = ms2_results[score_col].apply(
                self._assign_confidence_level
            )

        # Remove 2d spectrum columns to avoid errors when saving as a parquet file
        include_cols = [col for col in ms2_results.columns if col not in ['deconvoluted_spectrum', 'original_spectrum']]
        ms2_results = ms2_results[include_cols]

        return ms2_results
    
    def _assign_confidence_level(self, score: float) -> str:
        """Assign confidence level based on spectral match score."""
        if score >= 0.9:
            return 'High'
        elif score >= 0.7:
            return 'Medium'
        elif score >= 0.5:
            return 'Low'
        else:
            return 'Very_Low'
    
    def add_compound_annotations(self, results: pd.DataFrame, 
                               envnet_data: Dict) -> pd.DataFrame:
        """
        Add compound information from ENVnet nodes.
        
        Args:
            results: Annotation results
            envnet_data: ENVnet reference data
            
        Returns:
            pd.DataFrame: Results with compound annotations
        """
        # Merge with node data for compound information
        compound_info = envnet_data['nodes'][['original_index', 'precursor_mz']].copy()
        
        # Add any additional compound metadata available
        ref_col = 'ref' if 'ref' in results.columns else 'original_index'
        
        results = pd.merge(
            results,
            compound_info.add_suffix('_envnet'),
            left_on=ref_col,
            right_on='original_index_envnet',
            how='left'
        )
        
        return results
    
    def generate_summary_stats(self, results: pd.DataFrame) -> Dict:
        """Generate summary statistics for annotation results."""
        print(results.columns)
        stats = {
            'total_annotations': len(results),
            'unique_compounds': results['original_index'].nunique() if 'original_index' in results.columns else 0,
            'unique_samples': results['lcmsrun_observed'].nunique() if 'lcmsrun_observed' in results.columns else 0
        }
        
        # Add confidence level breakdown for MS2 results
        if 'confidence_level' in results.columns:
            conf_counts = results['confidence_level'].value_counts().to_dict()
            stats['confidence_breakdown'] = conf_counts
        
        # Add score statistics for MS2 results
        score_cols = [col for col in results.columns if 'score' in col.lower()]
        if score_cols:
            score_col = score_cols[0]
            stats['score_stats'] = {
                'mean': results[score_col].mean(),
                'median': results[score_col].median(),
                'min': results[score_col].min(),
                'max': results[score_col].max()
            }
        
        return stats