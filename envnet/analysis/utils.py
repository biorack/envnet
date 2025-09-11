"""
Utility functions for analysis workflows.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict

from ..config.analysis_config import AnalysisConfig


class AnalysisUtils:
    """Utility functions for data processing and analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def get_best_ms1_data(self, 
                         ms1_data: pd.DataFrame,
                         node_data: Optional[pd.DataFrame] = None,
                         peak_value: str = 'peak_area') -> pd.DataFrame:
        """
        Get best MS1 hit per compound (highest peak area/height).
        
        Args:
            ms1_data: MS1 annotation data
            node_data: ENVnet node data for ppm error calculation
            peak_value: 'peak_area' or 'peak_height'
            
        Returns:
            pd.DataFrame: Best MS1 hits per compound
        """
        if peak_value not in ['peak_area', 'peak_height']:
            raise ValueError("peak_value must be 'peak_area' or 'peak_height'")
        
        # Sort by peak value and get best hit per compound
        best_ms1 = ms1_data.copy()
        best_ms1.sort_values(peak_value, ascending=False, inplace=True)
        best_ms1.drop_duplicates(subset='original_index', inplace=True)
        
        # Calculate ppm error if node data provided
        if node_data is not None:
            best_ms1 = pd.merge(
                best_ms1, 
                node_data[['original_index', 'precursor_mz']], 
                on='original_index', 
                how='left'
            )
            best_ms1['ppm_error'] = best_ms1.apply(
                lambda x: ((x.precursor_mz - x.mz_centroid) / x.precursor_mz) * 1000000, 
                axis=1
            )
        
        return best_ms1
    
    def get_best_ms2_data(self, ms2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get best MS2 hit per compound (highest score).
        
        Args:
            ms2_data: MS2 annotation data
            
        Returns:
            pd.DataFrame: Best MS2 hits per compound
        """
        best_ms2 = ms2_data.copy()
        
        # Determine score column
        score_col = None
        for col in ['score', 'score_deconvoluted_match', 'score_original_match']:
            if col in best_ms2.columns:
                score_col = col
                break
        
        if score_col is None:
            print("Warning: No score column found in MS2 data")
            return best_ms2.drop_duplicates(subset='original_index')
        
        # Sort by score and get best hit per compound
        best_ms2.sort_values(score_col, ascending=False, inplace=True)
        best_ms2.drop_duplicates(subset='original_index', inplace=True)
        
        # Clean up columns
        drop_cols = ['query', 'ref']
        best_ms2.drop(columns=[col for col in drop_cols if col in best_ms2.columns], 
                     inplace=True)
        best_ms2.reset_index(drop=True, inplace=True)
        
        return best_ms2
    
    def combine_ms1_ms2_data(self, 
                            ms1_data: pd.DataFrame,
                            ms2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine MS1 and MS2 data with proper handling of overlapping columns.
        
        Args:
            ms1_data: Best MS1 hits
            ms2_data: Best MS2 hits
            
        Returns:
            pd.DataFrame: Combined MS1/MS2 data
        """
        # Prepare MS2 columns
        ms2_cols = ['original_index', 'score', 'matches', 'best_match_method', 'lcmsrun_observed']
        available_ms2_cols = [col for col in ms2_cols if col in ms2_data.columns]
        
        ms2_subset = ms2_data[available_ms2_cols].add_prefix('ms2_')
        
        # Merge data
        combined = pd.merge(
            ms1_data,
            ms2_subset,
            left_on='original_index',
            right_on='ms2_original_index',
            how='outer'
        )
        
        # Sort by MS2 score if available
        if 'ms2_score' in combined.columns:
            combined.sort_values('ms2_score', ascending=False, inplace=True)
        
        combined.reset_index(drop=True, inplace=True)
        return combined
    
    def create_output_dataframe(self, 
                               node_data: pd.DataFrame,
                               best_hits: pd.DataFrame,
                               stats_data: Optional[pd.DataFrame] = None,
                               output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive output DataFrame with all analysis results.
        
        Args:
            node_data: ENVnet node data
            best_hits: Best annotation hits
            stats_data: Statistical analysis results
            output_file: Optional file to save results
            
        Returns:
            pd.DataFrame: Comprehensive output data
        """
        # Start with node data
        output = node_data.copy().add_prefix('envnet_')
        output.set_index('envnet_original_index', inplace=True)
        
        # Add best hits
        if 'original_index' in best_hits.columns:
            best_hits = best_hits.rename(columns={'original_index': 'envnet_original_index'})
        
        output = output.join(
            best_hits.set_index('envnet_original_index'),
            rsuffix='_best_hit',
            how='left'
        )
        
        # Add statistical results
        if stats_data is not None:
            output = output.join(
                stats_data,
                rsuffix='_stats',
                how='left'
            )
        
        # Save if requested
        if output_file:
            output.to_csv(output_file)
        
        return output
    
    def validate_data_consistency(self, 
                                 ms1_data: pd.DataFrame,
                                 ms2_data: Optional[pd.DataFrame] = None,
                                 file_metadata: Optional[pd.DataFrame] = None) -> Dict[str, bool]:
        """
        Validate data consistency across datasets.
        
        Args:
            ms1_data: MS1 data
            ms2_data: MS2 data
            file_metadata: File metadata
            
        Returns:
            Dict with validation results
        """
        validation = {}
        
        # Check MS1 data
        validation['ms1_has_required_columns'] = all(
            col in ms1_data.columns 
            for col in ['original_index', 'lcmsrun_observed', self.config.peak_value]
        )
        
        # Check MS2 data consistency
        if ms2_data is not None:
            validation['ms2_has_original_index'] = 'original_index' in ms2_data.columns
            validation['ms2_has_score'] = any(
                col in ms2_data.columns 
                for col in ['score', 'score_deconvoluted_match', 'score_original_match']
            )
            
            # Check file overlap
            if validation['ms1_has_required_columns']:
                ms1_files = set(ms1_data['lcmsrun_observed'].unique())
                ms2_files = set(ms2_data['lcmsrun_observed'].unique()) if 'lcmsrun_observed' in ms2_data.columns else set()
                validation['file_overlap'] = len(ms1_files.intersection(ms2_files)) > 0
        
        # Check metadata consistency
        if file_metadata is not None:
            validation['metadata_has_categories'] = 'sample_category' in file_metadata.columns
            
            if validation['ms1_has_required_columns']:
                ms1_files = set(ms1_data['lcmsrun_observed'].unique())
                meta_files = set(file_metadata['filename'].unique()) if 'filename' in file_metadata.columns else set()
                validation['metadata_file_overlap'] = len(ms1_files.intersection(meta_files)) > 0
        
        return validation
    
    def filter_data_by_criteria(self, 
                               data: pd.DataFrame,
                               min_peak_value: Optional[float] = None,
                               max_ppm_error: Optional[float] = None,
                               min_ms2_score: Optional[float] = None) -> pd.DataFrame:
        """
        Filter data by various quality criteria.
        
        Args:
            data: Data to filter
            min_peak_value: Minimum peak area/height
            max_ppm_error: Maximum ppm error (requires ppm_error column)
            min_ms2_score: Minimum MS2 score (requires ms2_score column)
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered = data.copy()
        original_count = len(filtered)
        
        # Filter by peak value
        if min_peak_value is not None and self.config.peak_value in filtered.columns:
            filtered = filtered[filtered[self.config.peak_value] >= min_peak_value]
            print(f"Peak value filter: {len(filtered)}/{original_count} remaining")
        
        # Filter by ppm error
        if max_ppm_error is not None and 'ppm_error' in filtered.columns:
            filtered = filtered[abs(filtered['ppm_error']) <= max_ppm_error]
            print(f"PPM error filter: {len(filtered)}/{original_count} remaining")
        
        # Filter by MS2 score
        if min_ms2_score is not None:
            score_cols = [col for col in filtered.columns if 'score' in col.lower()]
            if score_cols:
                score_col = score_cols[0]
                filtered = filtered[filtered[score_col] >= min_ms2_score]
                print(f"MS2 score filter: {len(filtered)}/{original_count} remaining")
        
        return filtered


def read_dataframe_flexible(source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Read DataFrame from various sources.
    
    Args:
        source: DataFrame, CSV file path, or parquet file path
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if isinstance(source, pd.DataFrame):
        return source
    elif isinstance(source, str):
        if source.endswith('.parquet'):
            return pd.read_parquet(source)
        elif source.endswith('.csv'):
            return pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported file format: {source}")
    else:
        raise ValueError("Source must be DataFrame or file path")