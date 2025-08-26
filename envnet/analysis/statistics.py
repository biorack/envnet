"""
Statistical analysis functions for ENVnet data.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, binomtest
from typing import Dict, List, Union, Optional

from ..config.analysis_config import AnalysisConfig


class StatisticalAnalyzer:
    """Performs statistical analysis on ENVnet annotation data."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def run_analysis(self, 
                    ms1_data: pd.DataFrame,
                    file_metadata: pd.DataFrame,
                    control_group: str,
                    treatment_group: str) -> pd.DataFrame:
        """
        Run complete statistical analysis.
        
        Args:
            ms1_data: MS1 feature data
            file_metadata: File metadata with sample categories
            control_group: Name of control group
            treatment_group: Name of treatment group
            
        Returns:
            pd.DataFrame: Statistical results
        """
        if 'sample_category' not in file_metadata.columns:
            raise ValueError("File metadata must contain 'sample_category' column")
            
        return self._calculate_statistics(
            ms1_data, file_metadata, control_group, treatment_group
        )
    
    def _calculate_statistics(self, 
                            ms1_data: pd.DataFrame,
                            file_metadata: pd.DataFrame,
                            control_group: str,
                            treatment_group: str) -> pd.DataFrame:
        """Calculate t-tests and fold changes."""
        # Create pivot table
        pivot_data = ms1_data.pivot_table(
            columns='original_index',
            index=['lcmsrun_observed'],
            values=self.config.peak_value,
            aggfunc='mean',
            fill_value=0
        )
        
        # Normalize data if requested
        if self.config.normalize_data:
            pivot_data = self._normalize_data(pivot_data)
        
        # Melt back to long format
        melted_data = pivot_data.reset_index().melt(
            id_vars='lcmsrun_observed',
            var_name='original_index',
            value_name=self.config.peak_value
        )
        melted_data['original_index'] = melted_data['original_index'].astype(str)
        
        # Merge with sample categories
        analysis_df = pd.merge(
            melted_data,
            file_metadata[['filename', 'sample_category']],
            left_on='lcmsrun_observed',
            right_on='filename',
            how='inner'
        )
        
        # Filter to only include specified groups
        analysis_df = analysis_df[
            analysis_df['sample_category'].isin([control_group, treatment_group])
        ]
        
        if len(analysis_df) == 0:
            raise ValueError(f"No data found for groups: {control_group}, {treatment_group}")
        
        # Calculate group statistics
        group_stats = self._calculate_group_statistics(
            analysis_df, control_group, treatment_group
        )
        
        # Calculate t-tests
        test_results = self._calculate_ttests(
            analysis_df, control_group, treatment_group
        )
        
        # Combine results
        results = pd.merge(group_stats, test_results, left_index=True, right_index=True)
        
        # Calculate fold changes
        control_mean_col = f'mean-{control_group}'
        treatment_mean_col = f'mean-{treatment_group}'
        
        if control_mean_col in results.columns and treatment_mean_col in results.columns:
            results['log2_foldchange'] = np.log2(
                (1 + results[treatment_mean_col]) / (1 + results[control_mean_col])
            )
        else:
            results['log2_foldchange'] = 0
            
        # Add metadata
        results['peak_values_normalized'] = self.config.normalize_data
        results['peak_value_used'] = self.config.peak_value
        results['control_group'] = control_group
        results['treatment_group'] = treatment_group
        
        return results
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data by total signal per sample."""
        sample_totals = data.sum(axis=1)
        normalized_data = data.div(sample_totals, axis=0)
        return normalized_data * sample_totals.mean()
    
    def _calculate_group_statistics(self, 
                                  data: pd.DataFrame,
                                  control_group: str,
                                  treatment_group: str) -> pd.DataFrame:
        """Calculate group-wise statistics."""
        stats = data.groupby(['original_index', 'sample_category'])[self.config.peak_value].agg([
            'mean', 'median', 'std', lambda x: x.sem()
        ]).reset_index()
        
        stats.columns = ['original_index', 'sample_category', 'mean', 'median', 'std_dev', 'standard_error']
        
        # Pivot to get groups as columns
        stats_pivot = stats.pivot_table(
            index='original_index',
            values=['mean', 'median', 'std_dev', 'standard_error'],
            columns='sample_category'
        )
        
        # Flatten column names
        stats_pivot.columns = [f'{stat}-{group}' for stat, group in stats_pivot.columns]
        
        return stats_pivot
    
    def _calculate_ttests(self, 
                         data: pd.DataFrame,
                         control_group: str,
                         treatment_group: str) -> pd.DataFrame:
        """Calculate t-tests for each compound."""
        # Pivot data for t-tests
        test_data = data.pivot_table(
            columns='original_index',
            index=['lcmsrun_observed', 'sample_category'],
            values=self.config.peak_value,
            aggfunc='mean',
            fill_value=0
        )
        
        # Initialize results
        results = pd.DataFrame(index=test_data.columns)
        results['p_value'] = 1.0
        results['t_score'] = 0.0
        
        # Get group indices
        control_idx = test_data.index.get_level_values('sample_category') == control_group
        treatment_idx = test_data.index.get_level_values('sample_category') == treatment_group
        
        # Perform t-tests
        for compound in test_data.columns:
            control_vals = test_data.loc[control_idx, compound].values
            treatment_vals = test_data.loc[treatment_idx, compound].values
            
            if len(control_vals) > 1 and len(treatment_vals) > 1:
                try:
                    t_score, p_val = ttest_ind(control_vals, treatment_vals)
                    results.loc[compound, 'p_value'] = p_val
                    results.loc[compound, 't_score'] = t_score
                except:
                    # Handle cases where t-test fails (e.g., identical values)
                    pass
        
        return results


def calculate_binomial_test(positive_count: int, total_count: int) -> float:
    """Calculate binomial test p-value."""
    test_result = binomtest(positive_count, total_count, 0.5)
    return test_result.pvalue


def aggregate_fold_changes(fold_changes: pd.Series) -> Dict:
    """Aggregate fold change statistics for compound classes."""
    positive_count = (fold_changes > 0).sum()
    negative_count = (fold_changes < 0).sum()
    
    return {
        'mean': fold_changes.mean(),
        'stderror': fold_changes.std(),
        'count': fold_changes.count(),
        'positive_count': positive_count,
        'negative_count': negative_count
    }