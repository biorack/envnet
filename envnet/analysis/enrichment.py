"""
Compound class enrichment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest
from typing import Dict, Optional, List
import os

from ..config.analysis_config import AnalysisConfig
from .statistics import calculate_binomial_test, aggregate_fold_changes


class EnrichmentAnalyzer:
    """Performs compound class enrichment analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def run_enrichment_analysis(self, 
                               stats_results: pd.DataFrame,
                               envnet_data: Optional[pd.DataFrame] = None,
                               output_dir: Optional[str] = None) -> Dict:
        """
        Run compound class enrichment analysis.
        
        Args:
            stats_results: Statistical analysis results
            envnet_data: ENVnet node data with compound classifications
            output_dir: Optional directory to save results
            
        Returns:
            Dict with enrichment results
        """
        if envnet_data is None:
            print("Warning: No ENVnet data provided, skipping enrichment analysis")
            return {}
            
        # Merge statistical results with compound classifications
        merged_data = self._merge_with_classifications(stats_results, envnet_data)

        # Define class columns to analyze
        class_columns = self._get_class_columns(merged_data)
        
        results = {}
        
        if output_dir:
            pdf_file = os.path.join(output_dir, 'compound_class_enrichment.pdf')
            with PdfPages(pdf_file) as pdf:
                for class_col in class_columns:
                    enrichment_result = self._analyze_class_enrichment(
                        merged_data, class_col, output_dir, pdf
                    )
                    if enrichment_result is not None:
                        results[class_col] = enrichment_result
        else:
            for class_col in class_columns:
                enrichment_result = self._analyze_class_enrichment(merged_data, class_col)
                if enrichment_result is not None:
                    results[class_col] = enrichment_result
        
        return results
    
    def _merge_with_classifications(self, 
                                   stats_results: pd.DataFrame,
                                   envnet_data: pd.DataFrame) -> pd.DataFrame:
        """Merge statistical results with ENVnet compound classifications."""
        # Ensure original_index is consistent
        stats_results = stats_results.copy()
        stats_results.reset_index(inplace=True)
        if 'original_index' not in stats_results.columns:
            stats_results['original_index'] = stats_results.index
            
        # Select relevant columns from ENVnet data
        class_columns = [col for col in envnet_data.columns 
                        if any(x in col.lower() for x in ['class', 'pathway', 'superclass'])]
        merge_columns = ['original_index'] + class_columns
        
        merged = pd.merge(
            stats_results,
            envnet_data[merge_columns],
            on='original_index',
            how='left'
        )
        
        return merged
    
    def _get_class_columns(self, data: pd.DataFrame) -> List[str]:
        """Get available compound class columns."""
        class_patterns = [
            "NPC#pathway",
            "NPC#superclass",
            "NPC#class"
        ]
        
        available_columns = []
        for pattern in class_patterns:
            matching_cols = [col for col in data.columns if pattern == col]
            available_columns.extend(matching_cols)
        
        return available_columns
    
    def _analyze_class_enrichment(self, 
                                 data: pd.DataFrame,
                                 class_column: str,
                                 output_dir: Optional[str] = None,
                                 pdf: Optional[PdfPages] = None) -> Optional[pd.DataFrame]:
        """Analyze enrichment for a specific compound class column."""
        # Filter significant results
        if 'p_value' not in data.columns or data['p_value'].isnull().all():
            print(f"No p-values found for {class_column}")
            return None
            
        significant_data = data[
            (pd.notna(data[class_column])) & 
            (data['p_value'] < self.config.max_pvalue)
        ]

        if len(significant_data) == 0:
            print(f"No significant results for {class_column}")
            return None
        
        # Group by compound class and aggregate fold changes
        class_groups = significant_data.groupby(class_column)['log2_foldchange'].apply(
            aggregate_fold_changes
        )

        # Convert to DataFrame
        enrichment_results = class_groups.unstack().reset_index().astype({'count': int, 'positive_count': int, 'negative_count': int})
        
        # Filter for classes with at least one significant compound
        enrichment_results = enrichment_results[enrichment_results['positive_count'] >= 1]
        
        if len(enrichment_results) == 0:
            return None
        
        # Calculate binomial test p-values
        enrichment_results['binomial_pvalue'] = enrichment_results.apply(
            lambda x: calculate_binomial_test(x['positive_count'], x['count']), 
            axis=1
        )
        
        # Filter by binomial p-value
        enrichment_results = enrichment_results[
            enrichment_results['binomial_pvalue'] <= self.config.max_pvalue
        ]
        
        if len(enrichment_results) == 0:
            return None
        
        # Sort by mean fold change
        enrichment_results.sort_values('mean', ascending=False, inplace=True)
        
        # Save results
        if output_dir:
            results_file = os.path.join(output_dir, f'{class_column}_enrichment.csv')
            enrichment_results.to_csv(results_file)
        
        # Create visualization
        self._create_enrichment_plot(
            enrichment_results, class_column, output_dir, pdf
        )
        
        return enrichment_results
    
    def _create_enrichment_plot(self, 
                               results: pd.DataFrame,
                               class_column: str,
                               output_dir: Optional[str] = None,
                               pdf: Optional[PdfPages] = None) -> None:
        """Create enrichment bar plot."""
        # Calculate figure height
        height = max(3, results.shape[0] * 0.5)
        
        fig, ax = plt.subplots(figsize=(10, height))
        
        # Create horizontal bar plot
        y_pos = range(len(results))
        bars = ax.barh(y_pos, results['mean'], xerr=results['stderror'])
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results[class_column])
        ax.set_xlabel('Log2 Fold Change')
        ax.set_title(f'Compound Class Enrichment: {class_column}')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save or show plot
        if pdf:
            pdf.savefig(bbox_inches='tight')
        elif output_dir:
            plot_file = os.path.join(output_dir, f'{class_column}_enrichment.png')
            plt.savefig(plot_file, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()