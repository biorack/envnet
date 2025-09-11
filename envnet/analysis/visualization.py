"""
Visualization functions for ENVnet analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from upsetplot import UpSet
from typing import Dict, Optional, List
import os

from ..config.analysis_config import AnalysisConfig


class AnalysisVisualizer:
    """Creates visualizations for ENVnet analysis results."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def create_upset_plot(self, 
                         data: pd.DataFrame,
                         grouping_column: str = 'specific_environment',
                         output_file: Optional[str] = None,
                         fig_width: float=10,
                         fig_height: float=6) -> None:
        """
        Create UpSet plot showing compound overlap across groups.
        
        Args:
            data: Analysis data
            grouping_column: Column to group by
            output_file: Optional file to save plot
        """
        # Set font parameters
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.font_size + 2,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size,
            'ytick.labelsize': self.config.font_size,
            'legend.fontsize': self.config.font_size,
        })
        
        # Group compounds by environment
        grouped_compounds = data.groupby(grouping_column)['original_index'].unique()
        compound_dict = grouped_compounds.to_dict()
        
        # Convert to UpSet format
        upset_data = self._prepare_upset_data(compound_dict)
        
        # Create UpSet plot
        fig = plt.figure(figsize=(fig_width, fig_height))
        upset = UpSet(upset_data, min_subset_size=self.config.min_upset_subset_size)
        upset.plot(fig=fig)

        if output_file:
            plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return fig
    
    def _prepare_upset_data(self, compound_dict: Dict) -> pd.Series:
        """Prepare data for UpSet plot."""
        # Convert dictionary to list of tuples
        data_tuples = [
            (env, compound) 
            for env, compounds in compound_dict.items() 
            for compound in compounds
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data_tuples, columns=['Environment', 'Compound'])
        
        # Pivot to binary matrix
        binary_df = df.groupby('Compound')['Environment'].apply(
            lambda x: pd.Series(1, x)
        ).unstack().fillna(0).astype(bool)
        
        # Group by columns and count
        columns = binary_df.columns.tolist()
        grouped_counts = binary_df.groupby(columns).size()
        
        return grouped_counts
    
    def create_set_cover_plots(self, 
                              data: pd.DataFrame,
                              output_dir: Optional[str] = None) -> Dict:
        """
        Create set cover analysis plots for MS1 and MS2 data.
        
        Args:
            data: Analysis data
            output_dir: Optional directory to save plots
            
        Returns:
            Dict with set cover results
        """
        results = {}
        
        if output_dir:
            output_dir = os.path.abspath(output_dir)
            pdf_file = os.path.join(output_dir, 'set_cover_results.pdf')
            
            with PdfPages(pdf_file) as pdf:
                # MS1 set cover
                results['ms1'] = self._create_single_set_cover_plot(
                    data, 'ms1', output_dir, pdf
                )
                
                # MS2 set cover (if MS2 score data available)
                if 'ms2_score' in data.columns:
                    ms2_data = data[data['ms2_score'].notna()]
                    results['ms2'] = self._create_single_set_cover_plot(
                        ms2_data, 'ms2', output_dir, pdf
                    )
        else:
            results['ms1'] = self._create_single_set_cover_plot(data, 'ms1')
            
        return results
    
    def _create_single_set_cover_plot(self, 
                                     data: pd.DataFrame,
                                     data_type: str,
                                     output_dir: Optional[str] = None,
                                     pdf: Optional[PdfPages] = None) -> Dict:
        """Create a single set cover plot."""
        # Group compounds by sample
        sample_compounds = data.groupby('lcmsrun_observed')['original_index'].unique()
        compound_dict = {sample: set(compounds) for sample, compounds in sample_compounds.items()}
        
        # Apply set cover algorithm
        selected_samples = self._set_cover_algorithm(compound_dict)
        
        # Calculate coverage
        combined_compounds = set()
        cumulative_counts = []
        individual_counts = []
        
        for sample in selected_samples:
            if sample in compound_dict:
                combined_compounds.update(compound_dict[sample])
                cumulative_counts.append(len(combined_compounds))
                individual_counts.append(len(compound_dict[sample]))
        
        # Create plot
        n_samples = len(selected_samples)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Bar plot for individual counts
        ax.bar(range(n_samples), individual_counts, color='orange', alpha=0.5, 
               label='Individual')
        
        # Line plot for cumulative counts
        ax.plot(range(n_samples), cumulative_counts, '.-', markersize=18, 
                label='Cumulative')
        
        # Formatting
        ax.set_xlabel('Cumulative Addition of Environments')
        ax.set_ylabel('Number of Compounds')
        ax.set_xticks(range(n_samples))
        ax.set_xticklabels(selected_samples, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_title(f'{data_type.upper()} Set Cover Analysis')
        
        plt.tight_layout()
        
        # Save or show plot
        if pdf:
            pdf.savefig(bbox_inches='tight')
        elif output_dir:
            plot_file = os.path.join(output_dir, f'{data_type}_set_cover.png')
            plt.savefig(plot_file, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        return {
            'selected_samples': selected_samples,
            'cumulative_coverage': cumulative_counts,
            'individual_coverage': individual_counts,
            'total_compounds': len(combined_compounds)
        }
    
    def _set_cover_algorithm(self, coverage_dict: Dict) -> List[str]:
        """Implement set cover algorithm."""
        coverage_dict = coverage_dict.copy()
        all_compounds = set()
        for compounds in coverage_dict.values():
            all_compounds.update(compounds)
        
        selected_samples = []
        
        while all_compounds and coverage_dict:
            # Find sample that covers most uncovered compounds
            best_sample = None
            best_covered = set()
            
            for sample, compounds in coverage_dict.items():
                covered = compounds.intersection(all_compounds)
                if len(covered) > len(best_covered):
                    best_sample = sample
                    best_covered = covered
            
            if best_sample is None:
                break
                
            # Add best sample and remove covered compounds
            selected_samples.append(best_sample)
            all_compounds.difference_update(best_covered)
            del coverage_dict[best_sample]
        
        return selected_samples