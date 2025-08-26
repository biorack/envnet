"""
Multivariate analysis functions including PCA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

from ..config.analysis_config import AnalysisConfig


class MultivariateAnalyzer:
    """Performs multivariate analysis on ENVnet data."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def run_pca(self, 
                data: pd.DataFrame,
                grouping_column: str = 'specific_environment',
                output_file: Optional[str] = None) -> Dict:
        """
        Run PCA analysis and create visualization.
        
        Args:
            data: Analysis data
            grouping_column: Column to group/color samples by
            output_file: Optional file to save plot
            
        Returns:
            Dict with PCA results
        """
        # Create pivot table
        pivot_data = data.pivot_table(
            values=self.config.peak_value,
            index=['lcmsrun_observed', grouping_column],
            columns='original_index',
            aggfunc='mean'
        )
        
        # Handle missing values
        min_value = np.nanmin(pivot_data.values)
        pivot_data.fillna(min_value * 2/3, inplace=True)
        
        # Transform data
        X = pivot_data.values
        if self.config.log_transform:
            X = np.log2(X + 1)
        
        # Perform PCA
        pca = PCA(n_components=self.config.n_components)
        X_pca = pca.fit_transform(X)
        
        # Create visualization
        fig = self._create_pca_plot(X_pca, pivot_data, grouping_column)
        
        if output_file:
            fig.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        # Return results
        results = {
            'pca_coordinates': X_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_,
            'sample_groups': pivot_data.index.get_level_values(grouping_column),
            'figure': fig if not output_file else None
        }
        
        return results
    
    def _create_pca_plot(self, X_pca: np.ndarray, pivot_data: pd.DataFrame, 
                        grouping_column: str) -> plt.Figure:
        """Create PCA visualization."""
        # Set up plotting parameters
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.font_size + 2,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size,
            'ytick.labelsize': self.config.font_size,
            'legend.fontsize': self.config.font_size,
        })
        
        # Get groups and colors
        groups = pivot_data.index.get_level_values(grouping_column)
        unique_groups = groups.unique()
        
        # Create color mapping
        colors = dict(zip(unique_groups, plt.cm.tab20.colors[:len(unique_groups)]))
        color_values = groups.map(colors)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Special handling for groundwater if present
        if 'groundwater' in unique_groups:
            groundwater_idx = groups == 'groundwater'
            other_idx = ~groundwater_idx
            
            # Plot groundwater with special formatting
            ax.scatter(X_pca[groundwater_idx, 0], X_pca[groundwater_idx, 1],
                      c=color_values[groundwater_idx], alpha=1, s=70, 
                      edgecolors='k', linewidth=1)
            
            # Plot other environments
            ax.scatter(X_pca[other_idx, 0], X_pca[other_idx, 1],
                      c=color_values[other_idx], alpha=0.8, s=30)
            
            # Create legend entries
            for group in unique_groups:
                if group == 'groundwater':
                    ax.scatter([], [], label='ENIGMA groundwater', alpha=1, s=70,
                              color=colors[group], edgecolors='k', linewidth=1)
                else:
                    ax.scatter([], [], label=group, alpha=0.8, s=30,
                              color=colors[group])
        else:
            # Standard plotting
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_values, alpha=1, s=30)
            
            # Create legend
            for group in unique_groups:
                ax.scatter([], [], label=group, alpha=1, s=30, color=colors[group])
        
        # Formatting
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', frameon=False)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig