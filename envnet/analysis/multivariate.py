"""
Multivariate analysis functions including PCA and t-SNE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
                output_file: Optional[str] = None,
                highlight_group: Optional[str] = 'groundwater',
                method: str = 'pca') -> Dict:
        """
        Run PCA or t-SNE analysis and create visualization.
        
        Args:
            data: Analysis data
            grouping_column: Column to group/color samples by
            output_file: Optional file to save plot
            highlight_group: Optional group name to highlight in the plot. Set to None to disable.
            method: Method to use for dimensionality reduction ('pca' or 'tsne')
            
        Returns:
            Dict with analysis results
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
        
        # Standardize data for t-SNE (optional for PCA but good practice)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform dimensionality reduction
        if method.lower() == 'tsne':
            # Use t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
            X_reduced = reducer.fit_transform(X_scaled)
            explained_variance = None  # t-SNE doesn't provide explained variance
            components = None  # t-SNE doesn't provide components
        elif method.lower() == 'pca':
            # Use PCA
            reducer = PCA(n_components=self.config.n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            explained_variance = reducer.explained_variance_ratio_
            components = reducer.components_
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'pca' or 'tsne'.")
        
        # Create visualization
        # MODIFIED: Pass method and highlight_group to the plotting function
        fig = self._create_plot(X_reduced, pivot_data, grouping_column, highlight_group, method)
        
        if output_file:
            fig.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        # Return results
        results = {
            f'{method.lower()}_coordinates': X_reduced,
            'explained_variance': explained_variance,
            'components': components,
            'sample_groups': pivot_data.index.get_level_values(grouping_column),
            'figure': fig if not output_file else None,
            'method': method.lower()
        }
        
        return results
    
    def _create_plot(self, X_reduced: np.ndarray, pivot_data: pd.DataFrame, 
                        grouping_column: str, highlight_group: Optional[str], method: str) -> plt.Figure: # MODIFIED: Generalized function name and added method parameter
        """Create visualization for PCA or t-SNE."""
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
        
        # MODIFIED: Logic now depends on the highlight_group parameter instead of a hardcoded string
        if highlight_group and highlight_group in unique_groups:
            highlight_idx = groups == highlight_group
            other_idx = ~highlight_idx
            
            # Plot highlighted group with special formatting
            ax.scatter(X_reduced[highlight_idx, 0], X_reduced[highlight_idx, 1],
                      c=color_values[highlight_idx], alpha=1, s=70, 
                      edgecolors='k', linewidth=1)
            
            # Plot other environments
            ax.scatter(X_reduced[other_idx, 0], X_reduced[other_idx, 1],
                      c=color_values[other_idx], alpha=0.8, s=30)
            
            # Create legend entries
            for group in unique_groups:
                if group == highlight_group:
                    # You can customize this label if needed
                    ax.scatter([], [], label=f'{group} (highlighted)', alpha=1, s=70,
                              color=colors[group], edgecolors='k', linewidth=1)
                else:
                    ax.scatter([], [], label=group, alpha=0.8, s=30,
                              color=colors[group])
        else:
            # Standard plotting for all groups
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color_values, alpha=1, s=30)
            
            # Create legend
            for group in unique_groups:
                ax.scatter([], [], label=group, alpha=1, s=30, color=colors[group])
        
        # Formatting with method-specific labels
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', frameon=False)
        if method.lower() == 'tsne':
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
        else:  # PCA
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig