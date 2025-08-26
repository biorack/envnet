"""
Visualization tools for ENVnet molecular networks.

This module provides functions for creating publication-quality visualizations
of molecular networks, including compound class distributions and network layouts.


you use it like this:

from envnet.build.visualization import NetworkVisualizer, plot_network_classes

# Simple usage
fig = plot_network_classes("envnet.cyjs", network, "my_classes.pdf")

# Full report
visualizer = NetworkVisualizer()
files = visualizer.create_network_visualization_report("envnet.cyjs", network)
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config.build_config import BuildConfig


class NetworkVisualizer:
    """Handles visualization of molecular networks and compound class distributions."""
    
    def __init__(self, config: Optional[BuildConfig] = None):
        """Initialize visualizer with configuration."""
        self.config = config or BuildConfig()
        
    def load_cytoscape_network_data(self, cyjs_file: str, network: nx.Graph) -> pd.DataFrame:
        """
        Load network node data from Cytoscape .cyjs file.
        
        Args:
            cyjs_file: Path to Cytoscape .cyjs file
            network: NetworkX graph to filter nodes
            
        Returns:
            DataFrame with node positions and attributes
        """
        with open(cyjs_file, 'r') as f:
            network_data = json.load(f)
        
        # Extract and normalize node data
        nodes_df = pd.DataFrame(network_data['elements']['nodes'])
        data_df = pd.json_normalize(nodes_df['data'])
        position_df = pd.json_normalize(nodes_df['position'])
        
        # Combine data and positions
        network_data = pd.concat([data_df, position_df], axis=1)
        
        # Select and rename columns
        available_cols = network_data.columns.tolist()
        desired_cols = ['name', 'x', 'y', 'precursor_mz', 'n', 'smiles', 
                       'NPC#pathway', 'NPC#superclass', 'NPC#class']
        
        # Use only columns that exist
        cols = [c for c in desired_cols if c in available_cols]
        network_data = network_data[cols]
        
        # Rename columns
        rename_map = {
            'name': 'original_index',
            'n': 'N',
            'NPC#pathway': 'NPC_pathway',
            'NPC#superclass': 'NPC_superclass', 
            'NPC#class': 'NPC_class'
        }
        network_data.rename(columns=rename_map, inplace=True)
        
        # Convert types and filter
        network_data['original_index'] = network_data['original_index'].astype(int)
        network_data = network_data[network_data['original_index'].isin(network.nodes())]
        
        print(f"Loaded network visualization data for {len(network_data)} nodes")
        return network_data
    
    def make_plot_title(self, title: str, max_length: int = 20) -> str:
        """
        Wrap title text for better display in subplots.
        
        Args:
            title: Original title text
            max_length: Maximum characters per line
            
        Returns:
            Wrapped title string with newlines
        """
        words = title.split()
        wrapped_title = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += " " + word if current_line else word
            else:
                wrapped_title.append(current_line)
                current_line = word

        if current_line:
            wrapped_title.append(current_line)
        
        out = '\n'.join(wrapped_title)
        if len(out) < 20:
            out = out.replace(' ', '\n')
        
        # Limit to 2 lines max
        out = out.split('\n')
        if len(out) >= 3:
            out = '\n'.join(out[:2])
            out = '%s...' % out
        else:
            out = '\n'.join(out)
        
        return out.strip()
    
    def plot_compound_class_distribution(self, network_data: pd.DataFrame, 
                                       class_column: str = 'NPC_superclass',
                                       min_class_size: int = 15,
                                       ncols: int = 8, 
                                       nrows: int = 6,
                                       figsize: Optional[Tuple[int, int]] = None,
                                       output_file: Optional[str] = None) -> plt.Figure:
        """
        Create grid plot showing spatial distribution of compound classes.
        
        Args:
            network_data: DataFrame with node positions and class information
            class_column: Column name for compound classification
            min_class_size: Minimum number of nodes to include a class
            ncols: Number of columns in subplot grid
            nrows: Number of rows in subplot grid
            figsize: Figure size tuple (will calculate if None)
            output_file: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get classes with sufficient representation
        my_classes = network_data[class_column].value_counts()
        my_classes = my_classes[my_classes >= min_class_size]
        
        # Set up figure
        if figsize is None:
            figsize = (int(ncols * 2), int(nrows * 3.3))
        
        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
        ax = ax.flatten()
        
        print(f"Creating compound class distribution plot for {len(my_classes)} classes")
        
        counter = 0
        for my_term in my_classes.index.tolist():
            if counter >= len(ax):
                break
                
            # Create class mask
            idx = network_data[class_column] == my_term
            
            # Plot background (all other nodes)
            ax[counter].scatter(
                network_data.loc[~idx, 'x'], 
                network_data.loc[~idx, 'y'], 
                s=2, c='grey', alpha=0.6
            )
            
            # Plot class nodes
            ax[counter].scatter(
                network_data.loc[idx, 'x'], 
                network_data.loc[idx, 'y'], 
                s=1, c='red'
            )
            
            # Format subplot
            wrapped_title = self.make_plot_title(my_term, max_length=18)
            ax[counter].set_title(wrapped_title, fontsize=16)
            ax[counter].set_facecolor('black')
            ax[counter].invert_yaxis()
            ax[counter].set_xticks([])
            ax[counter].set_yticks([])
            ax[counter].set_aspect('equal', adjustable='box')
            
            counter += 1
        
        # Hide unused subplots
        for i in range(counter, len(ax)):
            ax[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved compound class distribution plot to: {output_file}")
        
        return fig
    
    def plot_network_overview(self, network_data: pd.DataFrame, 
                            color_column: Optional[str] = None,
                            size_column: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 10),
                            output_file: Optional[str] = None) -> plt.Figure:
        """
        Create overview plot of the entire network.
        
        Args:
            network_data: DataFrame with node positions and attributes
            color_column: Column to use for node coloring
            size_column: Column to use for node sizing
            figsize: Figure size tuple
            output_file: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine colors
        if color_column and color_column in network_data.columns:
            # Create color mapping for categorical data
            unique_vals = network_data[color_column].dropna().unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
            color_map = dict(zip(unique_vals, colors))
            node_colors = network_data[color_column].map(color_map).fillna('grey')
        else:
            node_colors = 'blue'
        
        # Determine sizes
        if size_column and size_column in network_data.columns:
            # Normalize sizes
            sizes = network_data[size_column].fillna(1)
            sizes = 10 + 50 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            sizes = 20
        
        # Create scatter plot
        scatter = ax.scatter(
            network_data['x'], 
            network_data['y'],
            c=node_colors,
            s=sizes,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Format plot
        ax.set_facecolor('black')
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('ENVnet Molecular Network Overview', fontsize=16, color='white')
        
        # Add legend if using colors
        if color_column and color_column in network_data.columns:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[val], 
                          markersize=8, label=str(val)[:20]) 
                for val in unique_vals[:20]  # Limit legend entries
            ]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Saved network overview plot to: {output_file}")
        
        return fig
    
    def plot_class_statistics(self, network_data: pd.DataFrame,
                            class_column: str = 'NPC_superclass',
                            figsize: Tuple[int, int] = (10, 6),
                            output_file: Optional[str] = None) -> plt.Figure:
        """
        Create bar plot showing compound class frequencies.
        
        Args:
            network_data: DataFrame with node class information
            class_column: Column name for compound classification
            figsize: Figure size tuple
            output_file: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get class counts
        class_counts = network_data[class_column].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        bars = ax.bar(range(len(class_counts)), class_counts.values)
        
        # Format plot
        ax.set_xlabel('Compound Class')
        ax.set_ylabel('Number of Nodes')
        ax.set_title(f'Distribution of {class_column} in Network')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels([self.make_plot_title(label, 10) for label in class_counts.index], 
                          rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved class statistics plot to: {output_file}")
        
        return fig
    
    def create_network_visualization_report(self, cyjs_file: str, network: nx.Graph,
                                          output_dir: str = "visualizations") -> Dict[str, str]:
        """
        Create complete visualization report for a molecular network.
        
        Args:
            cyjs_file: Path to Cytoscape .cyjs file
            network: NetworkX graph
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load network data
        network_data = self.load_cytoscape_network_data(cyjs_file, network)
        
        saved_files = {}
        
        # 1. Compound class distribution plot
        if 'NPC_superclass' in network_data.columns:
            class_dist_file = output_path / "compound_class_distribution.pdf"
            self.plot_compound_class_distribution(
                network_data, 
                class_column='NPC_superclass',
                output_file=str(class_dist_file)
            )
            saved_files['class_distribution'] = str(class_dist_file)
        
        # 2. Network overview plot
        overview_file = output_path / "network_overview.pdf"
        color_col = 'NPC_superclass' if 'NPC_superclass' in network_data.columns else None
        self.plot_network_overview(
            network_data,
            color_column=color_col,
            output_file=str(overview_file)
        )
        saved_files['network_overview'] = str(overview_file)
        
        # 3. Class statistics plot
        if 'NPC_superclass' in network_data.columns:
            stats_file = output_path / "class_statistics.pdf"
            self.plot_class_statistics(
                network_data,
                class_column='NPC_superclass',
                output_file=str(stats_file)
            )
            saved_files['class_statistics'] = str(stats_file)
        
        print(f"Created network visualization report in: {output_dir}")
        return saved_files


# Convenience functions
def plot_network_classes(cyjs_file: str, network: nx.Graph, 
                        output_file: str = "compound_classes.pdf",
                        class_column: str = 'NPC_superclass') -> plt.Figure:
    """
    Convenience function to create compound class distribution plot.
    
    Args:
        cyjs_file: Path to Cytoscape .cyjs file
        network: NetworkX graph
        output_file: Output file path
        class_column: Column for compound classification
        
    Returns:
        Matplotlib figure object
    """
    visualizer = NetworkVisualizer()
    network_data = visualizer.load_cytoscape_network_data(cyjs_file, network)
    return visualizer.plot_compound_class_distribution(
        network_data, 
        class_column=class_column,
        output_file=output_file
    )


def create_visualization_report(cyjs_file: str, network: nx.Graph, 
                              output_dir: str = "visualizations") -> Dict[str, str]:
    """
    Convenience function to create complete visualization report.
    
    Args:
        cyjs_file: Path to Cytoscape .cyjs file
        network: NetworkX graph
        output_dir: Output directory
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    visualizer = NetworkVisualizer()
    return visualizer.create_network_visualization_report(cyjs_file, network, output_dir)