"""
Integration of SIRIUS formula predictions and CANOPUS compound classifications.

This module handles loading SIRIUS results and integrating them back into
the molecular network for visualization and analysis.

You can use it like this
from .integration import SiriusIntegrator

integrator = SiriusIntegrator(config)
network = integrator.integrate_predictions(network, "path/to/sirius/results")

"""

import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ..config.build_config import BuildConfig


class SiriusIntegrator:
    """Handles integration of SIRIUS formula and compound class predictions."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def load_sirius_formula_results(self, sirius_dir: str) -> pd.DataFrame:
        """
        Load SIRIUS formula identification results.
        
        Args:
            sirius_dir: Directory containing SIRIUS results
            
        Returns:
            DataFrame with formula predictions (rank 1 only)
        """
        formula_file = Path(sirius_dir) / "formula_identifications.tsv"
        
        if not formula_file.exists():
            raise FileNotFoundError(f"SIRIUS formula file not found: {formula_file}")
        
        sirius_formula = pd.read_csv(formula_file, sep='\t')
        
        # Keep only rank 1 predictions
        sirius_formula = sirius_formula[sirius_formula['formulaRank'] == 1]
        
        print(f"Loaded {len(sirius_formula)} formula predictions from SIRIUS")
        return sirius_formula
    
    def load_sirius_canopus_results(self, sirius_dir: str) -> pd.DataFrame:
        """
        Load SIRIUS CANOPUS compound class predictions.
        
        Args:
            sirius_dir: Directory containing SIRIUS results
            
        Returns:
            DataFrame with compound class predictions (rank 1 only)
        """
        canopus_file = Path(sirius_dir) / "canopus_formula_summary.tsv"
        
        if not canopus_file.exists():
            raise FileNotFoundError(f"SIRIUS CANOPUS file not found: {canopus_file}")
        
        sirius_class = pd.read_csv(canopus_file, sep='\t')
        
        # Keep only rank 1 predictions
        sirius_class = sirius_class[sirius_class['formulaRank'] == 1]
        
        # Extract NPC columns and mapping feature ID
        npc_cols = [c for c in sirius_class.columns if c.startswith('NPC')] + ['mappingFeatureId']
        sirius_class = sirius_class[npc_cols]
        
        # Ensure mappingFeatureId is integer
        sirius_class['mappingFeatureId'] = sirius_class['mappingFeatureId'].astype(float).astype(int)
        
        print(f"Loaded {len(sirius_class)} compound class predictions from SIRIUS")
        return sirius_class
    
    def create_compound_class_colors(self, sirius_class: pd.DataFrame, 
                                   class_column: str = 'NPC#class', 
                                   top_n: int = 10) -> Dict[str, str]:
        """
        Create color mapping for top compound classes.
        
        Args:
            sirius_class: DataFrame with compound class predictions
            class_column: Column to use for class assignment
            top_n: Number of top classes to assign colors
            
        Returns:
            Dictionary mapping class names to hex colors
        """
        # Get top classes by frequency
        top_classes = sirius_class[class_column].value_counts().index[:top_n]
        
        # Use matplotlib's tab10 colormap
        colors = cm.tab10.colors[:top_n]
        hex_colors = [mcolors.to_hex(c) for c in colors]
        
        # Create mapping
        class_color_mapping = dict(zip(top_classes, hex_colors))
        
        print(f"Created color mapping for {len(class_color_mapping)} compound classes")
        return class_color_mapping
    
    def _save_class_legend_figure(self, class_colors: Dict[str, str], output_dir: str):
        """
        Generates and saves a legend figure as a high-quality PDF.

        Args:
            class_colors: Dictionary mapping class names to hex color codes.
            output_dir: The directory where the legend image will be saved.
        """
        if not class_colors:
            print("No class colors provided; skipping legend generation.")
            return

        # Store original matplotlib settings
        original_rc_params = plt.rcParams.copy()
        
        try:
            # Set parameters for publication-quality PDF output
            plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded as TrueType
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['font.family'] = 'sans-serif'

            fig, ax = plt.subplots(figsize=(8, 6))
            for class_name, color_hex in class_colors.items():
                ax.scatter([], [], color=color_hex, label=class_name, s=100)
            
            ax.legend(title="SIRIUS Compound Classes", loc="center", frameon=False, fontsize='large', title_fontsize='x-large')
            ax.axis('off')
            
            # Save the figure as a PDF
            legend_filename = Path(output_dir) / "sirius_class_legend.pdf"
            plt.savefig(legend_filename, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved compound class legend to: {legend_filename}")
        
        except Exception as e:
            print(f"Warning: Could not generate or save legend figure. Error: {e}")
        
        finally:
            # Restore original matplotlib settings to avoid side effects
            plt.rcParams.update(original_rc_params)

    def integrate_predictions(self, network: nx.Graph, sirius_dir: str,
                            class_column: str = 'NPC#class') -> nx.Graph:
        """
        Integrate SIRIUS predictions into NetworkX graph as node attributes.
        
        Args:
            network: NetworkX graph to add attributes to
            sirius_dir: Directory containing SIRIUS results
            class_column: NPC column to use for primary classification
            
        Returns:
            Updated NetworkX graph with SIRIUS attributes
        """
        # Load SIRIUS results
        formula_results = self.load_sirius_formula_results(sirius_dir)
        class_results = self.load_sirius_canopus_results(sirius_dir)
        
        # Create color mapping for compound classes
        class_colors = self.create_compound_class_colors(class_results, class_column)
        self._save_class_legend_figure(class_colors, sirius_dir)

        # Add color information to class results
        class_results['color_compound_class'] = class_results[class_column].map(class_colors)
        class_results['color_compound_class'] = class_results['color_compound_class'].fillna('#FFFFFF')  # White for unmapped
        
        # Prepare data for network integration
        integration_cols = ['mappingFeatureId', 'NPC#pathway', 'NPC#superclass', 'NPC#class', 'color_compound_class']
        class_data = class_results[integration_cols].copy()
        class_data.fillna('', inplace=True)
        class_data.set_index('mappingFeatureId', inplace=True)
        class_data.index.name = 'node_id'
        
        # Convert to dictionary for NetworkX
        class_dict = class_data.to_dict(orient='index')
        class_dict = {int(float(k)): v for k, v in class_dict.items()}
        
        # Ensure network node IDs are integers
        network = nx.relabel_nodes(network, {k: int(float(k)) for k in network.nodes()})
        
        # Set node attributes
        nx.set_node_attributes(network, class_dict)
        
        # Report integration statistics
        nodes_with_class = len([n for n in network.nodes() if n in class_dict])
        print(f"Integrated SIRIUS predictions for {nodes_with_class}/{network.number_of_nodes()} network nodes")
        
        return network
    
    def get_np_classifier_predictions(self, df: pd.DataFrame, 
                                    smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Get NP Classifier predictions for library matches using SMILES.
        
        Args:
            df: DataFrame containing SMILES strings
            smiles_column: Column name containing SMILES
            
        Returns:
            DataFrame with NP Classifier predictions
        """
        url = "https://npclassifier.gnps2.org/classify"
        
        # Filter to rows with valid SMILES
        valid_smiles = df[pd.notna(df[smiles_column]) & (df[smiles_column] != '')].copy()
        
        predictions = []
        
        for i, row in valid_smiles.iterrows():
            try:
                r = requests.get(url, params={"smiles": row[smiles_column]}, timeout=10)
                
                if r.status_code == 200:
                    result = r.json()
                else:
                    result = {
                        'class_results': None,
                        'superclass_results': None,
                        'pathway_results': None,
                        'isglycoside': False
                    }
                    
            except Exception as e:
                print(f"Error predicting for SMILES {row[smiles_column]}: {e}")
                result = {
                    'class_results': None,
                    'superclass_results': None,
                    'pathway_results': None,
                    'isglycoside': False
                }
            
            # Add identifier
            if 'inchi_key' in row:
                result['inchi_key'] = row['inchi_key']
            result['row_index'] = i
            
            predictions.append(result)
        
        # Convert to DataFrame and clean up
        np_results = pd.DataFrame(predictions)
        
        # Convert lists to comma-separated strings
        list_cols = ['class_results', 'superclass_results', 'pathway_results']
        for col in list_cols:
            np_results[col] = np_results[col].apply(
                lambda x: ','.join(sorted(x)) if isinstance(x, list) else ''
            )
        
        print(f"Retrieved NP Classifier predictions for {len(np_results)} compounds")
        return np_results
    
    def validate_formula_agreement(self, df: pd.DataFrame, sirius_formula: pd.DataFrame,
                                 predicted_formula_col: str = 'predicted_formula') -> pd.DataFrame:
        """
        Validate agreement between predicted formulas and SIRIUS formulas.
        
        Args:
            df: DataFrame with predicted formulas
            sirius_formula: SIRIUS formula results
            predicted_formula_col: Column name with predicted formulas
            
        Returns:
            DataFrame showing formula agreement statistics
        """
        # Merge predicted and SIRIUS formulas
        formula_merge = pd.merge(
            df[[predicted_formula_col]].rename(columns={predicted_formula_col: 'predicted_formula'}),
            sirius_formula[['mappingFeatureId', 'molecularFormula']],
            left_index=True, 
            right_on='mappingFeatureId', 
            how='left'
        )
        
        formula_merge.set_index('mappingFeatureId', inplace=True)
        formula_merge = formula_merge[
            formula_merge['molecularFormula'].notna() & 
            formula_merge['predicted_formula'].notna()
        ]
        
        # Check agreement
        formula_merge['agree'] = (
            formula_merge['predicted_formula'] == formula_merge['molecularFormula']
        )
        
        agreement_stats = formula_merge['agree'].value_counts()
        agreement_rate = agreement_stats.get(True, 0) / len(formula_merge) if len(formula_merge) > 0 else 0
        
        print(f"Formula agreement: {agreement_stats.get(True, 0)}/{len(formula_merge)} ({agreement_rate:.2%})")
        
        return formula_merge
    
    def load_cytoscape_positions(self, cyjs_file: str) -> pd.DataFrame:
        """
        Load node positions from Cytoscape .cyjs file.
        
        Args:
            cyjs_file: Path to Cytoscape .cyjs file
            
        Returns:
            DataFrame with node positions and attributes
        """
        with open(cyjs_file, 'r') as f:
            network_data = json.load(f)
        
        # Extract node data
        nodes_data = pd.DataFrame(network_data['elements']['nodes'])
        
        # Normalize nested dictionaries
        data_df = pd.json_normalize(nodes_data['data'])
        position_df = pd.json_normalize(nodes_data['position'])
        
        # Combine data and positions
        network_positions = pd.concat([data_df, position_df], axis=1)
        
        # Clean up column names and types
        if 'name' in network_positions.columns:
            network_positions['original_index'] = network_positions['name'].astype(int)
        
        print(f"Loaded positions for {len(network_positions)} nodes from Cytoscape")
        return network_positions


def integrate_sirius_results(network: nx.Graph, sirius_dir: str, 
                           config: Optional[BuildConfig] = None) -> nx.Graph:
    """
    Convenience function to integrate SIRIUS results into a network.
    
    Args:
        network: NetworkX graph
        sirius_dir: Directory containing SIRIUS results
        config: BuildConfig instance (will create default if None)
        
    Returns:
        Updated NetworkX graph with SIRIUS attributes
    """
    if config is None:
        config = BuildConfig()
    
    integrator = SiriusIntegrator(config)
    return integrator.integrate_predictions(network, sirius_dir)