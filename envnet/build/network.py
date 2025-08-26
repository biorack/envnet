"""
Molecular network construction using REM-BLINK scoring.

This module handles the creation of molecular networks from deconvoluted spectra
using REM (Random Forest for Mass spectrometry) and BLINK scoring algorithms.

You can use it like

from .network import NetworkBuilder

builder = NetworkBuilder(config)
network = builder.build_remblink_network(node_data)

"""

import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import List, Optional, Tuple
from joblib import load

from ..config.build_config import BuildConfig
import blink


class NetworkBuilder:
    """Handles molecular network construction using REM-BLINK."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def do_remblink_networking(self, query: pd.DataFrame, ref: pd.DataFrame,
                              mass_diffs: Optional[List[float]] = None,
                              spectra_attr: str = 'msv') -> pd.DataFrame:
        """
        Perform REM-BLINK networking between query and reference spectra.
        
        Args:
            query: DataFrame containing query spectra
            ref: DataFrame containing reference spectra  
            mass_diffs: List of mass differences for networking (default: MDM masses)
            spectra_attr: Column name containing spectral data
            
        Returns:
            DataFrame containing REM-BLINK scores between spectra pairs
        """
        polarity = 'negative'
        
        # Load the trained random forest model
        model_file = self.config.model_file
        print(f'Using model: {model_file}')
        
        with open(model_file, 'rb') as f:
            regressor = load(f)
        
        # Prepare spectral data
        query_spectra = query[spectra_attr].tolist()
        query_precursor_mzs = query['precursor_mz'].tolist()
        
        ref_spectra = ref[spectra_attr].tolist()
        ref_precursor_mzs = ref['precursor_mz'].tolist()
        
        # Use MDM masses if not provided
        if mass_diffs is None:
            mass_diffs = self.config.mdm_masses
        
        # Discretize spectra for networking
        d_specs = blink.discretize_spectra(
            query_spectra,
            ref_spectra,
            query_precursor_mzs,
            ref_precursor_mzs,
            intensity_power=self.config.intensity_power,
            bin_width=self.config.bin_width,
            tolerance=self.config.mz_tol,
            network_score=True,
            mass_diffs=mass_diffs
        )
        
        # Score spectra
        scores = blink.score_sparse_spectra(d_specs)
        stacked_scores, stacked_counts = blink.stack_network_matrices(scores)
        rem_scores, predicted_rows = blink.rem_predict(
            stacked_scores, scores, regressor, min_predicted_score=0.0001
        )
        
        # Create output dataframe
        score_rem_df, matches_rem_df = blink.make_rem_df(
            rem_scores, stacked_counts, predicted_rows, mass_diffs=mass_diffs
        )
        rem_df = pd.concat([score_rem_df, matches_rem_df], axis=1)
        
        # Convert sparse to dense
        rem_df = rem_df.sparse.to_dense()
        
        return rem_df
    
    def build_remblink_network(self, node_data: pd.DataFrame) -> nx.Graph:
        """
        Build complete molecular network from node data using REM-BLINK.
        
        Args:
            node_data: DataFrame containing spectral data for network nodes
            
        Returns:
            NetworkX graph representing the molecular network
        """
        print(f"Building network from {len(node_data)} spectra...")
        
        # Prepare spectral data for networking
        node_data = node_data.copy()
        node_data['deconvoluted_spectrum'] = node_data.apply(
            lambda row: np.array([row['deconvoluted_spectrum_mz_vals'],
                                row['deconvoluted_spectrum_intensity_vals']]), 
            axis=1
        )
        
        # Process in chunks to manage memory
        node_chunks = np.array_split(node_data, np.ceil(len(node_data) / self.config.chunk_size))
        print(f"Processing {len(node_chunks)} chunks...")
        
        rem_df_list = []
        
        for i, chunk in enumerate(node_chunks):
            print(f"Processing chunk {i+1}/{len(node_chunks)}")
            
            temp_edges = self.do_remblink_networking(
                chunk, node_data, 
                mass_diffs=self.config.mdm_masses,
                spectra_attr='deconvoluted_spectrum'
            )
            
            # Rename score column and filter
            temp_edges.rename(columns={'rem_predicted_score': 'rem_blink_score'}, inplace=True)
            cols = ['ref', 'query', 'rem_blink_score']
            temp_edges = temp_edges[cols]
            
            # Adjust query indices for chunk offset
            temp_edges['query'] = temp_edges['query'] + (i * chunk.shape[0])
            
            # Filter low scores
            temp_edges = temp_edges[temp_edges['rem_blink_score'] > self.config.remblink_cutoff]
            
            if temp_edges.shape[0] > 0:
                rem_df_list.append(temp_edges)
        
        # Combine all edges
        if not rem_df_list:
            print("No edges found - creating empty network")
            return nx.Graph()
            
        rem_df = pd.concat(rem_df_list, ignore_index=True)
        print(f"Total edges before filtering: {rem_df.shape[0]}")
        
        # Apply score cutoff
        edges_df = self._filter_edges(rem_df, node_data)
        
        # Create NetworkX graph
        network = self._create_networkx_graph(edges_df, node_data)
        
        return network
    
    def _filter_edges(self, rem_df: pd.DataFrame, node_data: pd.DataFrame) -> pd.DataFrame:
        """Filter edges based on score and m/z difference cutoffs."""
        # Apply REM-BLINK score cutoff
        temp = rem_df[rem_df['rem_blink_score'] > self.config.remblink_cutoff].copy()
        print(f"Edges after score cutoff ({self.config.remblink_cutoff}): {temp.shape[0]}")
        
        # Add precursor m/z information
        temp = pd.merge(
            temp, 
            node_data[['original_index', 'precursor_mz']].add_suffix('_ref'),
            left_on='ref', right_index=True, how='left'
        )
        temp = pd.merge(
            temp, 
            node_data[['original_index', 'precursor_mz']].add_suffix('_query'),
            left_on='query', right_index=True, how='left'
        )
        
        # Calculate m/z differences
        temp['mz_difference'] = abs(temp['precursor_mz_ref'] - temp['precursor_mz_query'])
        
        # Apply m/z difference cutoff
        temp = temp[temp['mz_difference'] < self.config.network_max_mz_difference]
        print(f"Edges after m/z cutoff ({self.config.network_max_mz_difference}): {temp.shape[0]}")
        
        # Clean up columns
        temp.drop(columns=['ref', 'query'], inplace=True)
        temp.rename(columns={
            'original_index_ref': 'ref',
            'original_index_query': 'query'
        }, inplace=True)
        
        # Ensure integer types
        temp['ref'] = temp['ref'].astype(int)
        temp['query'] = temp['query'].astype(int)
        
        return temp
    
    def _create_networkx_graph(self, edges_df: pd.DataFrame, 
                              node_data: pd.DataFrame) -> nx.Graph:
        """Create NetworkX graph from edges and add node attributes."""
        # Create graph from edges
        G = nx.from_pandas_edgelist(
            edges_df, 
            source='ref', 
            target='query', 
            edge_attr='rem_blink_score'
        )
        
        # Remove self-loops
        G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])
        
        # Prepare node attributes
        drop_cols = [
            'obs', 'coisolated_precursor_mz_list',
            'deconvoluted_spectrum_mz_vals', 'deconvoluted_spectrum_intensity_vals',
            'original_spectrum_mz_vals', 'original_spectrum_intensity_vals',
            'duplicate_entries', 'deconvoluted_spectrum', 'original_spectrum'
        ]
        
        temp_node_data = node_data.copy()
        temp_node_data.set_index('original_index', inplace=True)
        temp_node_data.drop(columns=drop_cols, errors='ignore', inplace=True)
        node_data_dict = temp_node_data.fillna('').to_dict(orient='index')
        
        # Add node attributes
        nx.set_node_attributes(G, node_data_dict)
        
        # Report network statistics
        num_nodes_in_network = len([n for n in node_data['original_index'] 
                                   if n in G.nodes()])
        print(f'Network nodes: {G.number_of_nodes()}')
        print(f'Network edges: {G.number_of_edges()}')
        print(f'Connected components: {nx.number_connected_components(G)}')
        print(f'Nodes from original data in network: {num_nodes_in_network}/{len(node_data)}')
        
        return G
    
    def save_network(self, network: nx.Graph, filepath: str) -> None:
        """Save network to GraphML format."""
        nx.write_graphml(network, filepath)
        print(f"Network saved to: {filepath}")
    
    def get_network_statistics(self, network: nx.Graph) -> dict:
        """Get basic network statistics."""
        stats = {
            'num_nodes': network.number_of_nodes(),
            'num_edges': network.number_of_edges(),
            'num_connected_components': nx.number_connected_components(network),
            'density': nx.density(network)
        }
        
        if stats['num_nodes'] > 0:
            # Get largest connected component
            largest_cc = max(nx.connected_components(network), key=len)
            stats['largest_component_size'] = len(largest_cc)
            stats['largest_component_fraction'] = len(largest_cc) / stats['num_nodes']
        
        return stats