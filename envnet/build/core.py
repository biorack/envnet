"""
Core ENVnet builder interface.

This module provides the main high-level interface for ENVnet molecular network
construction. It wraps all the specialized modules into a clean, easy-to-use API.

Use it like this

# Simple usage
from envnet.build import ENVnetBuilder
builder = ENVnetBuilder()
results = builder.build_network()

# Or even simpler
from envnet.build import build_envnet
builder = build_envnet()

# Access results
network = builder.network
node_data = builder.node_data
builder.save_network("my_network.graphml")

# Quick testing
quick_builder = quick_envnet(max_spectra=500)


"""

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from ..config.build_config import BuildConfig


class ENVnetBuilder:
    """
    Main interface for building ENVnet molecular networks.
    
    This class provides a simple, high-level API for the complete ENVnet pipeline
    from raw deconvoluted spectra to annotated molecular networks.
    
    Example:
        >>> from envnet.build import ENVnetBuilder
        >>> builder = ENVnetBuilder()
        >>> results = builder.build_network()
        >>> network = results['network']
        >>> builder.save_network("my_network.graphml")
    """
    
    def __init__(self, config: Optional[BuildConfig] = None, verbose: bool = True):
        """
        Initialize ENVnet builder.
        
        Args:
            config: Build configuration (uses defaults if None)
            verbose: Whether to print progress messages
        """
        self.config = config or BuildConfig()
        self.verbose = verbose

        # Import here to avoid circular dependency
        from .workflows import ENVnetWorkflows
        self.workflows = ENVnetWorkflows(self.config)
        
        # Results storage
        self._results: Dict[str, Any] = {}
        self._network: Optional[nx.Graph] = None
        self._node_data: Optional[pd.DataFrame] = None
        self._library_matches: Dict[str, pd.DataFrame] = {}
        
        if self.verbose:
            print("ENVnet Builder initialized")
            # print(f"Configuration: {self.config}")
    
    @property
    def network(self) -> Optional[nx.Graph]:
        """Get the current molecular network."""
        return self._network
    
    @property
    def node_data(self) -> Optional[pd.DataFrame]:
        """Get the current node data."""
        return self._node_data
    
    @property
    def library_matches(self) -> Dict[str, pd.DataFrame]:
        """Get library matching results."""
        return self._library_matches
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get all stored results."""
        return self._results
    
    def build_network(self, file_source: str = "google_sheets", 
                     output_dir: str = "data") -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Build complete ENVnet molecular network.
        
        This is the main method that runs the complete pipeline:
        1. Loads deconvoluted spectra
        2. Finds library matches
        3. Clusters duplicate spectra
        4. Builds molecular network
        5. Creates MGF files for SIRIUS
        
        Args:
            file_source: Source for loading file metadata
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing network, node data, and other results
        """
        if self.verbose:
            print("Starting complete ENVnet construction...")
        
        try:
            # Run complete workflow
            results = self.workflows.build_complete_network(file_source, output_dir)
            
            # Store results
            self._results.update(results)
            self._network = results['network']
            self._node_data = results['node_data']
            self._library_matches = results['library_matches']
            
            if self.verbose:
                stats = results['network_stats']
                print(f"\nNetwork built successfully!")
                print(f"Nodes: {stats['num_nodes']}")
                print(f"Edges: {stats['num_edges']}")
                print(f"Connected components: {stats['num_connected_components']}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Error building network: {e}")
            raise

    def quick_build(self, max_spectra: int = 1000, max_files: int = 5,
                   output_dir: str = "data") -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Quick network build for testing/development.
        
        Uses a subset of data for faster processing.
        
        Args:
            max_spectra: Maximum number of spectra to process
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing network and results
        """
        if self.verbose:
            print(f"Starting quick build with max {max_spectra} spectra...")
        
        try:
            results = self.workflows.quick_network_build(max_spectra, max_files, output_dir)
            
            # Store results
            self._results.update(results)
            self._network = results['network']
            self._node_data = results['node_data']
            self._library_matches = {'deconvoluted': results['library_matches']}
            
            if self.verbose:
                print(f"Quick build complete! Network: {self._network.number_of_nodes()} nodes, {self._network.number_of_edges()} edges")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Error in quick build: {e}")
            raise
    
    def analyze_library_matches(self, spectra_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze library matching performance.
        
        Args:
            spectra_df: Spectral data to analyze (loads if None)
            
        Returns:
            Dictionary with library matching results and statistics
        """
        if self.verbose:
            print("Analyzing library matches...")
        
        results = self.workflows.library_matching_workflow(spectra_df)
        self._library_matches = {
            'deconvoluted': results['deconvoluted_matches'],
            'original': results['original_matches']
        }
        
        if self.verbose:
            stats = results['statistics']
            print(f"Library matching complete:")
            print(f"  Deconvoluted match rate: {stats['deconv_match_rate']:.1%}")
            print(f"  Original match rate: {stats['orig_match_rate']:.1%}")
        
        return results
    
    def analyze_clustering(self, spectra_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze spectral clustering performance.
        
        Args:
            spectra_df: Spectral data to analyze (loads if None)
            
        Returns:
            Dictionary with clustering results and statistics
        """
        if self.verbose:
            print("Analyzing spectral clustering...")
        
        results = self.workflows.clustering_workflow(spectra_df)
        
        if self.verbose:
            stats = results['statistics']
            print(f"Clustering complete:")
            print(f"  Reduction factor: {stats['reduction_factor']:.1f}x")
            print(f"  Mean cluster size: {stats['mean_cluster_size']:.1f}")
        
        return results
    
    def integrate_sirius(self, sirius_dir: str) -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Integrate SIRIUS formula and compound class predictions.
        
        Args:
            sirius_dir: Directory containing SIRIUS results
            
        Returns:
            Dictionary with integrated network and SIRIUS data
        """
        if self._network is None:
            raise ValueError("Must build network before integrating SIRIUS results")
        
        if self.verbose:
            print("Integrating SIRIUS results...")
        
        results = self.workflows.sirius_integration_workflow(sirius_dir, self._network)
        
        # Update stored network
        self._network = results['network']
        self._results['sirius_results'] = results
        
        if self.verbose:
            formula_results = results['formula_results']
            class_results = results['class_results']
            print(f"SIRIUS integration complete:")
            print(f"  Formula predictions: {len(formula_results)}")
            print(f"  Compound classifications: {len(class_results)}")
        
        return results
    
    def save_network(self, filepath: str, format: str = "auto") -> None:
        """
        Save the molecular network to file.
        
        Args:
            filepath: Output file path
            format: File format ("graphml", "gexf", "gml", or "auto" to detect from extension)
        """
        if self._network is None:
            raise ValueError("No network to save. Build network first.")
        
        filepath = Path(filepath)
        
        # Auto-detect format from extension
        if format == "auto":
            ext = filepath.suffix.lower()
            if ext == ".graphml":
                format = "graphml"
            elif ext == ".gexf":
                format = "gexf"
            elif ext == ".gml":
                format = "gml"
            else:
                format = "graphml"  # default
                filepath = filepath.with_suffix(".graphml")
        
        # Create output directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save network
        if format == "graphml":
            nx.write_graphml(self._network, filepath)
        elif format == "gexf":
            nx.write_gexf(self._network, filepath)
        elif format == "gml":
            nx.write_gml(self._network, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            print(f"Network saved to: {filepath}")
    
    def save_node_data(self, filepath: str) -> None:
        """
        Save node data to CSV file.
        
        Args:
            filepath: Output file path
        """
        if self._node_data is None:
            raise ValueError("No node data to save. Build network first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving (remove complex columns)
        save_data = self._node_data.copy()
        complex_cols = [
            'deconvoluted_spectrum_mz_vals', 'deconvoluted_spectrum_intensity_vals',
            'original_spectrum_mz_vals', 'original_spectrum_intensity_vals'
        ]
        save_data.drop(columns=complex_cols, errors='ignore', inplace=True)
        
        save_data.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"Node data saved to: {filepath}")
    
    def save_library_matches(self, output_dir: str = "data") -> Dict[str, str]:
        """
        Save library matching results to CSV files.
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping match type to file path
        """
        if not self._library_matches:
            raise ValueError("No library matches to save. Run library matching first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for match_type, matches_df in self._library_matches.items():
            if len(matches_df) > 0:
                filepath = output_dir / f"library_matches_{match_type}.csv"
                matches_df.to_csv(filepath, index=False)
                saved_files[match_type] = str(filepath)
                
                if self.verbose:
                    print(f"Saved {match_type} matches to: {filepath}")
        
        return saved_files
    
    def get_network_summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Get summary statistics for the current network.
        
        Returns:
            Dictionary with network statistics
        """
        if self._network is None:
            return {"status": "No network built"}
        
        stats = self.workflows.network_builder.get_network_statistics(self._network)
        
        # Add additional summary info
        stats['status'] = 'Built'
        if self._node_data is not None:
            stats['total_spectra_analyzed'] = len(self._node_data)
        
        return stats
    
    def reset(self) -> None:
        """Reset the builder state, clearing all results."""
        self._results = {}
        self._network = None
        self._node_data = None
        self._library_matches = {}
        
        if self.verbose:
            print("Builder state reset")
    
    def __repr__(self) -> str:
        """String representation of the builder."""
        status = "No network" if self._network is None else f"Network: {self._network.number_of_nodes()} nodes"
        return f"ENVnetBuilder({status})"


# Convenience functions for common use cases
def build_envnet(config: Optional[BuildConfig] = None, 
               output_dir: str = "data", 
               verbose: bool = True) -> ENVnetBuilder:
    """
    Convenience function to build complete ENVnet.
    
    Args:
        config: Build configuration
        output_dir: Output directory
        verbose: Print progress messages
        
    Returns:
        ENVnetBuilder instance with completed network
    """
    builder = ENVnetBuilder(config, verbose)
    builder.build_network(output_dir=output_dir)
    return builder


def quick_envnet(max_spectra: int = 1000, max_files: int = 5,
               config: Optional[BuildConfig] = None,
               output_dir: str = "data",
               verbose: bool = True) -> ENVnetBuilder:
    """
    Convenience function for quick ENVnet testing.
    
    Args:
        max_spectra: Maximum spectra to process
        max_files: Maximum files to load
        config: Build configuration
        verbose: Print progress messages
        
    Returns:
        ENVnetBuilder instance with test network
    """
    builder = ENVnetBuilder(config, verbose)
    builder.quick_build(max_spectra=max_spectra, max_files=max_files, output_dir=output_dir)
    return builder