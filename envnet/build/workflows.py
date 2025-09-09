"""
High-level workflows for ENVnet molecular network construction.

This module provides complete end-to-end workflows that chain together
the core modules for common ENVnet analysis tasks.

Use it like this:
# Complete workflow
from envnet.build.workflows import build_envnet_from_scratch
results = build_envnet_from_scratch()

# Or step by step
workflows = ENVnetWorkflows()
network_results = workflows.build_complete_network()
sirius_results = workflows.sirius_integration_workflow("path/to/sirius/results")

"""

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config.build_config import BuildConfig
from .dataloading import SpectraLoader
from .library_matching import LibraryMatcher
from .clustering import SpectraClusterer
from .network import NetworkBuilder
from .mgf_tools import MGFGenerator
from .integration import SiriusIntegrator
from .formula_tools import get_formula_props


class ENVnetWorkflows:
    """High-level workflows for ENVnet construction and analysis."""
    
    def __init__(self, config: BuildConfig):
        """Initialize workflows with configuration."""
        self.config = config
        
        # Initialize component modules
        self.data_loader = SpectraLoader(self.config)
        self.library_matcher = LibraryMatcher(self.config)
        self.clusterer = SpectraClusterer(self.config)
        self.network_builder = NetworkBuilder(self.config)
        self.mgf_generator = MGFGenerator(self.config)
        self.sirius_integrator = SiriusIntegrator(self.config)
        
        # State tracking
        self.all_spectra: Optional[pd.DataFrame] = None
        self.node_data: Optional[pd.DataFrame] = None
        self.network: Optional[nx.Graph] = None
        self.library_matches: Dict[str, pd.DataFrame] = {}
        
    def build_complete_network(self,
                               output_dir: str = "data") -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Complete end-to-end network building workflow.
        
        This workflow:
        1. Loads deconvoluted spectra data
        2. Finds library matches (original and deconvoluted)
        3. Clusters duplicate spectra
        4. Eliminates redundant spectra
        5. Builds molecular network
        6. Creates MGF files for SIRIUS
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing network, node data, and library matches
        """
        print("=" * 60)
        print("Starting complete ENVnet construction workflow")
        print("=" * 60)
        self._dump_config_to_json(str(output_dir))

        # Step 1: Load data
        print("\n1. Loading deconvoluted spectra data...")
        self.all_spectra = self.data_loader.load_all_spectra()
        print(f"   Loaded {len(self.all_spectra)} spectra from {self.all_spectra['filename'].nunique()} files")
        
        # Step 2: Find library matches
        print("\n2. Finding library matches...")
        print("   - Scoring against deconvoluted reference library...")
        deconvoluted_matches = self.library_matcher.score_all_spectra(
            self.all_spectra, scoring_type='deconvoluted'
        )
        
        print("   - Scoring against original reference library...")
        original_matches = self.library_matcher.score_all_spectra(
            self.all_spectra, scoring_type='original'
        )
        
        self.library_matches = {
            'deconvoluted': deconvoluted_matches,
            'original': original_matches
        }
        
        print(f"   Found {len(deconvoluted_matches)} deconvoluted matches, {len(original_matches)} original matches")
        
        # Step 3: Cluster spectra
        print("\n3. Clustering duplicate spectra...")
        clustered_spectra = self.clusterer.cluster_duplicate_spectra(self.all_spectra)
        print(f"   Identified {clustered_spectra['cluster_label'].nunique()} unique spectral clusters")
        
        # Step 4: Eliminate redundant spectra
        print("\n4. Eliminating redundant spectra...")
        self.node_data = self.clusterer.eliminate_redundant_spectra(
            clustered_spectra, deconvoluted_matches, original_matches
        )
        print(f"   Reduced to {len(self.node_data)} unique representative spectra")
        
        # Step 5: Build network
        print("\n5. Building molecular network...")
        self.network = self.network_builder.build_remblink_network(self.node_data)
        stats = self.network_builder.get_network_statistics(self.network)
        print(f"   Network: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        print(f"   Connected components: {stats['num_connected_components']}")
        
        # Step 6: Create MGF files
        print("\n6. Creating MGF files...")
        mgf_files = self.mgf_generator.create_mgf_files(self.node_data, self.network, output_dir)
        for spectrum_type, filepath in mgf_files.items():
            print(f"   Created {spectrum_type} MGF: {filepath}")
        
        print("\n" + "=" * 60)
        print("ENVnet construction complete!")
        print("=" * 60)

        return {
            'network': self.network,
            'node_data': self.node_data,
            'all_spectra': self.all_spectra,
            'library_matches': self.library_matches,
            'mgf_files': mgf_files,
            'network_stats': stats
        }
    
    def library_matching_workflow(self, spectra_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Workflow focused on library matching analysis.
        
        Args:
            spectra_df: Spectral data (will load if None)
            
        Returns:
            Dictionary with library matching results and statistics
        """
        print("Starting library matching workflow...")
        
        if spectra_df is None:
            if self.all_spectra is None:
                print("Loading spectral data...")
                self.all_spectra = self.data_loader.load_all_spectra()
            spectra_df = self.all_spectra
        
        # Find matches for both spectrum types
        deconvoluted_matches = self.library_matcher.score_all_spectra(
            spectra_df, scoring_type='deconvoluted'
        )
        
        original_matches = self.library_matcher.score_all_spectra(
            spectra_df, scoring_type='original'
        )
        
        # Generate match statistics
        stats = self._calculate_library_match_stats(deconvoluted_matches, original_matches, spectra_df)
        
        results = {
            'deconvoluted_matches': deconvoluted_matches,
            'original_matches': original_matches,
            'statistics': stats,
            'spectra_data': spectra_df
        }
        
        self.library_matches = {
            'deconvoluted': deconvoluted_matches,
            'original': original_matches
        }
        
        return results
    
    def clustering_workflow(self, spectra_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Workflow focused on spectral clustering analysis.
        
        Args:
            spectra_df: Spectral data (will load if None)
            
        Returns:
            Dictionary with clustering results and statistics
        """
        print("Starting spectral clustering workflow...")
        
        if spectra_df is None:
            if self.all_spectra is None:
                print("Loading spectral data...")
                self.all_spectra = self.data_loader.load_all_spectra()
            spectra_df = self.all_spectra
        
        # Perform clustering
        clustered_spectra = self.clusterer.cluster_duplicate_spectra(spectra_df)
        
        # Generate clustering statistics
        stats = self._calculate_clustering_stats(clustered_spectra)
        
        return {
            'clustered_spectra': clustered_spectra,
            'original_spectra': spectra_df,
            'statistics': stats
        }
    
    def network_building_workflow(self, node_data: Optional[pd.DataFrame] = None,
                                output_dir: str = "data") -> Dict[str, Union[nx.Graph, Dict]]:
        """
        Workflow focused on network construction.
        
        Args:
            node_data: Prepared node data (will use existing if None)
            output_dir: Directory for output files
            
        Returns:
            Dictionary with network and related outputs
        """
        print("Starting network building workflow...")
        
        if node_data is None:
            if self.node_data is None:
                raise ValueError("No node data available. Run complete workflow first.")
            node_data = self.node_data
        
        # Build network
        network = self.network_builder.build_remblink_network(node_data)
        
        # Create MGF files
        mgf_files = self.mgf_generator.create_mgf_files(node_data, network, output_dir)
        
        # Get network statistics
        stats = self.network_builder.get_network_statistics(network)

        self.network = network
        
        return {
            'network': network,
            'mgf_files': mgf_files,
            'statistics': stats,
            'node_data': node_data
        }
    
    def sirius_integration_workflow(self, sirius_dir: str, 
                                  network: Optional[nx.Graph] = None) -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Workflow for integrating SIRIUS results.
        
        Args:
            sirius_dir: Directory containing SIRIUS results
            network: Network to integrate results into (will use existing if None)
            
        Returns:
            Dictionary with integrated network and SIRIUS data
        """
        print("Starting SIRIUS integration workflow...")
        
        if network is None:
            if self.network is None:
                raise ValueError("No network available. Build network first.")
            network = self.network
        
        # Load SIRIUS results
        formula_results = self.sirius_integrator.load_sirius_formula_results(sirius_dir)
        class_results = self.sirius_integrator.load_sirius_canopus_results(sirius_dir)
        
        # Integrate into network
        integrated_network = self.sirius_integrator.integrate_predictions(network, sirius_dir)
        
        # Validate formula agreement if we have predicted formulas
        formula_agreement = None
        if self.node_data is not None and 'predicted_formula' in self.node_data.columns:
            formula_agreement = self.sirius_integrator.validate_formula_agreement(
                self.node_data, formula_results
            )
        
        self.network = integrated_network
        
        return {
            'network': integrated_network,
            'formula_results': formula_results,
            'class_results': class_results,
            'formula_agreement': formula_agreement
        }
    
    def quick_network_build(self, max_spectra: int = 1000, max_files: int = 5,
                          output_dir: str = "data") -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
        """
        Quick network building workflow for testing/development.
        
        Uses subset of data for faster processing.
        
        Args:
            max_spectra: Maximum number of spectra to process
            max_files: Maximum number of files to load
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing network and related data
        """
        print(f"Starting quick network build (max {max_spectra} spectra, {max_files} files)...")
        self._dump_config_to_json(str(output_dir))

        # Load and subsample data
        all_spectra = self.data_loader.load_all_spectra(max_files=max_files)
        if len(all_spectra) > max_spectra:
            all_spectra = all_spectra.sample(n=max_spectra, random_state=42)
            print(f"Subsampled to {len(all_spectra)} spectra")
        
        # Run abbreviated workflow
        deconvoluted_matches = self.library_matcher.score_all_spectra(
            all_spectra, scoring_type='deconvoluted'
        )
        
        clustered_spectra = self.clusterer.cluster_duplicate_spectra(all_spectra)
        
        self.node_data = self.clusterer.eliminate_redundant_spectra(
            clustered_spectra, deconvoluted_matches, pd.DataFrame()
        )

        self.network = self.network_builder.build_remblink_network(self.node_data)

        # create mgf files
        mgf_files = self.mgf_generator.create_mgf_files(self.node_data, self.network, output_dir)
        for spectrum_type, filepath in mgf_files.items():
            print(f"   Created {spectrum_type} MGF: {filepath}")

        return {
            'network': self.network,
            'node_data': self.node_data,
            'library_matches': deconvoluted_matches
        }
    
    def _calculate_library_match_stats(self, deconv_matches: pd.DataFrame, 
                                     orig_matches: pd.DataFrame,
                                     spectra_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Calculate statistics for library matching results."""
        total_spectra = len(spectra_df)
        
        deconv_matched = deconv_matches['original_index'].nunique() if len(deconv_matches) > 0 else 0
        orig_matched = orig_matches['original_index'].nunique() if len(orig_matches) > 0 else 0
        
        return {
            'total_spectra': total_spectra,
            'deconvoluted_matches': len(deconv_matches),
            'original_matches': len(orig_matches),
            'spectra_with_deconv_matches': deconv_matched,
            'spectra_with_orig_matches': orig_matched,
            'deconv_match_rate': deconv_matched / total_spectra,
            'orig_match_rate': orig_matched / total_spectra
        }
    
    def _calculate_clustering_stats(self, clustered_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Calculate statistics for clustering results."""
        total_spectra = len(clustered_df)
        unique_clusters = clustered_df['cluster_label'].nunique()
        
        cluster_sizes = clustered_df.groupby(['cluster_label', 'precursor_mz_group']).size()
        
        return {
            'total_spectra': total_spectra,
            'unique_clusters': unique_clusters,
            'reduction_factor': total_spectra / unique_clusters,
            'mean_cluster_size': cluster_sizes.mean(),
            'median_cluster_size': cluster_sizes.median(),
            'max_cluster_size': cluster_sizes.max(),
            'singleton_clusters': (cluster_sizes == 1).sum()
        }

    def _dump_config_to_json(self, output_dir: str):
        """
        Save configuration parameters to JSON file in output directory.
        
        Args:
            output_dir: Output directory path
        """
        import json
        from datetime import datetime
        from pathlib import Path
        import dataclasses
        
        output_path = Path(output_dir)
        config_file = output_path / "build _config.json"

        # Convert config to dictionary
        if dataclasses.is_dataclass(self.config):
            config_dict = dataclasses.asdict(self.config)
        else:
            config_dict = vars(self.config)
        
        # Add workflow metadata
        config_data = {
            "workflow_info": {
                "timestamp": datetime.now().isoformat(),
                "envnet_version": getattr(self.config, 'version', 'unknown')
            },
            "configuration": config_dict
        }
        
        # Handle non-serializable objects
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            else:
                return str(obj)
        
        # Save to file
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=json_serializer)
            print(f"Configuration saved to: {config_file}")
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {e}")

# Convenience functions for common workflows
def build_envnet_from_scratch(config: Optional[BuildConfig] = None, 
                            output_dir: str = "data") -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
    """
    Convenience function to build complete ENVnet from scratch.
    
    Args:
        config: Build configuration
        output_dir: Output directory
        
    Returns:
        Complete ENVnet results
    """
    workflows = ENVnetWorkflows(config)
    return workflows.build_complete_network(output_dir=output_dir)


def quick_envnet_test(max_spectra: int = 500, 
                     config: Optional[BuildConfig] = None) -> Dict[str, Union[nx.Graph, pd.DataFrame]]:
    """
    Convenience function for quick ENVnet testing.
    
    Args:
        max_spectra: Maximum spectra to process
        config: Build configuration
        
    Returns:
        Quick test results
    """
    workflows = ENVnetWorkflows(config)
    return workflows.quick_network_build(max_spectra=max_spectra)