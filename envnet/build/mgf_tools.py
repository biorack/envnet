"""
MGF (Mascot Generic Format) file generation for SIRIUS input.

This module handles the creation of MGF files from spectral data for use with
SIRIUS formula prediction and compound classification tools.

Use it like this:

from .mgf_tools import MGFGenerator

generator = MGFGenerator(config)
files = generator.create_mgf_files(node_data, network, "data")
# Returns: {'deconvoluted': 'data/envnet_deconvoluted_spectra.mgf', 
#           'original': 'data/envnet_original_spectra.mgf'}

"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Union
from pyteomics import mgf

from ..config.build_config import BuildConfig


class MGFGenerator:
    """Handles MGF file generation for SIRIUS input."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def prepare_spectral_data(self, node_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare spectral data by converting spectrum arrays to proper format.
        
        Args:
            node_data: DataFrame containing spectral data
            
        Returns:
            DataFrame with prepared spectral arrays
        """
        data = node_data.copy()
        
        # Create spectrum arrays for both original and deconvoluted spectra
        data['original_spectrum'] = data.apply(
            lambda row: np.array([row['original_spectrum_mz_vals'],
                                row['original_spectrum_intensity_vals']]), 
            axis=1
        )
        
        data['deconvoluted_spectrum'] = data.apply(
            lambda row: np.array([row['deconvoluted_spectrum_mz_vals'],
                                row['deconvoluted_spectrum_intensity_vals']]), 
            axis=1
        )
        
        return data
    
    def create_mgf_file(self, output_filename: str, df: pd.DataFrame, 
                       network: nx.Graph, spectra_type: str = 'deconvoluted_spectrum') -> None:
        """
        Create MGF file from spectral data for SIRIUS input.
        
        Args:
            output_filename: Path for output MGF file
            df: DataFrame containing spectral data
            network: NetworkX graph (only nodes in network will be included)
            spectra_type: Type of spectrum to export ('original_spectrum' or 'deconvoluted_spectrum')
        """
        # MGF header columns required by SIRIUS
        mgf_cols = [
            'FEATURE_ID', 'SCANS', 'ORIGINAL_ID', 'PEPMASS', 
            'PRECURSOR_MZ', 'RTINSECONDS', 'CHARGE', 'MSLEVEL'
        ]
        
        # Filter to only include nodes that are in the network
        temp = df[df['original_index'].isin(network.nodes())].copy()
        temp.reset_index(inplace=True, drop=True)
        
        # Prepare MGF metadata
        temp['FEATURE_ID'] = temp['original_index']
        temp['SCANS'] = temp.index.tolist()  # Sequential scan numbers
        temp['ORIGINAL_ID'] = temp['original_index']
        temp['CHARGE'] = '1-'  # Assuming negative mode
        temp['MSLEVEL'] = 2
        temp['RTINSECONDS'] = temp['rt'] * 60  # Convert minutes to seconds
        temp['PRECURSOR_MZ'] = temp['precursor_mz']
        temp['PEPMASS'] = temp['precursor_mz']
        
        # Build spectra list for pyteomics
        spectra = []
        for i, row in temp.iterrows():
            spectrum_data = row[spectra_type]
            
            spectrum_entry = {
                'params': row[mgf_cols].to_dict(),
                'm/z array': spectrum_data[0],  # m/z values
                'intensity array': spectrum_data[1]  # intensity values
            }
            spectra.append(spectrum_entry)
        
        # Write MGF file
        mgf.write(spectra, output_filename)
        print(f"Created MGF file: {output_filename} with {len(spectra)} spectra")
    
    def create_mgf_files(self, node_data: pd.DataFrame, network: nx.Graph, 
                        output_dir: str = "data") -> Dict[str, str]:
        """
        Create both original and deconvoluted MGF files for SIRIUS.
        
        Args:
            node_data: DataFrame containing spectral data
            network: NetworkX graph
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping spectrum type to output file path
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare spectral data
        prepared_data = self.prepare_spectral_data(node_data)
        
        # Create file paths
        files_created = {}
        
        # Create deconvoluted spectra MGF
        deconv_file = output_path / "envnet_deconvoluted_spectra.mgf"
        self.create_mgf_file(
            str(deconv_file), 
            prepared_data, 
            network, 
            spectra_type='deconvoluted_spectrum'
        )
        files_created['deconvoluted'] = str(deconv_file)
        
        # Create original spectra MGF
        original_file = output_path / "envnet_original_spectra.mgf"
        self.create_mgf_file(
            str(original_file), 
            prepared_data, 
            network, 
            spectra_type='original_spectrum'
        )
        files_created['original'] = str(original_file)
        
        return files_created
    
    def validate_mgf_file(self, mgf_file: str) -> Dict[str, Union[int, bool]]:
        """
        Validate MGF file by reading it back and checking basic properties.
        
        Args:
            mgf_file: Path to MGF file to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            spectra = list(mgf.read(mgf_file))
            
            validation = {
                'valid': True,
                'num_spectra': len(spectra),
                'has_params': all('params' in spec for spec in spectra),
                'has_mz_arrays': all('m/z array' in spec for spec in spectra),
                'has_intensity_arrays': all('intensity array' in spec for spec in spectra),
                'error': None
            }
            
            # Check that all spectra have non-empty arrays
            if validation['has_mz_arrays']:
                validation['all_non_empty_mz'] = all(
                    len(spec['m/z array']) > 0 for spec in spectra
                )
            
            if validation['has_intensity_arrays']:
                validation['all_non_empty_intensity'] = all(
                    len(spec['intensity array']) > 0 for spec in spectra
                )
                
        except Exception as e:
            validation = {
                'valid': False,
                'error': str(e),
                'num_spectra': 0
            }
        
        return validation
    
    def get_mgf_statistics(self, mgf_file: str) -> Dict[str, Union[int, float]]:
        """
        Get statistics about an MGF file.
        
        Args:
            mgf_file: Path to MGF file
            
        Returns:
            Dictionary with MGF file statistics
        """
        spectra = list(mgf.read(mgf_file))
        
        if not spectra:
            return {'num_spectra': 0}
        
        # Calculate statistics
        num_peaks = [len(spec['m/z array']) for spec in spectra]
        precursor_mzs = [spec['params'].get('PRECURSOR_MZ', 0) for spec in spectra]
        
        stats = {
            'num_spectra': len(spectra),
            'mean_peaks_per_spectrum': np.mean(num_peaks),
            'median_peaks_per_spectrum': np.median(num_peaks),
            'min_peaks_per_spectrum': np.min(num_peaks),
            'max_peaks_per_spectrum': np.max(num_peaks),
            'mean_precursor_mz': np.mean(precursor_mzs),
            'min_precursor_mz': np.min(precursor_mzs),
            'max_precursor_mz': np.max(precursor_mzs)
        }
        
        return stats


def create_mgf_files_for_sirius(node_data: pd.DataFrame, network: nx.Graph, 
                               output_dir: str = "data", config: Optional[BuildConfig] = None) -> Dict[str, str]:
    """
    Convenience function to create MGF files for SIRIUS input.
    
    Args:
        node_data: DataFrame containing spectral data
        network: NetworkX graph
        output_dir: Directory for output files
        config: BuildConfig instance (will create default if None)
        
    Returns:
        Dictionary mapping spectrum type to output file path
    """
    if config is None:
        config = BuildConfig()
        
    generator = MGFGenerator(config)
    return generator.create_mgf_files(node_data, network, output_dir)