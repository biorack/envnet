"""
Data loading utilities for ENVnet reference data and experimental data.
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from ..config.annotation_config import AnnotationConfig
from ..vendor.google_sheets import get_google_sheet
import blink as blink


class ENVnetLoader:
    """Loads ENVnet reference data (nodes, edges, spectra)."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def load_reference_data(self, 
                           graphml_file: str = 'envnet.graphml',
                           mgf_base_name: str = 'envnet',
                           custom_mgf_paths: Optional[Dict] = None) -> Dict:
        """
        Load complete ENVnet reference data.
        
        Args:
            graphml_file: GraphML file name (relative to data dir) or absolute path
            mgf_base_name: Base name for MGF files (ignored if custom_mgf_paths provided)
            custom_mgf_paths: Dict with 'deconvoluted' and 'original' keys for custom MGF paths
        
        Returns:
            Dict containing nodes, edges, and spectra data
        """
        data_dir = Path(self.config.module_path) / 'data'
        
        # Handle GraphML file path
        if Path(graphml_file).is_absolute():
            graphml_path = Path(graphml_file)
        else:
            graphml_path = data_dir / graphml_file
            
        # Load graph data
        nodes = self._load_nodes(graphml_path)
        edges = self._load_edges(graphml_path)
        
        # Load spectral data
        if custom_mgf_paths:
            spectra_data = self._load_spectral_data_custom(custom_mgf_paths)
        else:
            spectra_data = self._load_spectral_data(data_dir, mgf_base_name)
        
        # Merge node and spectral data
        merged_nodes = self._merge_node_spectral_data(nodes, spectra_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'merged_nodes': merged_nodes,
            'reference_pmzs': merged_nodes['precursor_mz'].tolist(),
            'deconvoluted_spectra': merged_nodes['spectrum_nl_spectra'].tolist(),
            'original_spectra': merged_nodes['spectrum_original_spectra'].tolist()
        }
    
    def _load_spectral_data(self, data_dir: Path, mgf_base_name: str) -> Dict:
        """Load spectral data from MGF files using base name."""
        original_mgf = data_dir / f'{mgf_base_name}_original_spectra.mgf'
        deconv_mgf = data_dir / f'{mgf_base_name}_deconvoluted_spectra.mgf'
        
        return self._load_spectral_data_from_paths(original_mgf, deconv_mgf)
    
    def _load_spectral_data_custom(self, custom_paths: Dict) -> Dict:
        """Load spectral data from custom MGF file paths."""
        original_mgf = custom_paths.get('original')
        deconv_mgf = custom_paths.get('deconvoluted')
        
        if not original_mgf or not deconv_mgf:
            raise ValueError("Both 'original' and 'deconvoluted' MGF paths must be provided")
            
        return self._load_spectral_data_from_paths(Path(original_mgf), Path(deconv_mgf))
    
    def _load_spectral_data_from_paths(self, original_mgf: Path, deconv_mgf: Path) -> Dict:
        """Load spectral data from specific MGF file paths."""
        if not original_mgf.exists():
            raise FileNotFoundError(f"Original spectra MGF file not found: {original_mgf}")
        if not deconv_mgf.exists():
            raise FileNotFoundError(f"Deconvoluted spectra MGF file not found: {deconv_mgf}")
            
        original_spectra = blink.open_msms_file(str(original_mgf))
        nl_spectra = blink.open_msms_file(str(deconv_mgf))
        
        # Process indices
        original_spectra['original_index'] = original_spectra['original_id'].apply(
            lambda x: int(float(x))
        )
        nl_spectra['original_index'] = nl_spectra['original_id'].apply(
            lambda x: int(float(x))
        )
        
        return {
            'original_spectra': original_spectra,
            'nl_spectra': nl_spectra
        }
    
    def _load_nodes(self, graphml_path: Path) -> pd.DataFrame:
        """Load node data from GraphML file."""
        G = nx.read_graphml(graphml_path)
        node_data = dict(G.nodes(data=True))
        node_data = pd.DataFrame(node_data).T
        node_data.index.name = 'original_index'
        node_data.index = node_data.index.astype(int)
        node_data.reset_index(inplace=True, drop=False)
        return node_data
    
    def _load_edges(self, graphml_path: Path) -> pd.DataFrame:
        """Load edge data from GraphML file."""
        G = nx.read_graphml(graphml_path)
        return nx.to_pandas_edgelist(G)

    def _merge_node_spectral_data(self, node_data: pd.DataFrame,
                                 spectra_data: Dict) -> pd.DataFrame:
        """Merge node data with spectral data."""
        node_data['original_index'] = node_data['original_index'].apply(
            lambda x: int(float(x))
        )
        
        # Add suffixes to avoid column conflicts
        original_spectra = spectra_data['original_spectra'].copy()
        nl_spectra = spectra_data['nl_spectra'].copy()
        
        original_spectra = original_spectra.rename(columns={
            c: c + '_original_spectra' for c in original_spectra.columns 
            if c not in ['original_index']
        })
        
        nl_spectra = nl_spectra.rename(columns={
            c: c + '_nl_spectra' for c in nl_spectra.columns 
            if c not in ['original_index']
        })
        
        # Merge data
        merged = pd.merge(node_data, original_spectra, on='original_index', how='left')
        merged = pd.merge(merged, nl_spectra, on='original_index', how='left')
        
        return merged


class ExperimentalDataLoader:
    """Loads experimental MS1 and MS2 data."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def load_file_metadata(self, 
                          google_sheet_config: Optional[Dict] = None,
                          file_list: Optional[List[str]] = None,
                          csv_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load experimental file metadata from various sources.
        
        Args:
            google_sheet_config: Dict with notebook_name, file_sheet, envo_sheet
            file_list: Direct list of file paths
            csv_file: CSV file with file metadata
            
        Returns:
            pd.DataFrame: File metadata with standardized columns
        """
        if google_sheet_config:
            return self._load_from_google_sheets(google_sheet_config)
        elif file_list:
            return self._load_from_file_list(file_list)
        elif csv_file:
            return self._load_from_csv(csv_file)
        else:
            raise ValueError("Must provide either google_sheet_config, file_list, or csv_file")
    
    def _load_from_google_sheets(self, config: Dict) -> pd.DataFrame:
        """Load file metadata from Google Sheets."""
        # Load file data
        file_df = get_google_sheet(
            notebook_name=config['notebook_name'],
            sheet_name=config['file_sheet']
        )
        
        # Clean up headers
        file_df.columns = file_df.iloc[0]
        file_df = file_df[1:]
        
        # Process file paths
        file_df['parquet'] = file_df['parquet'].apply(
            lambda x: x.replace('.parquet', '_deconvoluted.parquet')
        )
        
        if 'h5' not in file_df.columns:
            file_df['h5'] = file_df['parquet'].str.replace('.parquet', '.h5')
        
        # Load environmental data if provided
        if 'envo_sheet' in config:
            envo_name = get_google_sheet(
                notebook_name=config['notebook_name'],
                sheet_name=config['envo_sheet']
            )
            envo_name.columns = envo_name.iloc[0]
            envo_name = envo_name[1:]
            
            file_df = pd.merge(
                file_df, 
                envo_name[['name', 'id', 'common parent name']],
                left_on='environmental_subclass', 
                right_on='id', 
                how='inner'
            )
        
        # Create run names
        file_df['lcmsrun_observed'] = file_df['h5'].str.replace('.h5', '', regex=False)
        file_df['lcmsrun_observed'] = file_df['lcmsrun_observed'].str.replace(
            '/global/cfs/cdirs/metatlas/projects/carbon_network/raw_data/', '',
            regex=False
        )
        file_df = file_df.sample(10)
        return file_df
    
    def _load_from_file_list(self, file_list: List[str]) -> pd.DataFrame:
        """Load file metadata from a simple file list."""
        df = pd.DataFrame({'parquet': file_list})
        df['h5'] = df['parquet'].str.replace('.parquet', '.h5')
        df['lcmsrun_observed'] = df['h5'].apply(lambda x: Path(x).stem)
        return df
    
    def _load_from_csv(self, csv_file: str) -> pd.DataFrame:
        """Load file metadata from CSV file."""
        return pd.read_csv(csv_file)
    
    def load_ms2_data(self, parquet_files: List[str]) -> pd.DataFrame:
        """Load MS2 data from parquet files."""
        ms2_data = []
        
        for file in tqdm(parquet_files, desc="Loading MS2 data", unit='file'):
            try:
                data = pd.read_parquet(file)
                if data is None or data.shape[0] == 0:
                    continue
                if 'deconvoluted_spectrum_mz_vals' not in data.columns:
                    continue
                ms2_data.append(data)
            except Exception as e:
                print(f'Error loading {file}: {e}')
                continue
        
        if not ms2_data:
            raise ValueError("No valid MS2 data files found")
            
        return pd.concat(ms2_data, ignore_index=True)