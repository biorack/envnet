"""
Core annotation functionality for mapping experimental data to ENVnet.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from ..config.annotation_config import AnnotationConfig
from .dataloading import ENVnetLoader, ExperimentalDataLoader
from .ms1_matching import MS1Matcher
from .ms2_matching import MS2Matcher
from .preprocessing import AnnotationPreprocessor
from .postprocessing import AnnotationPostprocessor


class AnnotationEngine:
    """
    Main engine for annotating experimental LCMS data using ENVnet reference database.
    
    This class orchestrates the complete annotation workflow:
    1. Load ENVnet reference data (nodes, spectra)
    2. Load experimental data (MS1 features, MS2 spectra)
    3. Match experimental features to ENVnet nodes using MS1 and MS2
    4. Export annotation results
    """
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        """Initialize annotation engine with configuration."""
        self.config = config or AnnotationConfig()
        
        # Initialize components
        self.envnet_loader = ENVnetLoader(self.config)
        self.exp_loader = ExperimentalDataLoader(self.config)
        self.ms1_matcher = MS1Matcher(self.config)
        self.ms2_matcher = MS2Matcher(self.config)
        self.preprocessor = AnnotationPreprocessor(self.config)
        self.postprocessor = AnnotationPostprocessor(self.config)
        
        # Data storage
        self.envnet_data: Optional[Dict] = None
        self.experimental_files: Optional[pd.DataFrame] = None
        
    def load_envnet_reference(self, 
                            graphml_file: str = 'envnet.graphml',
                            mgf_base_name: str = 'envnet',
                            custom_mgf_paths: Optional[Dict] = None) -> Dict:
        """Load ENVnet reference data (nodes and spectra)."""
        print("Loading ENVnet reference data...")
        print(f"  GraphML file: {graphml_file}")
        if custom_mgf_paths:
            print(f"  Deconvoluted MGF: {custom_mgf_paths.get('deconvoluted', 'Not specified')}")
            print(f"  Original MGF: {custom_mgf_paths.get('original', 'Not specified')}")
        else:
            data_dir = Path(self.config.module_path) / 'data'
            print(f"  Deconvoluted MGF: {data_dir}/{mgf_base_name}_deconvoluted_spectra.mgf")
            print(f"  Original MGF: {data_dir}/{mgf_base_name}_original_spectra.mgf")
        
        self.envnet_data = self.envnet_loader.load_reference_data(
            graphml_file, mgf_base_name, custom_mgf_paths
        )
        print(f"Loaded {len(self.envnet_data['nodes'])} ENVnet nodes")
        return self.envnet_data
    
    def load_experimental_files(self, 
                               google_sheet_config: Optional[Dict] = None,
                               file_list: Optional[List[str]] = None,
                               csv_file: Optional[str] = None) -> pd.DataFrame:
        """Load experimental file metadata."""
        print("Loading experimental file metadata...")
        self.experimental_files = self.exp_loader.load_file_metadata(
            google_sheet_config, file_list, csv_file
        )
        print(f"Loaded metadata for {len(self.experimental_files)} files")
        return self.experimental_files
    
    def annotate_ms1_features(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Annotate experimental MS1 features by matching to ENVnet nodes.
        
        Returns:
            pd.DataFrame: MS1 annotation results
        """
        if self.envnet_data is None:
            raise ValueError("Must load ENVnet reference data first")
        if self.experimental_files is None:
            raise ValueError("Must load experimental files first")
            
        print("Starting MS1 feature annotation...")
        
        # Create node atlas for matching
        node_atlas = self.ms1_matcher.create_node_atlas(self.envnet_data['nodes'])

        # Get experimental file type (All files already known to be same type)
        file_type = self.experimental_files['original_file_type'].unique()[0]
        
        # Extract MS1 features from experimental files
        ms1_data = self.ms1_matcher.extract_ms1_features(
            node_atlas, self.experimental_files[file_type.lower()].tolist()
        )
        
        # Post-process results
        ms1_results = self.postprocessor.format_ms1_results(ms1_data, self.experimental_files)
        
        # Save results
        if output_file:
            ms1_results.to_parquet(output_file)
            print(f"MS1 annotation results saved to {output_file}")
            
        return ms1_results
    
    def annotate_ms2_spectra(self, 
                            spectrum_type: str = 'deconvoluted',
                            output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Annotate experimental MS2 spectra by matching to ENVnet spectra.
        
        Args:
            spectrum_type: 'deconvoluted' or 'original'
            output_file: Optional output file path
            
        Returns:
            pd.DataFrame: MS2 annotation results
        """
        if self.envnet_data is None:
            raise ValueError("Must load ENVnet reference data first")
        if self.experimental_files is None:
            raise ValueError("Must load experimental files first")
            
        print(f"Starting MS2 {spectrum_type} spectra annotation...")

        # Get input file type for potential runtime deconvolution (All files already known to be same type)
        file_type = self.experimental_files['original_file_type'].unique()[0]
        
        # Load experimental MS2 data
        ms2_data = self.exp_loader.load_ms2_data(self.experimental_files['parquet'].tolist(), original_file_type=file_type)
        
        # Preprocess MS2 data
        ms2_data = self.preprocessor.filter_ms2_data(ms2_data, self.envnet_data['nodes'])
        
        # Perform spectral matching
        ms2_results = self.ms2_matcher.match_spectra(
            ms2_data, self.envnet_data, spectrum_type
        )
        
        # Post-process results
        ms2_results = self.postprocessor.format_ms2_results(ms2_results, ms2_data)
        
        # Save results
        if output_file:
            ms2_results.to_parquet(output_file)
            print(f"MS2 {spectrum_type} annotation results saved to {output_file}")
            
        return ms2_results
    
    def run_full_annotation(self, 
                           output_dir: str,
                           google_sheet_config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete annotation workflow for both MS1 and MS2.
        
        Returns:
            Dict containing all annotation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Load reference data
        self.load_envnet_reference()
        
        # Load experimental files
        if google_sheet_config:
            self.load_experimental_files(google_sheet_config=google_sheet_config)
        else:
            # Use default Google Sheets config
            default_config = {
                'notebook_name': 'Supplementary Tables',
                'file_sheet': 'Table 1a', 
                'envo_sheet': 'Table 1b'
            }
            self.load_experimental_files(google_sheet_config=default_config)
        
        # MS1 annotation
        ms1_output = output_dir / "ms1_annotations.parquet"
        results['ms1'] = self.annotate_ms1_features(str(ms1_output))
        
        # MS2 annotation - deconvoluted spectra
        ms2_deconv_output = output_dir / "ms2_deconvoluted_annotations.parquet"
        results['ms2_deconvoluted'] = self.annotate_ms2_spectra(
            'deconvoluted', str(ms2_deconv_output)
        )
        
        # MS2 annotation - original spectra  
        ms2_orig_output = output_dir / "ms2_original_annotations.parquet"
        results['ms2_original'] = self.annotate_ms2_spectra(
            'original', str(ms2_orig_output)
        )
        
        print("Complete annotation workflow finished!")
        return results