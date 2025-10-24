"""
MS1-based feature matching for annotation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from ..config.annotation_config import AnnotationConfig
from ..vendor.feature_tools import group_consecutive, get_atlas_data_from_file, get_atlas_data_from_mzml, calculate_ms1_summary



class MS1Matcher:
    """Handles MS1-based feature matching to ENVnet nodes."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def create_node_atlas(self, node_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a node atlas for MS1 feature extraction.
        
        Args:
            node_data: ENVnet node data
            
        Returns:
            pd.DataFrame: Node atlas with m/z tolerances and RT windows
        """
        node_atlas = node_data[['original_index', 'precursor_mz']].copy()
        node_atlas.rename(columns={
            'precursor_mz': 'mz', 
            'original_index': 'label'
        }, inplace=True)
        
        node_atlas.sort_values('mz', inplace=True)
        
        # Set RT parameters
        node_atlas['rt_tolerance'] = 100
        node_atlas['rt_min'] = self.config.min_rt
        node_atlas['rt_max'] = self.config.max_rt
        node_atlas['rt_peak'] = (self.config.min_rt + self.config.max_rt) / 2
        
        # Set m/z tolerances
        node_atlas['ppm_tolerance'] = self.config.ppm_tolerance
        node_atlas['extra_time'] = 0
        
        # Group consecutive m/z values for efficient processing
        node_atlas['group_index'] = group_consecutive(
            node_atlas['mz'].values,
            stepsize=self.config.ppm_tolerance,
            do_ppm=True
        )
        
        return node_atlas
    
    def extract_ms1_features(self, node_atlas: pd.DataFrame, 
                           sample_files: List[str]) -> pd.DataFrame:
        """
        Extract MS1 features from experimental files using node atlas.
        
        Args:
            node_atlas: Node atlas for feature extraction
            sample_files: List of experimental data files
            
        Returns:
            pd.DataFrame: Extracted MS1 features with matches to ENVnet nodes
        """        

        ms1_data = []
        
        for file in tqdm(sample_files, desc="Extracting MS1 features", unit='file'):
            try:
                # Convert parquet to h5 path if needed
                if file.endswith('parquet'):
                    file = file.replace('.parquet', '.h5')
                
                # Load data based on file type
                if file.endswith(('.mzML', '.mzml')):
                    data = get_atlas_data_from_mzml(file, node_atlas, desired_key='ms1_neg')
                elif file.endswith('.h5'):
                    data = get_atlas_data_from_file(file, node_atlas, desired_key='ms1_neg')
                else:
                    print(f"Unrecognized file type: {file}")
                    continue
                
                # Calculate feature summaries
                feature_data = calculate_ms1_summary(data)
                # feature_data = (data[data['in_feature'] == True]
                #               .groupby('label')
                #               .apply(self._calculate_ms1_summary)
                #               .reset_index())
                feature_data['lcmsrun_observed'] = file
                
                # Filter by minimum data points
                feature_data = feature_data[
                    feature_data['num_datapoints'] >= self.config.min_ms1_datapoints
                ]
                if feature_data is not None and len(feature_data) > 0:
                    ms1_data.append(feature_data)
                
            except Exception as e:
                print(f'Error processing file {file}: {e}')
                continue
        
        if not ms1_data:
            raise ValueError("No valid MS1 data extracted")
            
        # Combine all data
        combined_data = pd.concat(ms1_data, ignore_index=True)
        combined_data.rename(columns={'label': 'original_index'}, inplace=True)
        combined_data['original_index'] = combined_data['original_index'].astype(int)
        combined_data.reset_index(inplace=True, drop=True)

        return combined_data
    
    # def _calculate_ms1_summary(self, row: pd.Series) -> pd.Series:
    #     """
    #     Calculate summary properties for MS1 features.
        
    #     Args:
    #         row: Grouped feature data
            
    #     Returns:
    #         pd.Series: Summary statistics
    #     """
    #     summary = {}
        
    #     summary['num_datapoints'] = row['i'].count()
    #     summary['peak_area'] = row['i'].sum()
        
    #     # Find peak apex
    #     idx = row['i'].idxmax()
    #     summary['peak_height'] = row.loc[idx, 'i']
    #     summary['rt_peak'] = row.loc[idx, 'rt']
        
    #     # Calculate mass centroid
    #     summary['mz_centroid'] = (row['i'] * row['mz']).sum() / summary['peak_area']
        
    #     return pd.Series(summary)