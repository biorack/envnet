"""
Data loading and preparation for analysis workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
import os

from ..config.analysis_config import AnalysisConfig
from ..vendor.google_sheets import get_google_sheet


class AnnotationDataLoader:
    """Loads and prepares annotation results for analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def load_annotation_data(self, 
                           ms1_file: Optional[str] = None,
                           ms2_deconv_file: Optional[str] = None,
                           ms2_original_file: Optional[str] = None,
                           file_metadata: Optional[Union[str, pd.DataFrame]] = None) -> Dict:
        """
        Load annotation results from files.
        
        Args:
            ms1_file: Path to MS1 annotation results
            ms2_deconv_file: Path to MS2 deconvoluted results
            ms2_original_file: Path to MS2 original results
            file_metadata: File metadata (DataFrame, CSV path, or Google Sheets config)
            
        Returns:
            Dict with loaded data
        """
        data = {}
        
        ms1_cols = ['original_index','num_datapoints','rt_peak','peak_height','peak_area','lcmsrun_observed']
        ms2_cols_deconvoluted = ['original_index_deconvoluted_match', 'score_deconvoluted_match','matches_deconvoluted_match', 'filename']
        ms2_cols_original = ['original_index_original_match', 'score_original_match', 'matches_original_match', 'filename']

        # Load MS1 data
        if ms1_file and Path(ms1_file).exists():
            data['ms1'] = pd.read_parquet(ms1_file,columns=ms1_cols)
            print(f"Loaded MS1 data: {len(data['ms1'])} records")
        else:
            data['ms1'] = None
            print("No MS1 data loaded")
        
        # Load MS2 data
        ms2_data = {}
        if ms2_deconv_file and Path(ms2_deconv_file).exists():
            ms2_data['deconvoluted'] = pd.read_parquet(ms2_deconv_file, columns=ms2_cols_deconvoluted)
            # filter by score threshold and match threshold
            print('Filtering MS2 deconvoluted data by score and match thresholds')
            print(f"Initial MS2 deconvoluted data records: {len(ms2_data['deconvoluted'])}")
            print(f"Score threshold: {self.config.ms2_support_score_threshold}, Match threshold: {self.config.ms2_support_match_threshold}")
            ms2_data['deconvoluted'] = ms2_data['deconvoluted'][
                (ms2_data['deconvoluted']['score_deconvoluted_match'] >= self.config.ms2_support_score_threshold) &
                (ms2_data['deconvoluted']['matches_deconvoluted_match'] >= self.config.ms2_support_match_threshold)
            ]
            print(f"Filtered MS2 deconvoluted data records: {len(ms2_data['deconvoluted'])}")
            print(f"Loaded MS2 deconvoluted data: {len(ms2_data['deconvoluted'])} records")
            
        if ms2_original_file and Path(ms2_original_file).exists():
            ms2_data['original'] = pd.read_parquet(ms2_original_file, columns=ms2_cols_original)
            # filter by score threshold and match threshold
            print('Filtering MS2 original data by score and match thresholds')
            print(f"Initial MS2 original data records: {len(ms2_data['original'])}")
            print(f"Score threshold: {self.config.ms2_support_score_threshold}, Match threshold: {self.config.ms2_support_match_threshold}")
            ms2_data['original'] = ms2_data['original'][
                (ms2_data['original']['score_original_match'] >= self.config.ms2_support_score_threshold) &
                (ms2_data['original']['matches_original_match'] >= self.config.ms2_support_match_threshold)
            ]
            print(f"Filtered MS2 original data records: {len(ms2_data['original'])}")
            print(f"Loaded MS2 original data: {len(ms2_data['original'])} records")
            
        data['ms2'] = ms2_data if ms2_data else None
        
        # Load file metadata
        data['metadata'] = self._load_file_metadata(file_metadata)
        
        return data
    
    def _load_file_metadata(self, metadata_source: Optional[Union[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        """Load file metadata from various sources."""
        if metadata_source is None:
            return None
            
        if isinstance(metadata_source, pd.DataFrame):
            return metadata_source
            
        if isinstance(metadata_source, str):
            if metadata_source.endswith('.csv'):
                return pd.read_csv(metadata_source)
            else:
                # Assume it's a Google Sheets config
                try:
                    return self._load_from_google_sheets()
                except Exception as e:
                    print(f"Error loading Google Sheets metadata: {e}")
                    return None
        
        return None
    
    def _load_from_google_sheets(self) -> pd.DataFrame:
        """Load file metadata from Google Sheets (default config)."""
        # Load file data
        file_df = get_google_sheet(
            notebook_name='Supplementary Tables',
            sheet_name='Table 1a'
        )
        
        # Clean headers
        file_df.columns = file_df.iloc[0]
        file_df = file_df[1:]
        
        # Load environmental data
        envo_name = get_google_sheet(
            notebook_name='Supplementary Tables',
            sheet_name='Table 1b'
        )
        envo_name.columns = envo_name.iloc[0]
        envo_name = envo_name[1:]
        
        # Merge environmental data
        file_df = pd.merge(
            file_df,
            envo_name[['name', 'id', 'common parent name']],
            left_on='environmental_subclass',
            right_on='id',
            how='inner'
        )
        
        # Clean file paths and create consistent naming
        file_df['lcmsrun_observed'] = file_df['h5'].str.replace('.h5', '')
        file_df['lcmsrun_observed'] = file_df['lcmsrun_observed'].str.replace(
            '/global/cfs/cdirs/metatlas/projects/carbon_network/raw_data/', ''
        )
        
        # Rename columns for consistency
        file_df.rename(columns={
            'common parent name': 'environment',
            'name': 'specific_environment'
        }, inplace=True)
        
        return file_df
    
    def prepare_analysis_data(self, 
                            ms1_data: pd.DataFrame,
                            ms2_data: Optional[Dict[str, pd.DataFrame]],
                            file_metadata: Optional[pd.DataFrame],
                            require_ms2_support: bool = False) -> pd.DataFrame:
        """
        Prepare integrated data for analysis.
        
        Args:
            ms1_data: MS1 annotation data
            ms2_data: MS2 annotation data (dict with 'deconvoluted' and 'original')
            file_metadata: File metadata
            require_ms2_support: Whether to require MS2 support
            
        Returns:
            pd.DataFrame: Prepared analysis data
        """
        # Start with MS1 data
        analysis_data = ms1_data.copy()
        
        # Filter by data points
        analysis_data = analysis_data[
            analysis_data['num_datapoints'] >= self.config.min_ms1_datapoints
        ]
        
        # Ensure original_index is int
        analysis_data['original_index'] = analysis_data['original_index'].astype(int)
        
        # Clean file names
        analysis_data = self._clean_file_names(analysis_data)
        # Filter by MS2 support if required
        if require_ms2_support and ms2_data:
            analysis_data = self._filter_by_ms2_support(analysis_data, ms2_data)
        # file_metadata.rename(columns={'h5': 'lcmsrun_observed'}, inplace=True)
        print('analysis columns:', analysis_data.columns.tolist())
        print('file metadata columns:', file_metadata.columns.tolist())
        file_metadata['filename'] = file_metadata['h5'].str.replace('.h5', '')
        print('Analysis filename: ', analysis_data.loc[0,'lcmsrun_observed'])
        print('File metadata filename: ', file_metadata.loc[0,'filename'])
        # analysis_data.rename(columns={'lcmsrun_observed': 'filename'}, inplace=True)
        if 'lcmsrun_observed' in file_metadata.columns:
            file_metadata.drop(columns=['lcmsrun_observed'], inplace=True)
        if 'filename' in analysis_data.columns:
            analysis_data.drop(columns=['filename'], inplace=True)
        # Merge with file metadata
        if file_metadata is not None:
            analysis_data = pd.merge(
                analysis_data, file_metadata,
                left_on='lcmsrun_observed',
                right_on='filename',
                how='left'
            )
        # clean up columns and keep lcmsrun_observed only
        if 'lcmsrun_observed' in analysis_data.columns:
            drop_cols = ['parquet','h5','filename']
            for col in drop_cols:
                if col in analysis_data.columns:
                    analysis_data.drop(columns=[col], inplace=True)
        return analysis_data
    
    def _clean_file_names(self, data: pd.DataFrame,
                          filename_column: str='lcmsrun_observed') -> pd.DataFrame:
        """Clean file names for consistency."""
        data = data.copy()
        unique_lcmsruns = data[filename_column].unique()
        # clean unique and merge back
        clean_lcmsruns = [name.replace('.h5', '').replace('.mzML', '') for name in unique_lcmsruns]
        clean_lcmsruns = [name.replace('.parquet','') for name in clean_lcmsruns]
        # if original file is deconvoluted parquet, remove _deconvoluted from name
        # only remove _deconvoluted suffix at the end
        clean_lcmsruns = [name[:-13] if name.endswith('_deconvoluted') else name for name in clean_lcmsruns]
        lcmsrun_mapping = dict(zip(unique_lcmsruns, clean_lcmsruns))
        data[filename_column] = data[filename_column].map(lcmsrun_mapping)
        
        return data
    
    def _filter_by_ms2_support(self, ms1_data: pd.DataFrame, 
                              ms2_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Filter MS1 data to only include features with MS2 support."""
        # Collect MS2 matches
        matches_list = []
        
        for spectrum_type, ms2_df in ms2_data.items():
            if ms2_df is not None and len(ms2_df) > 0:
                # Extract relevant columns
                if spectrum_type == 'deconvoluted':
                    cols = ['original_index', 'filename']
                    # if 'ref_deconvoluted_match' in ms2_df.columns:
                        # Add original_index from ref column
                    ms2_df = ms2_df.copy()
                    ms2_df['original_index'] = ms2_df['original_index_deconvoluted_match']
                elif spectrum_type == 'original':
                    cols = ['original_index', 'filename'] 
                    # if 'ref_original_match' in ms2_df.columns:
                    ms2_df = ms2_df.copy()
                    ms2_df['original_index'] = ms2_df['original_index_original_match']

                if all(col in ms2_df.columns for col in cols):
                    matches_list.append(ms2_df[cols])
        
        if not matches_list:
            print("Warning: No MS2 matches found, returning original MS1 data")
            return ms1_data
            
        # Combine and deduplicate matches
        all_matches = pd.concat(matches_list, ignore_index=True)
        all_matches.drop_duplicates(inplace=True)
        all_matches['original_index'] = all_matches['original_index'].astype(int)
        all_matches = self._clean_file_names(all_matches, filename_column='filename')

        # Filter MS1 data
        original_count = len(ms1_data)
        filtered_data = pd.merge(
            ms1_data, all_matches,
            left_on=['original_index', 'lcmsrun_observed'],
            right_on=['original_index', 'filename'],
            how='inner'
        )

        print(f"MS2 filtering: {original_count} -> {len(filtered_data)} features")
        return filtered_data