"""Data preprocessing functions for LCMS deconvolution.

This module handles all data loading, filtering, and basic preprocessing
operations for LCMS mass spectrometry data before deconvolution analysis.

Usage Examples:

from envnet.config.deconvolution_config import DeconvolutionConfig
from envnet.deconvolution.preprocessing import LCMSDataProcessor

# Initialize with config
config = DeconvolutionConfig()
processor = LCMSDataProcessor(config)

# Load and preprocess data
df = processor.load_lcms_data('my_file.h5')
df = processor.filter_by_retention_time(df)

# Or use convenience function
from envnet.deconvolution.preprocessing import preprocess_lcms_data
df = preprocess_lcms_data('my_file.h5', config)


"""

import pandas as pd
import numpy as np
import os
from typing import Tuple
from pathlib import Path

# Import from vendored feature_tools
try:
    from ..vendor import feature_tools as ft
except ImportError:
    print("Warning: vendor.feature_tools not available. mzML file support may be limited.")
    ft = None

from ..config.deconvolution_config import DeconvolutionConfig


class LCMSDataProcessor:
    """Handles LCMS data loading and preprocessing operations.
    
    This class encapsulates all the data loading, filtering, and basic
    preprocessing operations needed before running deconvolution algorithms.
    """
    
    def __init__(self, config: DeconvolutionConfig):
        """Initialize the data processor with configuration.
        
        Args:
            config: DeconvolutionConfig object containing processing parameters
        """
        self.config = config
    
    def load_lcms_data(self, file_path: str) -> pd.DataFrame:
        """Load LCMS data from file based on file extension.
        
        Supports both HDF5 (.h5) and mzML formats. Column names are
        standardized to lowercase for consistency.
        
        Args:
            file_path: Path to the LCMS data file
            
        Returns:
            DataFrame with LCMS spectral data
            
        Raises:
            ValueError: If file format is not supported
            ImportError: If metatlas is not available for mzML files
        """
        if file_path.endswith('h5'):
            ms2_df = pd.read_hdf(file_path, self.config.file_key)
        elif file_path.endswith(('mzML', 'mzml')):
            if ft is None:
                raise ImportError("vendor.feature_tools is required for mzML file support")
            ms2_df = ft.df_container_from_mzml_file(file_path, self.config.file_key)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        # Standardize column names to lowercase
        ms2_df.columns = [c.lower() for c in ms2_df.columns]
        return ms2_df
    
    def filter_by_retention_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by retention time range.
        
        Removes spectra outside the configured RT window to focus
        analysis on the chromatographic region of interest.
        
        Args:
            df: Input DataFrame with 'rt' column
            
        Returns:
            Filtered DataFrame within RT bounds
        """
        return df[(df['rt'] > self.config.min_rt) & (df['rt'] < self.config.max_rt)]
    
    def get_original_spectra(self, file_path: str) -> pd.DataFrame:
        """Load and process original spectra from file.
        
        Loads the original (non-deconvoluted) spectra for comparison
        and merging with deconvoluted results. Performs grouping by
        retention time and spectral filtering.
        
        Args:
            file_path: Path to the LCMS data file
            
        Returns:
            DataFrame with processed original spectra
        """
        # Load raw data
        if file_path.endswith('h5'):
            ref = pd.read_hdf(file_path, self.config.file_key)
        elif file_path.endswith(('mzML', 'mzml')):
            if ft is None:
                raise ImportError("vendor.feature_tools is required for mzML file support")
            ref = ft.df_container_from_mzml_file(file_path, self.config.file_key)
            
        # Standardize columns
        ref.columns = [c.lower() for c in ref.columns]
        
        # Clean up columns
        if 'collision_energy' in ref.columns:
            ref.drop(columns=['collision_energy'], inplace=True)
        ref = ref[pd.notna(ref['precursor_mz'])]
        
        # Group duplicates by retention time
        ref = self.group_duplicates(ref, 'rt')
        
        # Process precursor data - take first value from grouped lists
        ref['precursor_intensity'] = ref['precursor_intensity'].apply(lambda x: x[0])
        ref['precursor_mz'] = ref['precursor_mz'].apply(lambda x: x[0])
        
        # Create spectrum arrays and filter by intensity
        ref['spectrum'] = ref.apply(lambda x: np.asarray([x['mz'], x['i']]), axis=1)
        ref['spectrum'] = ref['spectrum'].apply(
            lambda x: self.filter_spectra_by_percent(x, self.config.filter_percent)
        )
        
        # Clean up final dataframe
        ref.reset_index(inplace=True, drop=True)
        drop_cols = [c for c in ref.columns if c in ['mz', 'i', 'polarity']]
        ref.drop(columns=drop_cols, inplace=True)
        ref['filename'] = file_path
        
        return ref
    
    def group_duplicates(self, df: pd.DataFrame, group_col: str, rt_precision: int = 6) -> pd.DataFrame:
        """Group duplicate entries by specified column.
        
        Combines multiple spectral entries that have the same retention time
        (within precision) into grouped arrays for further processing.
        
        Args:
            df: Input DataFrame to group
            group_col: Column name to group by (typically 'rt')
            rt_precision: Number of decimal places for RT rounding
            
        Returns:
            DataFrame with grouped entries as arrays
        """
        # Get column indices for grouping
        all_cols = np.asarray(df.columns)
        idx_group = np.argwhere(all_cols == group_col).flatten()
        idx_list = np.argwhere(all_cols != group_col).flatten()
        cols = all_cols[idx_list]

        # Convert to numpy array for efficient processing
        a = df.sort_values(group_col).values.T
        ukeys, index = np.unique(a[idx_group, :], return_index=True)
        arrays = np.split(a[idx_list, :], index[1:], axis=1)
        
        # Create grouped compound dictionaries
        ucpds = [dict([(c, aa) for c, aa in zip(cols, a)]) for a in arrays]

        # Convert back to DataFrame
        df2 = pd.DataFrame(ucpds, index=ukeys)
        df2.index = df2.index.set_names(group_col)
        df2.reset_index(inplace=True)
        
        return df2
    
    def filter_spectra_by_percent(self, x: np.ndarray, p: float = 0.001) -> np.ndarray:
        """Filter spectra by intensity percentage threshold.
        
        Removes low-intensity peaks that are below a percentage of the
        maximum intensity to reduce noise and focus on significant signals.
        
        Args:
            x: Spectral array with shape (2, n) where x[0] is m/z, x[1] is intensity
            p: Percentage threshold (0.001 = 0.1%)
            
        Returns:
            Filtered spectral array with only peaks above threshold
        """
        if x.size == 0 or x.shape[1] == 0:
            return x
            
        max_intensity = np.max(x[1])
        threshold = p * max_intensity
        above_threshold_indices = np.where(x[1] > threshold)[0]
        return x[:, above_threshold_indices]
    
    def set_rt_precision(self, df: pd.DataFrame, orig_spectra: pd.DataFrame, 
                        rt_precision: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Set retention time precision for merging.
        
        Rounds retention times to specified precision to ensure proper
        matching between deconvoluted and original spectra datasets.
        
        Args:
            df: Deconvoluted spectra DataFrame
            orig_spectra: Original spectra DataFrame  
            rt_precision: Number of decimal places for RT rounding
            
        Returns:
            Tuple of (df, orig_spectra) with standardized RT precision
        """
        for data in [df, orig_spectra]:
            data['rt'] = data['rt'].astype(float).round(rt_precision)
        return df, orig_spectra
    
    def add_file_metadata_and_merge_original(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Add file metadata and merge with original spectra.
        
        This method combines deconvoluted results with original spectral data
        and adds file metadata for tracking and analysis.
        
        Args:
            df: Deconvoluted spectra DataFrame
            file_path: Path to the original data file
            
        Returns:
            DataFrame with merged deconvoluted and original spectra data
        """
        df['filename'] = file_path
        
        # Get and merge original spectra
        orig_spectra = self.get_original_spectra(file_path)
        df, orig_spectra = self.set_rt_precision(df, orig_spectra)
        
        # Merge datasets
        df = pd.merge(df, orig_spectra.add_prefix('original_'), 
                     left_on=['filename', 'rt'], 
                     right_on=['original_filename', 'original_rt'], 
                     how='left')
        
        # Clean up merge columns
        cols_to_drop = ['original_rt', 'original_precursor_mz', 
                       'original_precursor_intensity', 'original_filename']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Add basename for easier identification
        df['basename'] = df['filename'].apply(lambda x: os.path.basename(x))
        
        return df


# Convenience functions for standalone use
def load_lcms_file(file_path: str, config: DeconvolutionConfig) -> pd.DataFrame:
    """Convenience function to load LCMS data with configuration.
    
    Args:
        file_path: Path to LCMS data file
        config: DeconvolutionConfig object
        
    Returns:
        Loaded and standardized DataFrame
    """
    processor = LCMSDataProcessor(config)
    return processor.load_lcms_data(file_path)


def preprocess_lcms_data(file_path: str, config: DeconvolutionConfig) -> pd.DataFrame:
    """Convenience function to perform complete preprocessing.
    
    Loads data and applies standard preprocessing steps including
    RT filtering and column standardization.
    
    Args:
        file_path: Path to LCMS data file
        config: DeconvolutionConfig object
        
    Returns:
        Preprocessed DataFrame ready for deconvolution
    """
    processor = LCMSDataProcessor(config)
    df = processor.load_lcms_data(file_path)
    df = processor.filter_by_retention_time(df)
    return df