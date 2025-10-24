"""
Data preprocessing utilities for annotation workflows.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from typing import Dict

from ..config.annotation_config import AnnotationConfig


class AnnotationPreprocessor:
    """Handles data preprocessing for annotation workflows."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        
    def filter_ms2_data(self, ms2_data: pd.DataFrame, 
                       node_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter MS2 data to only include spectra with precursors near ENVnet nodes.
        
        Args:
            ms2_data: Experimental MS2 data
            node_data: ENVnet node data
            
        Returns:
            pd.DataFrame: Filtered MS2 data
        """
        # Get sorted m/z values from ENVnet nodes
        envnet_mzs = node_data['precursor_mz'].sort_values().values
        
        # Filter MS2 data to reasonable m/z range
        ms2_data = ms2_data[
            (ms2_data['precursor_mz'] >= envnet_mzs.min()) &
            (ms2_data['precursor_mz'] <= envnet_mzs.max())
        ].copy()
        
        # Find nearest ENVnet m/z for each experimental spectrum
        interpolator = interpolate.interp1d(
            envnet_mzs, 
            np.arange(envnet_mzs.size),
            kind='nearest', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Get indices of nearest ENVnet m/z values
        nearest_indices = interpolator(ms2_data['precursor_mz'].values).astype(int)
        ms2_data['nearest_precursor'] = envnet_mzs[nearest_indices]
        ms2_data['mz_diff'] = abs(ms2_data['nearest_precursor'] - ms2_data['precursor_mz'])
        ms2_data['mz_diff_ppm'] = (ms2_data['mz_diff'] / ms2_data['precursor_mz']) * 1e6
        # Filter by m/z tolerance
        ms2_data = ms2_data[ms2_data['mz_diff_ppm'] < self.config.ppm_tolerance]
        ms2_data.reset_index(inplace=True, drop=True)
        
        return ms2_data
    
    def validate_ms2_spectra(self, ms2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate MS2 spectra have required attributes.
        
        Args:
            ms2_data: MS2 data to validate
            
        Returns:
            pd.DataFrame: Validated MS2 data
        """
        required_columns = [
            'precursor_mz',
            'deconvoluted_spectrum_mz_vals',
            'deconvoluted_spectrum_intensity_vals'
        ]
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in ms2_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing spectral data
        initial_count = len(ms2_data)
        ms2_data = ms2_data.dropna(subset=required_columns[:1])  # At minimum, need precursor_mz
        
        # Check for empty spectra
        valid_spectra = ms2_data.apply(self._has_valid_spectrum, axis=1)
        ms2_data = ms2_data[valid_spectra]
        
        final_count = len(ms2_data)
        if final_count < initial_count:
            print(f"Filtered {initial_count - final_count} invalid spectra "
                  f"({final_count} remaining)")
        
        return ms2_data
    
    def _has_valid_spectrum(self, row: pd.Series) -> bool:
        """Check if a spectrum has valid m/z and intensity data."""
        try:
            mz_vals = row['deconvoluted_spectrum_mz_vals']
            int_vals = row['deconvoluted_spectrum_intensity_vals']
            
            # Check if data exists and is not empty
            if pd.isna(mz_vals) or pd.isna(int_vals):
                return False
                
            # Convert to arrays if needed
            if isinstance(mz_vals, str):
                mz_vals = np.fromstring(mz_vals.strip('[]'), sep=',')
            if isinstance(int_vals, str):
                int_vals = np.fromstring(int_vals.strip('[]'), sep=',')
                
            # Check for valid data
            return len(mz_vals) > 0 and len(int_vals) > 0 and len(mz_vals) == len(int_vals)
            
        except Exception:
            return False