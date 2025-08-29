"""
Data loading functionality for ENVnet build pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import multiprocessing
from pathlib import Path
import os

from ..config.build_config import BuildConfig
from ..vendor.google_sheets import get_google_sheet
from ..vendor.feature_tools import group_consecutive, get_atlas_data_from_file


class SpectraLoader:
    """Handles loading and preprocessing of deconvoluted spectra data."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def load_file_metadata(self, source: str = "google_sheets") -> pd.DataFrame:
        """Load file metadata from Google Sheets or other sources."""
        if source == "google_sheets":
            file_df = get_google_sheet(notebook_name='Supplementary Tables', sheet_name='Table 1a')
            # Clean up the dataframe
            file_df.columns = file_df.iloc[0]
            file_df = file_df[1:]

            file_df['parquet'] = file_df['parquet'].apply(lambda x: x.replace('.parquet', '_deconvoluted.parquet') if not '_deconvoluted' in x else x)

            # if 'h5' not in file_df.columns:
                # file_df['h5'] = file_df['parquet'].str.replace('.parquet', '.h5')
                
            # Load environmental class information
            envo_name = get_google_sheet(notebook_name='Supplementary Tables', sheet_name='Table 1b')
            envo_name.columns = envo_name.iloc[0]
            envo_name = envo_name[1:]
            
            file_df = pd.merge(file_df, envo_name[['name', 'id', 'common parent name']], 
                             left_on='environmental_subclass', right_on='id', how='inner')
            
            # Filter for existing files
            cols = ['parquet', 'h5', 'environmental_subclass']
            file_df = file_df[cols]
            found_files = [f for f in file_df['parquet'].tolist() if os.path.exists(f)]
            file_df = file_df[file_df['parquet'].isin(found_files)]
            print('temporarily dumping files. remove this line when happy')
            print(file_df.head()['parquet'])
            file_df.to_csv('my_files.csv', index=False)
            return file_df
        else:
            raise ValueError(f"Unsupported file source: {source}")
    
    def load_single_parquet(self, parquet_file: str) -> Optional[pd.DataFrame]:
        """Load a single parquet file."""
        try:
            temp = pd.read_parquet(parquet_file)
            return temp if temp.shape[0] > 0 else None
        except Exception as e:
            print(f"Error loading {parquet_file}: {e}")
            return None

    def load_all_spectra(self, max_files: Optional[int] = None, file_source: str = "google_sheets") -> pd.DataFrame:
        """Load all deconvoluted spectra from multiple files."""
        # Get file list
        file_df = self.load_file_metadata(file_source)
        files = file_df[pd.notna(file_df['parquet'])]['parquet'].tolist()
        if max_files is not None:
            files = files[:max_files]
        # Load files in parallel
        with multiprocessing.Pool(20) as pool:
            results = pool.map(self.load_single_parquet, files)
        
        # Combine results
        all_spectra = [r for r in results if r is not None]
        all_spectra = pd.concat(all_spectra, ignore_index=True)
        
        # Preprocessing
        all_spectra = self._preprocess_spectra(all_spectra)
        
        return all_spectra
    
    def _preprocess_spectra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess loaded spectra data."""
        # Sort and index
        df.sort_values('precursor_mz', inplace=True, ascending=True)
        df.reset_index(inplace=True, drop=True)
        df.index.name = 'original_index'
        df.reset_index(inplace=True, drop=False)
        df['original_index'] = df['original_index'].astype(int)
        
        # Calculate theoretical masses and errors
        unique_formulas = df['predicted_formula'].unique()
        masses = {f: self._calculate_mass(f) for f in unique_formulas}
        df['predicted_mass'] = df['predicted_formula'].map(masses)
        df['predicted_mass'] = df['predicted_mass'] - 1.007276  # Remove proton for negative mode
        df['mass_error'] = abs(df['precursor_mz'] - df['predicted_mass'])
        
        # Verify MS1 evidence
        df = self._verify_ms1_evidence(df)
        
        # Filter by retention time
        df = df[df['rt_peak'] > self.config.min_rt]
        
        return df
    
    def _calculate_mass(self, formula: str) -> float:
        """Calculate molecular mass from formula."""
        import re
        from rdkit.Chem import rdchem
        
        pattern = r'([A-Z][a-z]*)(\d*)'
        mass = 0
        pt = rdchem.GetPeriodicTable()
        
        for el, count in re.findall(pattern, formula):
            count = int(count) if count else 1
            mass += pt.GetMostCommonIsotopeMass(el) * count
            
        return mass
    
    def _verify_ms1_evidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify MS1 evidence for deconvoluted spectra."""
        # This implements the MS1 verification logic from the original code
        # Group by filename and process in parallel
        cols = ['original_index', 'precursor_mz', 'rt']
        groups = [(_, gg[cols]) for _, gg in df.groupby('filename')]
        
        with multiprocessing.Pool(20) as pool:
            results = pool.map(self._calculate_ms1_summary_for_group, groups)
        
        ms1_data = pd.concat(results)
        cols = ['num_datapoints', 'peak_area', 'peak_height', 'mz_centroid', 'rt_peak']
        ms1_data = ms1_data[cols]
        
        # Merge back with original data
        df = pd.merge(df, ms1_data, left_index=True, right_index=True, 
                     how='inner', suffixes=('', '_ms1'))
        
        return df
    
    def _calculate_ms1_summary_for_group(self, group_data) -> pd.DataFrame:
        """Calculate MS1 summary for a single file group."""
        filename, atlas_data = group_data
        atlas = self._make_atlas(atlas_data)
        
        try:
            d = get_atlas_data_from_file(filename, atlas, desired_key='ms1_neg')
            d = d.groupby('label', group_keys=True).apply(self._calculate_ms1_summary)
            d = d[d['num_datapoints'] >= self.config.min_matches]
            return d
        except Exception as e:
            print(f'Cannot read {filename}: {e}')
            return pd.DataFrame()
    
    def _make_atlas(self, df: pd.DataFrame, ppm_tolerance: int = 5, 
                   extra_rt: float = 1) -> pd.DataFrame:
        """Create atlas for feature extraction."""
        atlas = df.copy()
        atlas.rename(columns={'original_index': 'label', 'rt': 'rt_peak', 
                             'precursor_mz': 'mz'}, inplace=True)
        atlas['rt_min'] = atlas['rt_peak'] - extra_rt
        atlas['rt_max'] = atlas['rt_peak'] + extra_rt
        atlas['mz_tolerance'] = self.config.mz_tol
        atlas['ppm_tolerance'] = ppm_tolerance
        atlas['extra_time'] = 0
        atlas['group_index'] = group_consecutive(atlas['mz'].values[:], 
                                               stepsize=ppm_tolerance, do_ppm=True)
        return atlas
    
    def _calculate_ms1_summary(self, row) -> pd.Series:
        """Calculate summary properties for MS1 features."""
        d = {}
        d['num_datapoints'] = row['i'].count()
        if d['num_datapoints'] == 0:
            return pd.Series(d)
            
        d['peak_area'] = row['i'].sum()
        idx = row['i'].idxmax()
        d['peak_height'] = row.loc[idx, 'i']
        d['mz_centroid'] = sum(row['i'] * row['mz']) / d['peak_area']
        d['rt_peak'] = row.loc[idx, 'rt']
        
        return pd.Series(d)