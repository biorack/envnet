"""
deconvolution_tools.py contains two pieces for preprocessing lcms runs
1. deconvolution: this uses a list of defined neutral losses to make a
deconvoluted spectrum and calcualte an accurate precursor mz from 
chimeric spectra.  this capability is DOM specific and not a general 
implementation
2. MS-BUDDY: this is a companion tool to predict the formula from the
deonvoluted spectrum.

Outputing a parquet file with the original spectra, deconvoluted spectra,
and formula predictions is the purpose of these functions.  Typically
one will run this on every mzml file in their data warehouse so you can
analyze the parquet files for DOM characteristics with ENVnet more quickly.

"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import sys
from pathlib import Path

# Find the project root (where your config/ directory is)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Adjust if your script is in a subdirectory
print(project_root)
# Add to path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# # Add module paths - CORRECTED VERSION
# file_path = Path(__file__).resolve()

# # Find the envnet root directory by looking for the 'data' folder
# current_path = file_path.parent
# envnet_root = None

# # Walk up the directory tree to find envnet root
# for parent in [current_path] + list(current_path.parents):
#     potential_data_dir = parent / 'data'
#     potential_mdm_file = potential_data_dir / 'mdm_neutral_losses.csv'
#     if potential_mdm_file.exists():
#         envnet_root = parent
#         break

# if envnet_root is None:
#     # Fallback: assume we're in envnet/envnet/build/ and go up 3 levels
#     envnet_root = file_path.parents[3] if len(file_path.parents) > 3 else file_path.parent

from config.deconvolution_params import DeconvolutionConfig
from config.buddy_params import BuddyAnalysisConfig
params = DeconvolutionConfig()
# module_path = str(envnet_root)
sys.path.insert(0, params.module_path)
sys.path.insert(1, os.path.join(params.module_path, 'metatlas'))
sys.path.insert(2, os.path.join(params.module_path, 'blink'))
# mdm_deltas_table_filename = os.path.join(params.module_path, 'data', 'mdm_neutral_losses.csv')

# Set MDM deltas file path

# # Verify the file exists
# if not os.path.exists(mdm_deltas_table_filename):
#     print(f"Warning: MDM deltas file not found at {mdm_deltas_table_filename}")
#     print(f"Searched from: {file_path}")
#     print(f"Envnet root determined as: {envnet_root}")
#     mdm_deltas_table_filename = None

from metatlas.io import feature_tools as ft






class LCMSDeconvolutionTools:
    """Main class for LCMS spectral deconvolution processing."""
    
    def __init__(self, config: DeconvolutionConfig):
        self.config = config
        
    def deconvolute_lcms_file(self, file_path: str, deltas: Optional[Dict[str, float]] = None) -> Optional[pd.DataFrame]:
        """
        Main deconvolution entry point.
        
        Args:
            file_path: Path to LCMS data file
            deltas: Optional mass differences for neutral loss analysis. 
                   If None, uses config.mdm_deltas
            
        Returns:
            Deconvoluted DataFrame or None if no data
        """
        # Use config deltas if none provided
        if deltas is None:
            deltas = self.config.mdm_deltas
            
        # Load and preprocess spectral data
        df = self._build_deconvoluted_dataframe(file_path, deltas)
        if df is None:
            print('There is not any data in the dataframe')
            return None
            
        # Clean up final dataframe
        df = self._cleanup_final_dataframe(df)
        # # Add coisolation information
        df = self._add_coisolation_data(df)     
        
        return df
    
    def _build_deconvoluted_dataframe(self, file_path: str, deltas: Dict[str, float]) -> Optional[pd.DataFrame]:
        """Build the main deconvoluted dataframe from file and deltas."""
        # Load raw LCMS data
        ms2_df = self._load_lcms_data(file_path)
        
        # Filter by retention time
        ms2_df = self._filter_by_retention_time(ms2_df)
        
        # Process mass difference data
        ms2_df = self._add_mass_difference_data(ms2_df, deltas)
        
        # Perform deconvolution clustering
        delta_keys = list(deltas.keys())

        ms2_df = ms2_df.groupby('rt').apply(
            lambda x: self._perform_deconvolution_clustering(x, delta_keys)
        )
        ms2_df.reset_index(inplace=True, drop=True)

        
        # # # # Aggregate clustered spectra
        ms2_df = self._aggregate_deconvoluted_spectra(ms2_df)
        if ms2_df is None:
            return None
            
        # # # Add file metadata and merge with original spectra
        ms2_df = self._add_file_metadata_and_merge_original(ms2_df, file_path)
        
        return ms2_df
    
    def _load_lcms_data(self, file_path: str) -> pd.DataFrame:
        """Load LCMS data from file based on file extension."""
        if file_path.endswith('h5'):
            ms2_df = pd.read_hdf(file_path, self.config.file_key)
        elif file_path.endswith(('mzML', 'mzml')):
            ms2_df = ft.df_container_from_mzml_file(file_path, self.config.file_key)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        ms2_df.columns = [c.lower() for c in ms2_df.columns]
        return ms2_df
    
    def _filter_by_retention_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by retention time range."""
        return df[(df['rt'] > self.config.min_rt) & (df['rt'] < self.config.max_rt)]
    
    def _add_mass_difference_data(self, df: pd.DataFrame, deltas: Dict[str, float]) -> pd.DataFrame:
        """Add mass difference columns to dataframe."""
        return self._add_mdm_dict_to_dataframe(df, deltas)
    
    def _aggregate_deconvoluted_spectra(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Aggregate spectra within each deconvoluted cluster."""
        
        result = df.groupby(['rt','cluster']).apply(lambda x: self._deconvoluted_spectra_agg_func(x)).reset_index()
        return result if not result.empty else None
    
    def _add_file_metadata_and_merge_original(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Add file metadata and merge with original spectra."""
        df['filename'] = file_path
        
        # Get and merge original spectra
        orig_spectra = self._get_original_spectra(file_path)
        df, orig_spectra = self._set_rt_precision(df, orig_spectra)
        
        df = pd.merge(df, orig_spectra.add_prefix('original_'), 
                     left_on=['filename', 'rt'], 
                     right_on=['original_filename', 'original_rt'], 
                     how='left')
        
        # Clean up merge columns
        cols_to_drop = ['original_rt', 'original_precursor_mz', 'original_precursor_intensity', 'original_filename']
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Add basename
        df['basename'] = df['filename'].apply(lambda x: os.path.basename(x))
        
        return df

    def _set_rt_precision(self, df: pd.DataFrame, orig_spectra: pd.DataFrame, rt_precision: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Set retention time precision for merging."""
        for data in [df, orig_spectra]:
            data['rt'] = data['rt'].astype(float).round(rt_precision)
        return df, orig_spectra
    
    def _add_coisolation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add coisolation information to dataframe."""
        g_cols = ['basename', 'rt']
        df.reset_index(inplace=True, drop=True)
        df.index.name = 'temp_index'
        df.reset_index(inplace=True, drop=False)
        grouped = df.groupby(g_cols).agg({
            'temp_index': 'count', 
            'precursor_mz': lambda x: list(x)
        })
        grouped.rename(columns={
            'precursor_mz': 'coisolated_precursor_mz_list',
            'temp_index': 'coisolated_precursor_count'
        }, inplace=True)
        grouped.reset_index(inplace=True)
        return pd.merge(df, grouped, on=g_cols, how='left')
    
    def _cleanup_final_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up the final dataframe by removing temporary columns."""
        cols_to_remove = ['temp_index', 'cluster']
        # 'count', 'sum_frag_intensity', 
        #     'max_frag_intensity', 'obs', 'original_spectrum', 'basename', 
        #     'coisolated_precursor_mz_list'
        # ]
        
        # # Only drop columns that actually exist
        cols_to_remove = [c for c in cols_to_remove if c in df.columns]
        df.drop(columns=cols_to_remove, inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        return df

    # Atomic helper functions
    
    def _add_mdm_dict_to_dataframe(self, my_spectra: pd.DataFrame, mdm_dict: Dict[str, float]) -> pd.DataFrame:
        """Add MDM mass differences to spectra dataframe."""
        temp = {}
        for k, v in mdm_dict.items():
            temp[k] = my_spectra['mz'] + v
        temp = pd.DataFrame(temp)
        
        return pd.concat([my_spectra, temp], axis=1)


    def _perform_deconvolution_clustering(self, df: pd.DataFrame, mdm_keys: List[str]) -> pd.DataFrame:
        """Perform deconvolution clustering on the LCMS data by breaking chimeric spectra into clusters."""
        df.index.name = 'ion_index'
        df.reset_index(inplace=True, drop=False)
        
        # cols = ['ion_index', 'mz', 'i', 'rt', 'polarity', 'precursor_mz', 'precursor_intensity', 'collision_energy']
        cols = [c for c in df.columns if c not in mdm_keys]
        df = df.melt(id_vars=cols, value_vars=mdm_keys)
        df['mz_error'] = abs(df['value'] - df['precursor_mz'])
        df = df[df['mz_error'] < self.config.isolation_tolerance]
        if df.empty:
            return df
        df.sort_values(by=['value'], ascending=True, inplace=True)
        # Perform deconvolution clustering
        z_clusters = self._moving_zscore_clustering(df['value'].values)
        df['cluster'] = z_clusters
        
        # Process cluster sizes and ranks
        df = self._process_cluster_sizes(df)
        
        return df

    def _moving_zscore_clustering(self, values: np.ndarray) -> np.ndarray:
        """Detect deconvolution clusters using moving z-score method."""
        values = np.array(values)
        idx = np.argsort(values)
        values = values[idx]
        n = len(values)
        clusters = np.zeros(n, dtype=int)
        current_cluster = 0
        
        # Handle first num_points with simple threshold
        clusters[0] = current_cluster
        for i in range(1, min(self.config.num_points, n)):
            diff = abs(values[i] - values[i-1])
            if diff > self.config.mz_tolerance:
                current_cluster += 1
            clusters[i] = current_cluster
        
        # Continue with z-score method for remaining points
        for i in range(self.config.num_points, n):
            prev_points = values[i-self.config.num_points:i]
            local_mean = np.mean(prev_points)
            local_std = np.std(prev_points, ddof=1)
            d = abs(values[i] - local_mean)
            
            if d > self.config.instrument_precision and local_std > 0: 
                z_score = d / local_std
                if z_score > self.config.z_score_threshold:
                    current_cluster += 1
            
            clusters[i] = current_cluster
            
        return clusters[idx]

    def _calc_cluster_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cluster sizes and ranks, removing small clusters."""
        for c in ['cluster_size', 'cluster_rank']:
            if c in df.columns:
                df.drop(columns=c, inplace=True)
        cluster_size = df['cluster'].value_counts().to_frame()
        cluster_size.columns = ['cluster_size']
        cluster_size = cluster_size.sort_values('cluster_size', ascending=False)
        cluster_size['cluster_rank'] = range(1, len(cluster_size) + 1)
        df = pd.merge(df, cluster_size, left_on='cluster', right_index=True)
        df = df[df['cluster_size'] >= self.config.num_points]
        return df

    def _process_cluster_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process deconvolution cluster sizes and remove small clusters."""
        df = self._calc_cluster_sizes(df)
        df.sort_values('cluster_rank', ascending=True, inplace=True)
        df.drop_duplicates('ion_index', inplace=True, keep='first')
        # Recalculate after deduplication
        df = self._calc_cluster_sizes(df)
        
        return df

    def _deconvoluted_spectra_agg_func(self, x: pd.DataFrame) -> pd.Series:
        """Aggregate deconvoluted spectra within a cluster."""

        x.sort_values('mz',inplace=True)
        x.reset_index(inplace=True, drop=False)
        d = {}
        d['count'] = x['value'].count()
        if sum(x['i']) == 0:
            d['precursor_mz'] = 0
        else:
            d['precursor_mz'] = sum(x['i'] * x['value']) / sum(x['i'])
        d['deconvoluted_spectrum'] = np.asarray([x['mz'], x['i']])
        d['sum_frag_intensity'] = sum(x['i'])
        d['max_frag_intensity'] = max(x['i'])
        d['obs'] = x['variable'].values
        d['isolated_precursor_mz'] = x['precursor_mz'].values[0]
        
        # currently, rt is used as a grouping term so you get this back when reset
        #d['rt'] = x['rt'].to_list()[0]
        
        return pd.Series(d, index=d.keys())

    def _get_original_spectra(self, file_path: str) -> pd.DataFrame:
        """Load and process original spectra from file."""
        if file_path.endswith('h5'):
            ref = pd.read_hdf(file_path, self.config.file_key)
        elif file_path.endswith(('mzML', 'mzml')):
            ref = ft.df_container_from_mzml_file(file_path, self.config.file_key)
            
        ref.columns = [c.lower() for c in ref.columns]
        ref.drop(columns=['collision_energy'], inplace=True)
        ref = ref[pd.notna(ref['precursor_mz'])]
        
        ref = self._group_duplicates(ref, 'rt')
        ref['precursor_intensity'] = ref['precursor_intensity'].apply(lambda x: x[0])
        ref['precursor_mz'] = ref['precursor_mz'].apply(lambda x: x[0])
        ref['spectrum'] = ref.apply(lambda x: np.asarray([x['mz'], x['i']]), axis=1)
        ref['spectrum'] = ref['spectrum'].apply(
            lambda x: self._filter_spectra_by_percent(x, self.config.filter_percent)
        )
        
        ref.reset_index(inplace=True, drop=True)
        drop_cols = [c for c in ref.columns if c in ['mz', 'i', 'polarity']]
        ref.drop(columns=drop_cols, inplace=True)
        ref['filename'] = file_path
        
        return ref

    def _group_duplicates(self, df: pd.DataFrame, group_col: str, rt_precision: int = 6) -> pd.DataFrame:
        """Group duplicate entries by specified column."""
        precision = {'i': 3, 'mz': 5, 'rt': rt_precision}
        
        all_cols = np.asarray(df.columns)
        idx_group = np.argwhere(all_cols == group_col).flatten()
        idx_list = np.argwhere(all_cols != group_col).flatten()
        cols = all_cols[idx_list]

        a = df.sort_values(group_col).values.T
        ukeys, index = np.unique(a[idx_group, :], return_index=True)
        arrays = np.split(a[idx_list, :], index[1:], axis=1)
        ucpds = [dict([(c, aa) for c, aa in zip(cols, a)]) for a in arrays]

        df2 = pd.DataFrame(ucpds, index=ukeys)
        df2.index = df2.index.set_names(group_col)
        df2.reset_index(inplace=True)
        
        return df2

    def _filter_spectra_by_percent(self, x: np.ndarray, p: float = 0.001) -> np.ndarray:
        """Filter spectra by intensity percentage threshold."""
        max_intensity = np.max(x[1])
        threshold = p * max_intensity
        above_threshold_indices = np.where(x[1] > threshold)[0]
        return x[:, above_threshold_indices]


class MSBuddyAnalyzer:
    """Standalone class for MS-Buddy molecular formula analysis."""
    
    def __init__(self, config: BuddyAnalysisConfig):
        # Import MS-Buddy dependencies only when needed
        try:
            from msbuddy import Msbuddy, MsbuddyConfig
            from msbuddy.base import MetaFeature, Spectrum
            self.Msbuddy = Msbuddy
            self.MsbuddyConfig = MsbuddyConfig
            self.MetaFeature = MetaFeature
            self.Spectrum = Spectrum
        except ImportError as e:
            raise ImportError(f"MS-Buddy dependencies not available: {e}. Install with: pip install msbuddy")
            
        self.config = config
        self.msb_engine = None
        
    def analyze_spectra(self, df: pd.DataFrame, spectrum_key: str = 'deconvoluted_spectrum') -> pd.DataFrame:
        """
        Run MS-Buddy analysis on spectral data.
        
        Args:
            df: DataFrame containing spectral data
            spectrum_key: Column name containing spectral arrays
            
        Returns:
            DataFrame with MS-Buddy analysis results merged
        """
        # Run buddy analysis
        result, self.msb_engine = self._run_buddy_analysis(df, spectrum_key)
        
        # Process buddy results
        result = self._process_buddy_results(result)
        
        # Merge with main dataframe
        df_with_buddy = pd.merge(df, result.drop(columns=['mz', 'rt']), 
                                left_index=True, right_on='identifier', how='inner')
        df_with_buddy.drop(columns=['identifier'], inplace=True)
        
        return df_with_buddy
    
    def _run_buddy_analysis(self, mgf_df: pd.DataFrame, spectrum_key: str) -> Tuple[pd.DataFrame, 'Msbuddy']:
        """Run MS-Buddy analysis on spectra."""
        msb_config = self.MsbuddyConfig(
            ppm=self.config.ppm,
            ms1_tol=self.config.ms1_tol,
            ms2_tol=self.config.ms2_tol,
            rel_int_denoise_cutoff=self.config.rel_int_denoise_cutoff,
            max_frag_reserved=self.config.max_frag_reserved,
            parallel=self.config.parallel,
            n_cpu=self.config.n_cpu,
            halogen=self.config.halogen,
            batch_size=self.config.batch_size
        )

        msb_engine = self.Msbuddy(msb_config)
        msb_engine.data = mgf_df.apply(
            lambda row: self._make_buddy_spec(row, spectrum_key), axis=1
        )

        msb_engine.annotate_formula()
        result = msb_engine.get_summary()
        r = pd.DataFrame(result)
        r = r[r['estimated_fdr'] < self.config.max_fdr]
        
        return r, msb_engine

    def _make_buddy_spec(self, row: pd.Series, spectrum_key: str) -> 'MetaFeature':
        """Create MSBuddy MetaFeature from row data."""
        return self.MetaFeature(
            mz=row['precursor_mz'],
            charge=-1 if self.config.polarity == 'negative' else 1,
            rt=0,
            adduct='[M-H]-' if self.config.polarity == 'negative' else '[M+H]+',
            ms2=self.Spectrum(row[spectrum_key][0], row[spectrum_key][1]) if row[spectrum_key].shape[1] > 0 else None,
            identifier=row.name
        )
    
    def _process_buddy_results(self, result: pd.DataFrame) -> pd.DataFrame:
        """Process and clean buddy analysis results."""
        result.rename(columns={
            'adduct': 'assumed_adduct',
            'formula_rank_1': 'predicted_formula'
        }, inplace=True)
        
        # Remove rank columns
        rank_cols = [c for c in result.columns if 'rank_' in c]
        result.drop(columns=rank_cols, inplace=True)
        
        return result


class LCMSWorkflow:
    """Combined workflow class that integrates deconvolution and MS-Buddy analysis."""
    
    def __init__(self, 
                 deconv_config: Optional[DeconvolutionConfig] = None,
                 buddy_config: Optional[BuddyAnalysisConfig] = None,
                 do_buddy: bool = True):
        self.deconv_config = deconv_config or DeconvolutionConfig()
        self.buddy_config = buddy_config or BuddyAnalysisConfig()
        self.do_buddy = do_buddy
        
        self.deconvolution_tools = LCMSDeconvolutionTools(self.deconv_config)
        if self.do_buddy:
            self.buddy_analyzer = MSBuddyAnalyzer(self.buddy_config)
    
    def run_full_workflow(self, file_path: str, deltas: Optional[Dict[str, float]] = None) -> Optional[pd.DataFrame]:
        """
        Run complete LCMS workflow with deconvolution and optional MS-Buddy analysis.
        
        Args:
            file_path: Path to LCMS data file
            deltas: Optional mass differences for neutral loss analysis
            
        Returns:
            Processed DataFrame with deconvolution and optional MS-Buddy results
        """
        # Run deconvolution
        df = self.deconvolution_tools.deconvolute_lcms_file(file_path, deltas)
        if df is None:
            return None
            
        # Run MS-Buddy analysis if enabled
        if self.do_buddy:
            df = self.buddy_analyzer.analyze_spectra(df, spectrum_key='deconvoluted_spectrum')
            
        return df


# Convenience functions
def deconvolute_lcms_file(file_path: str, deltas: Optional[Dict[str, float]] = None, 
                         config: Optional[DeconvolutionConfig] = None) -> Optional[pd.DataFrame]:
    """Convenience function to run just deconvolution with default or custom config."""
    if config is None:
        config = DeconvolutionConfig()

    deconv_tools = LCMSDeconvolutionTools(config)
    return deconv_tools.deconvolute_lcms_file(file_path, deltas)


def run_lcms_workflow_with_defaults(file_path: str, 
                                   deconv_config: Optional[DeconvolutionConfig] = None,
                                   buddy_config: Optional[BuddyAnalysisConfig] = None,
                                   do_buddy: bool = True) -> Optional[pd.DataFrame]:
    """Run complete LCMS workflow using default MDM deltas from config."""
    workflow = LCMSWorkflow(deconv_config, buddy_config, do_buddy)
    return workflow.run_full_workflow(file_path)


def run_lcms_workflow(file_path: str, 
                     deltas: Optional[Dict[str, float]] = None,
                     deconv_config: Optional[DeconvolutionConfig] = None,
                     buddy_config: Optional[BuddyAnalysisConfig] = None,
                     do_buddy: bool = True) -> Optional[pd.DataFrame]:
    """Convenience function to run complete workflow with custom parameters."""
    workflow = LCMSWorkflow(deconv_config, buddy_config, do_buddy)
    return workflow.run_full_workflow(file_path, deltas)
# ===================================================================
# USAGE EXAMPLES
# ===================================================================

"""
Example 1: Basic deconvolution only (no MS-Buddy)
"""
# from deconvolution_tools import deconvolute_lcms_file, DeconvolutionConfig
# 
# # Use default settings
# df = deconvolute_lcms_file('/path/to/your/file.h5')
# 
# # Or with custom config
# config = DeconvolutionConfig(
#     mz_tolerance=0.005,
#     max_rt=10.0,
#     isolation_tolerance=0.8
# )
# df = deconvolute_lcms_file('/path/to/your/file.h5', config=config)

"""
Example 2: Full workflow with MS-Buddy analysis
"""
# from deconvolution_tools import LCMSWorkflow, DeconvolutionConfig, BuddyAnalysisConfig
# 
# # Use default settings for both deconvolution and MS-Buddy
# workflow = LCMSWorkflow(do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.h5')
# 
# # Custom configurations
# deconv_config = DeconvolutionConfig(
#     mz_tolerance=0.005,
#     max_rt=8.0,
#     min_rt=0.5
# )
# 
# buddy_config = BuddyAnalysisConfig(
#     polarity='positive',
#     max_fdr=0.01,
#     n_cpu=20
# )
# 
# workflow = LCMSWorkflow(deconv_config, buddy_config, do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.h5')

"""
Example 3: Using convenience functions
"""
# from deconvolution_tools import run_lcms_workflow_with_defaults, run_lcms_workflow
# 
# # Quick start with all defaults
# df = run_lcms_workflow_with_defaults('/path/to/your/file.h5', do_buddy=True)
# 
# # With custom deltas and parameters
# custom_deltas = {'loss_H2O': -18.010565, 'loss_CO2': -43.989829}
# df = run_lcms_workflow(
#     '/path/to/your/file.h5',
#     deltas=custom_deltas,
#     do_buddy=True
# )

"""
Example 4: Standalone MS-Buddy analysis on existing deconvoluted data
"""
# from deconvolution_tools import MSBuddyAnalyzer, BuddyAnalysisConfig
# import pandas as pd
# 
# # Load previously deconvoluted data
# df = pd.read_parquet('/path/to/deconvoluted_data.parquet')
# 
# # Run MS-Buddy analysis
# buddy_config = BuddyAnalysisConfig(polarity='negative', max_fdr=0.05)
# buddy_analyzer = MSBuddyAnalyzer(buddy_config)
# df_with_formulas = buddy_analyzer.analyze_spectra(df, spectrum_key='deconvoluted_spectrum')

"""
Example 5: Batch processing multiple files
"""
# from deconvolution_tools import LCMSWorkflow
# import os
# from pathlib import Path
# 
# # Setup workflow once
# workflow = LCMSWorkflow(do_buddy=True)
# 
# # Process multiple files
# input_dir = Path('/path/to/input/files')
# output_dir = Path('/path/to/output')
# output_dir.mkdir(exist_ok=True)
# 
# for file_path in input_dir.glob('*.h5'):
#     try:
#         df = workflow.run_full_workflow(str(file_path))
#         if df is not None:
#             output_file = output_dir / f"{file_path.stem}_deconvoluted.parquet"
#             df.to_parquet(output_file)
#             print(f"Processed: {file_path.name} -> {output_file.name}")
#         else:
#             print(f"No data found in: {file_path.name}")
#     except Exception as e:
#         print(f"Error processing {file_path.name}: {e}")

"""
Example 6: Working with mzML files
"""
# from deconvolution_tools import DeconvolutionConfig, LCMSWorkflow
# 
# # For mzML files, you might need different file_key
# config = DeconvolutionConfig(file_key='ms2_neg')  # for negative mode
# workflow = LCMSWorkflow(deconv_config=config, do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.mzML')

"""
Example 7: Accessing intermediate results and engine objects
"""
# from deconvolution_tools import LCMSWorkflow
# 
# workflow = LCMSWorkflow(do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.h5')
# 
# # Access the MS-Buddy engine for additional analysis
# if workflow.do_buddy and hasattr(workflow.buddy_analyzer, 'msb_engine'):
#     msb_engine = workflow.buddy_analyzer.msb_engine
#     # Use msb_engine for additional MS-Buddy operations
#     # msb_engine.plot_summary()  # if such methods exist

"""
Example 8: Custom mass difference deltas
"""
# from deconvolution_tools import LCMSWorkflow
# 
# # Define custom neutral losses for your specific analysis
# custom_deltas = {
#     'loss_H2O': -18.010565,
#     'loss_NH3': -17.026549,
#     'loss_CO2': -43.989829,
#     'loss_CH2O2': -46.005479,
#     'custom_loss_123': -123.045678
# }
# 
# workflow = LCMSWorkflow(do_buddy=False)  # deconvolution only
# df = workflow.run_full_workflow('/path/to/your/file.h5', deltas=custom_deltas)

"""
Example 9: Error handling and logging
"""
# from deconvolution_tools import LCMSWorkflow
# import logging
# 
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# 
# workflow = LCMSWorkflow(do_buddy=True)
# 
# try:
#     df = workflow.run_full_workflow('/path/to/your/file.h5')
#     if df is not None:
#         logger.info(f"Successfully processed file. Found {len(df)} deconvoluted features.")
#         # Save or further process df
#     else:
#         logger.warning("No data found in file.")
# except Exception as e:
#     logger.error(f"Processing failed: {e}")
#     raise

"""
Example 10: Memory-efficient processing for large files
"""
# from deconvolution_tools import LCMSDeconvolutionTools, MSBuddyAnalyzer
# from deconvolution_tools import DeconvolutionConfig, BuddyAnalysisConfig
# 
# # Process in steps to manage memory
# deconv_config = DeconvolutionConfig(batch_size=5000)  # if such parameter existed
# buddy_config = BuddyAnalysisConfig(batch_size=1000, n_cpu=10)
# 
# # Step 1: Deconvolution only
# deconv_tools = LCMSDeconvolutionTools(deconv_config)
# df = deconv_tools.deconvolute_lcms_file('/path/to/large_file.h5')
# 
# # Save intermediate result
# df.to_parquet('/tmp/deconvoluted_intermediate.parquet')
# del df  # Free memory
# 
# # Step 2: MS-Buddy analysis on smaller chunks if needed
# # df = pd.read_parquet('/tmp/deconvoluted_intermediate.parquet')
# # buddy_analyzer = MSBuddyAnalyzer(buddy_config)
# # df_final = buddy_analyzer.analyze_spectra(df)


# Command-line interface
def main(args):
    """Main function for command-line execution."""
    
    # Create configuration objects from command-line arguments
    deconv_config = DeconvolutionConfig(
        mz_tolerance=args.mz_tol,
        isolation_tolerance=args.isolation_tol,
        max_rt=args.max_rt,
        min_rt=args.min_rt,
        similarity_cutoff=args.similarity_cutoff,
        min_intensity_ratio=args.min_intensity_ratio,
        filter_percent=args.filter_percent,
        z_score_threshold=args.z_score_threshold,
        num_points=args.num_points,
        instrument_precision=args.instrument_precision,
        file_key=args.file_key
    )
    
    buddy_config = BuddyAnalysisConfig(
        polarity=args.polarity,
        max_fdr=args.max_fdr,
        ms1_tol=args.ms1_tol,
        ms2_tol=args.ms2_tol,
        max_frag_reserved=args.max_frag_reserved,
        rel_int_denoise_cutoff=args.rel_int_denoise_cutoff,
        n_cpu=args.n_cpu,
        ppm=args.ppm,
        halogen=args.halogen,
        parallel=args.parallel,
        batch_size=args.batch_size
    )
    
    # Determine output file name
    if args.output:
        out_file = args.output
    else:
        # Auto-generate output filename
        base, ext = os.path.splitext(args.file)
        out_file = f"{base}_deconvoluted.parquet"
    
    print(f"Processing: {args.file}")
    print(f"Output: {out_file}")
    
    try:
        # Run the workflow
        workflow = LCMSWorkflow(deconv_config, buddy_config, args.do_buddy)
        df = workflow.run_full_workflow(args.file)
        
        if df is not None:
            # Save results
            os.makedirs(os.path.dirname(out_file), exist_ok=True) if os.path.dirname(out_file) else None
            cols = [c for c in df.columns if c.endswith('_spectrum')]
            for my_col in cols:
                df[f'{my_col}_mz_vals'] = df[my_col].apply(lambda x: x[0])
                df[f'{my_col}_intensity_vals'] = df[my_col].apply(lambda x: x[1])
            df.drop(columns=cols,inplace=True)

            df.to_parquet(out_file)
            print(f"Successfully processed {args.file} -> {out_file}")
            print(f"Output contains {len(df)} deconvoluted features")
        else:
            print(f"No data found in {args.file}")
            # Create empty failure file
            failure_file = f"{out_file}-failed"
            open(failure_file, 'w').close()
            
    except Exception as e:
        print(f"Error processing {args.file}: {e}")
        # Create failure marker file
        failure_file = f"{out_file}-failed"
        open(failure_file, 'w').close()
        raise


if __name__ == "__main__":
    description_string = (
        "Run LCMS deconvolution with optional MS-Buddy analysis on H5/mzML files.\n\n"
        "Examples:\n"
        "  # Basic deconvolution only\n"
        "  python deconvolution_tools.py my_file.h5\n\n"
        "  # With MS-Buddy analysis\n"
        "  python deconvolution_tools.py --do_buddy my_file.h5\n\n"
        "  # Custom parameters\n"
        "  python deconvolution_tools.py --mz_tol 0.005 --max_rt 10 --do_buddy --polarity positive my_file.h5\n\n"
        "  # Specify output file\n"
        "  python deconvolution_tools.py --output /path/to/results.parquet --do_buddy my_file.h5\n"
    )
    parser = argparse.ArgumentParser(
        prog='LCMS Deconvolution of DOM Chimeric Spectra and MS-Buddy Analysis',
        description=description_string,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('file', type=str, help='Input H5 or mzML file path')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output parquet file path (auto-generated if not specified)')
    
    # Deconvolution parameters
    parser.add_argument('--mz_tol', type=float, default=0.01, help='m/z tolerance for clustering')
    parser.add_argument('--isolation_tol', type=float, default=0.5, help='Isolation tolerance for precursor matching')
    parser.add_argument('--similarity_cutoff', type=float, default=0.8, help='Similarity cutoff for spectral comparison')
    parser.add_argument('--min_intensity_ratio', type=float, default=2.0, help='Minimum intensity ratio threshold')
    parser.add_argument('--max_rt', type=float, default=7.0, help='Maximum retention time for filtering')
    parser.add_argument('--min_rt', type=float, default=1.0, help='Minimum retention time for filtering')
    parser.add_argument('--filter_percent', type=float, default=0.0, help='Intensity filter percentage threshold')
    
    # Clustering parameters
    parser.add_argument('--z_score_threshold', type=float, default=2.5, help='Z-score threshold for clustering')
    parser.add_argument('--num_points', type=int, default=3, help='Number of points for moving z-score calculation')
    parser.add_argument('--instrument_precision', type=float, default=0.001, help='Instrument precision threshold')
    
    # File parameters
    parser.add_argument('--file_key', type=str, default='ms2_neg', help='HDF5 key for reading data')
    
    # MS-Buddy options
    parser.add_argument('--do_buddy', action='store_true', 
                       help='Run MS-Buddy molecular formula prediction')
    parser.add_argument('--polarity', type=str, default='negative', 
                       choices=['positive', 'negative'], 
                       help='Ionization polarity')
    parser.add_argument('--max_fdr', type=float, default=0.05, help='Maximum FDR for MS-Buddy results')
    parser.add_argument('--ms1_tol', type=float, default=0.001, help='MS1 tolerance for MS-Buddy')
    parser.add_argument('--ms2_tol', type=float, default=0.002, help='MS2 tolerance for MS-Buddy')
    parser.add_argument('--max_frag_reserved', type=int, default=50, help='Maximum fragments reserved for MS-Buddy')
    parser.add_argument('--rel_int_denoise_cutoff', type=float, default=0.01, 
                       help='Relative intensity denoising cutoff for MS-Buddy')
    parser.add_argument('--n_cpu', type=int, default=40, help='Number of CPUs for MS-Buddy parallel processing')
    parser.add_argument('--ppm', action='store_true', help='Use PPM for MS-Buddy tolerances')
    parser.add_argument('--halogen', action='store_true', help='Include halogen elements in MS-Buddy')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing in MS-Buddy')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for MS-Buddy processing')
    
    args = parser.parse_args()
    
    # Handle parallel flag (convert --no_parallel to parallel=False)
    args.parallel = not args.no_parallel
    
    main(args)