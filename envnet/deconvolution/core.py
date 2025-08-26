"""Core deconvolution workflows and main entry points.

Usage Examples:
from envnet.config.deconvolution_config import DeconvolutionConfig
from envnet.deconvolution import LCMSDeconvolutionTools

# Basic usage - everything is handled internally
config = DeconvolutionConfig()
deconv_tools = LCMSDeconvolutionTools(config)
df = deconv_tools.deconvolute_lcms_file('my_file.h5')

# Advanced usage - access individual components if needed
from envnet.deconvolution import LCMSDataProcessor, DeconvolutionAlgorithms, DeconvolutionPostProcessor

config = DeconvolutionConfig()
preprocessor = LCMSDataProcessor(config)
algorithms = DeconvolutionAlgorithms(config)
postprocessor = DeconvolutionPostProcessor(config)

# Step-by-step processing
df = preprocessor.load_lcms_data('my_file.h5')
df = preprocessor.filter_by_retention_time(df)
df = algorithms.add_mass_difference_data(df, config.mdm_deltas)

"""

from typing import Optional, Dict
import pandas as pd
from .preprocessing import LCMSDataProcessor  
from .algorithms import DeconvolutionAlgorithms
from .postprocessing import DeconvolutionPostProcessor
from ..config.deconvolution_config import DeconvolutionConfig
from ..config.buddy_config import BuddyConfig  # Fixed import


class LCMSDeconvolutionTools:
    """Main class for LCMS spectral deconvolution processing."""
    
    def __init__(self, config: DeconvolutionConfig):
        self.config = config
        self.preprocessor = LCMSDataProcessor(config)
        self.algorithms = DeconvolutionAlgorithms(config)
        self.postprocessor = DeconvolutionPostProcessor(config)
        
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
            
        # Build deconvoluted dataframe
        df = self._build_deconvoluted_dataframe(file_path, deltas)
        if df is None:
            print('There is not any data in the dataframe')
            return None
            
        # Clean up final dataframe
        df = self.postprocessor.cleanup_final_dataframe(df)
        
        # Add coisolation information
        df = self.postprocessor.add_coisolation_data(df)     
        
        return df
    
    def _build_deconvoluted_dataframe(self, file_path: str, deltas: Dict[str, float]) -> Optional[pd.DataFrame]:
        """Build the main deconvoluted dataframe from file and deltas."""
        
        # 1. Load and preprocess raw LCMS data
        ms2_df = self.preprocessor.load_lcms_data(file_path)
        ms2_df = self.preprocessor.filter_by_retention_time(ms2_df)
        
        # 2. Add mass difference data for neutral loss analysis
        ms2_df = self.algorithms.add_mass_difference_data(ms2_df, deltas)
        
        # 3. Perform deconvolution clustering by retention time groups
        delta_keys = list(deltas.keys())
        ms2_df = ms2_df.groupby('rt').apply(
            lambda x: self.algorithms.perform_deconvolution_clustering(x, delta_keys)
        )
        ms2_df.reset_index(inplace=True, drop=True)
        
        # 4. Aggregate clustered spectra
        ms2_df = self.algorithms.aggregate_deconvoluted_spectra(ms2_df)
        if ms2_df is None:
            return None
            
        # 5. Add file metadata and merge with original spectra
        ms2_df = self.preprocessor.add_file_metadata_and_merge_original(ms2_df, file_path)
        
        return ms2_df


class MSBuddyAnalyzer:
    """Standalone class for MS-Buddy molecular formula analysis."""
    
    def __init__(self, config: BuddyConfig):
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
    
    def _run_buddy_analysis(self, mgf_df: pd.DataFrame, spectrum_key: str):
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

    def _make_buddy_spec(self, row: pd.Series, spectrum_key: str):
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