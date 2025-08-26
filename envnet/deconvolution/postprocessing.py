"""Post-processing functions for deconvolution results.

This module handles data enrichment and cleanup operations that occur
after the main deconvolution algorithms have been applied.
"""

import pandas as pd
import os
from ..config.deconvolution_config import DeconvolutionConfig


class DeconvolutionPostProcessor:
    """Handles post-processing of deconvoluted spectral data.
    
    This class contains methods for enriching and cleaning up the
    results after deconvolution has been performed.
    """
    
    def __init__(self, config: DeconvolutionConfig):
        """Initialize the post-processor with configuration.
        
        Args:
            config: DeconvolutionConfig object containing parameters
        """
        self.config = config
    
    def add_coisolation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add coisolation information to dataframe.
        
        Identifies cases where multiple precursors were co-isolated
        in the same spectrum by grouping by basename and retention time.
        
        Args:
            df: DataFrame with deconvoluted results
            
        Returns:
            DataFrame with coisolation information added
        """
        g_cols = ['basename', 'rt']
        df.reset_index(inplace=True, drop=True)
        df.index.name = 'temp_index'
        df.reset_index(inplace=True, drop=False)
        
        # Group by file and retention time to find co-isolated precursors
        grouped = df.groupby(g_cols).agg({
            'temp_index': 'count', 
            'precursor_mz': lambda x: list(x)
        })
        
        # Rename columns to be descriptive
        grouped.rename(columns={
            'precursor_mz': 'coisolated_precursor_mz_list',
            'temp_index': 'coisolated_precursor_count'
        }, inplace=True)
        grouped.reset_index(inplace=True)
        
        # Merge back with main dataframe
        return pd.merge(df, grouped, on=g_cols, how='left')
    
    def cleanup_final_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up the final dataframe by removing temporary columns.
        
        Removes intermediate columns that were used during processing
        but are not needed in the final results.
        
        Args:
            df: DataFrame to clean up
            
        Returns:
            Cleaned DataFrame with temporary columns removed
        """
        cols_to_remove = ['temp_index', 'cluster']
        
        # Only drop columns that actually exist
        cols_to_remove = [c for c in cols_to_remove if c in df.columns]
        df.drop(columns=cols_to_remove, inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        return df


# Convenience functions for standalone use
def add_coisolation_info(df: pd.DataFrame, config: DeconvolutionConfig) -> pd.DataFrame:
    """Convenience function to add coisolation information.
    
    Args:
        df: DataFrame with deconvoluted results
        config: DeconvolutionConfig object
        
    Returns:
        DataFrame with coisolation data added
    """
    processor = DeconvolutionPostProcessor(config)
    return processor.add_coisolation_data(df)


def cleanup_dataframe(df: pd.DataFrame, config: DeconvolutionConfig) -> pd.DataFrame:
    """Convenience function to clean up final dataframe.
    
    Args:
        df: DataFrame to clean up
        config: DeconvolutionConfig object
        
    Returns:
        Cleaned DataFrame
    """
    processor = DeconvolutionPostProcessor(config)
    return processor.cleanup_final_dataframe(df)