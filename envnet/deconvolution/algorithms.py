"""Core algorithms for spectral deconvolution.

This module contains the main algorithmic components for deconvoluting
chimeric LCMS/MSMS spectra using neutral loss patterns and clustering methods.


Usage Examples:

from envnet.config.deconvolution_config import DeconvolutionConfig
from envnet.deconvolution.algorithms import DeconvolutionAlgorithms

# Initialize with config
config = DeconvolutionConfig()
algorithms = DeconvolutionAlgorithms(config)

# Add mass differences
df_with_losses = algorithms.add_mdm_dict_to_dataframe(spectral_df, config.mdm_deltas)

# Perform clustering
clustered_df = algorithms.perform_deconvolution_clustering(df_with_losses, list(config.mdm_deltas.keys()))

# Aggregate results
final_df = algorithms.aggregate_deconvoluted_spectra(clustered_df)

"""


import pandas as pd
import numpy as np
from typing import Dict, List
from ..config.deconvolution_config import DeconvolutionConfig


class DeconvolutionAlgorithms:
    """Contains core deconvolution algorithms and spectral processing.
    
    This class implements the main algorithms for breaking down chimeric spectra
    into individual components using mass defect matching (MDM) and clustering
    approaches based on neutral loss patterns.
    """
    
    def __init__(self, config: DeconvolutionConfig):
        """Initialize the algorithms with configuration parameters.
        
        Args:
            config: DeconvolutionConfig object containing algorithm parameters
        """
        self.config = config
    
    def add_mdm_dict_to_dataframe(self, my_spectra: pd.DataFrame, mdm_dict: Dict[str, float]) -> pd.DataFrame:
        """Add MDM mass differences to spectra dataframe.
        
        Creates additional columns for each neutral loss by adding the mass
        difference to each fragment m/z value. This enables identification
        of potential precursor masses based on known neutral loss patterns.
        
        Args:
            my_spectra: DataFrame containing spectral data with 'mz' column
            mdm_dict: Dictionary mapping neutral loss names to mass differences
            
        Returns:
            DataFrame with additional columns for each neutral loss calculation
        """
        temp = {}
        for k, v in mdm_dict.items():
            temp[k] = my_spectra['mz'] + v
        temp = pd.DataFrame(temp)
        
        return pd.concat([my_spectra, temp], axis=1)
    
    def perform_deconvolution_clustering(self, df: pd.DataFrame, mdm_keys: List[str]) -> pd.DataFrame:
        """Perform deconvolution clustering on the LCMS data by breaking chimeric spectra into clusters.
        
        This method transforms the spectral data into a format suitable for clustering,
        then applies the moving z-score clustering algorithm to identify groups of
        fragments that likely originate from the same precursor ion.
        
        Args:
            df: DataFrame containing spectral data for a single retention time
            mdm_keys: List of neutral loss column names to consider
            
        Returns:
            DataFrame with cluster assignments and processed data
        """
        df.index.name = 'ion_index'
        df.reset_index(inplace=True, drop=False)
        
        # Prepare columns for melting (exclude MDM columns)
        cols = [c for c in df.columns if c not in mdm_keys]
        
        # Melt the dataframe to create one row per fragment-neutral_loss combination
        df = df.melt(id_vars=cols, value_vars=mdm_keys)
        
        # Calculate mass error between theoretical precursor and observed precursor
        df['mz_error'] = abs(df['value'] - df['precursor_mz'])
        
        # Filter by isolation tolerance
        df = df[df['mz_error'] < self.config.isolation_tolerance]
        if df.empty:
            return df
            
        # Sort by theoretical precursor mass for clustering
        df.sort_values(by=['value'], ascending=True, inplace=True)
        
        # Perform deconvolution clustering using moving z-score method
        z_clusters = self.moving_zscore_clustering(df['value'].values)
        df['cluster'] = z_clusters
        
        # Process cluster sizes and ranks
        df = self.process_cluster_sizes(df)
        
        return df
    
    def moving_zscore_clustering(self, values: np.ndarray) -> np.ndarray:
        """Detect deconvolution clusters using moving z-score method.
        
        This algorithm identifies clusters by analyzing the statistical distribution
        of mass values. It uses a moving window to calculate local statistics and
        identifies breakpoints where new clusters should begin based on z-scores.
        
        Args:
            values: Array of mass values (should be sorted)
            
        Returns:
            Array of cluster assignments for each input value
        """
        values = np.array(values)
        idx = np.argsort(values)
        values = values[idx]
        n = len(values)
        clusters = np.zeros(n, dtype=int)
        current_cluster = 0
        
        # Handle first num_points with simple threshold-based clustering
        clusters[0] = current_cluster
        for i in range(1, min(self.config.num_points, n)):
            diff = abs(values[i] - values[i-1])
            if diff > self.config.mz_tol:
                current_cluster += 1
            clusters[i] = current_cluster
        
        # Continue with z-score method for remaining points
        for i in range(self.config.num_points, n):
            # Calculate statistics from previous points in the window
            prev_points = values[i-self.config.num_points:i]
            local_mean = np.mean(prev_points)
            local_std = np.std(prev_points, ddof=1)
            d = abs(values[i] - local_mean)
            
            # Only apply z-score test if difference is above instrument precision
            # and we have sufficient variance
            if d > self.config.instrument_precision and local_std > 0: 
                z_score = d / local_std
                if z_score > self.config.z_score_threshold:
                    current_cluster += 1
            
            clusters[i] = current_cluster
        
        # Return clusters in original order
        return clusters[np.argsort(idx)]
    
    def calc_cluster_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cluster sizes and ranks, removing small clusters.
        
        Analyzes the size distribution of clusters and ranks them by size.
        Removes clusters that are too small to be reliable based on the
        minimum number of points required.
        
        Args:
            df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with cluster size information and small clusters removed
        """
        # Remove existing cluster size columns if present
        for c in ['cluster_size', 'cluster_rank']:
            if c in df.columns:
                df.drop(columns=c, inplace=True)
        
        # Calculate cluster sizes and ranks
        cluster_size = df['cluster'].value_counts().to_frame()
        cluster_size.columns = ['cluster_size']
        cluster_size = cluster_size.sort_values('cluster_size', ascending=False)
        cluster_size['cluster_rank'] = range(1, len(cluster_size) + 1)
        
        # Merge back with main dataframe
        df = pd.merge(df, cluster_size, left_on='cluster', right_index=True)
        
        # Filter out clusters that are too small
        df = df[df['cluster_size'] >= self.config.num_points]
        
        return df
    
    def process_cluster_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process deconvolution cluster sizes and remove small clusters.
        
        Performs the complete cluster size processing workflow including
        deduplication and size-based filtering.
        
        Args:
            df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with processed clusters
        """
        # Calculate initial cluster sizes
        df = self.calc_cluster_sizes(df)
        
        # Sort by cluster rank and remove duplicates (keep best cluster per ion)
        df.sort_values('cluster_rank', ascending=True, inplace=True)
        df.drop_duplicates('ion_index', inplace=True, keep='first')
        
        # Recalculate cluster sizes after deduplication
        df = self.calc_cluster_sizes(df)
        
        return df
    
    def aggregate_deconvoluted_spectra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate spectra within each deconvoluted cluster.
        
        Groups the clustered fragments and applies aggregation function
        to create representative spectra for each identified precursor.
        
        Args:
            df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with aggregated spectra for each cluster
        """
        result = df.groupby(['rt', 'cluster']).apply(
            lambda x: self.deconvoluted_spectra_agg_func(x)
        ).reset_index()
        
        return result if not result.empty else None
    
    def deconvoluted_spectra_agg_func(self, x: pd.DataFrame) -> pd.Series:
        """Aggregate deconvoluted spectra within a cluster.
        
        This function combines all fragments assigned to the same cluster
        into a single representative spectrum with calculated precursor mass
        and spectral characteristics.
        
        Args:
            x: DataFrame containing fragments from a single cluster
            
        Returns:
            Series with aggregated spectral data
        """
        x.sort_values('mz', inplace=True)
        x.reset_index(inplace=True, drop=False)
        
        d = {}
        d['count'] = x['value'].count()
        
        # Calculate intensity-weighted precursor mass
        if sum(x['i']) == 0:
            d['precursor_mz'] = 0
        else:
            d['precursor_mz'] = sum(x['i'] * x['value']) / sum(x['i'])
        
        # Create deconvoluted spectrum as m/z, intensity array
        d['deconvoluted_spectrum'] = np.asarray([x['mz'], x['i']])

        # Create split deconvoluted spectrum data columns for downstream tasks
        d['deconvoluted_spectrum_mz_vals'] = np.asarray(x['mz'])
        d['deconvoluted_spectrum_intensity_vals'] = np.asarray(x['i'])

        # Calculate spectral statistics
        d['sum_frag_intensity'] = sum(x['i'])
        d['max_frag_intensity'] = max(x['i'])
        
        # Store neutral loss information and original precursor
        d['obs'] = x['variable'].values
        d['isolated_precursor_mz'] = x['precursor_mz'].values[0]
        
        return pd.Series(d, index=d.keys())
    
    def add_mass_difference_data(self, df: pd.DataFrame, deltas: Dict[str, float]) -> pd.DataFrame:
        """Add mass difference columns to dataframe.
        
        Convenience method that wraps add_mdm_dict_to_dataframe for
        consistency with the main workflow.
        
        Args:
            df: Input spectral DataFrame
            deltas: Dictionary of mass differences
            
        Returns:
            DataFrame with mass difference columns added
        """
        return self.add_mdm_dict_to_dataframe(df, deltas)