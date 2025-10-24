"""Validation framework for spectral deconvolution algorithms.

This module provides comprehensive validation of deconvolution performance using
synthetic chimeric spectra created from P2D2 reference data. It measures both
precursor recovery and fragment ion assignment accuracy.

Run Complete Validation
cd ~/repos/envnet
PYTHONPATH=pwd 
python -m envnet.deconvolution.validation


Custom validation examples

from envnet.deconvolution.validation import DeconvolutionValidator
from envnet.config import DeconvolutionConfig, BuildConfig

# Custom configuration
deconv_config = DeconvolutionConfig(
    z_score_threshold=2.5,
    num_points=4,
    isolation_tolerance=0.75
)

# Run validation
validator = DeconvolutionValidator(deconv_config)
results_df, summary = validator.run_comprehensive_validation(max_groups=50)

print(f"Fragment Precision: {summary['overall_fragment_precision']:.4f}")
print(f"Fragment Recall: {summary['overall_fragment_recall']:.4f}")


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Configure matplotlib for vector-based PDF text (not rasterized)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts (searchable text)
matplotlib.rcParams['ps.fonttype'] = 42   # Also for PostScript output
# Use matplotlib's default font instead of specifying one that might not exist
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']#,'Geneva', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .core import LCMSDeconvolutionTools
from ..config.deconvolution_config import DeconvolutionConfig
from ..config.build_config import BuildConfig


class DeconvolutionValidator:
    """Validates deconvolution algorithms using synthetic chimeric spectra.
    
    Creates synthetic chimeric spectra from P2D2 reference data and measures
    how well the deconvolution algorithm recovers original precursors and
    correctly assigns fragment ions.
    """
    
    def __init__(self, 
                 deconv_config: Optional[DeconvolutionConfig] = None,
                 build_config: Optional[BuildConfig] = None):
        """Initialize validator with configuration objects.
        
        Args:
            deconv_config: Configuration for deconvolution parameters
            build_config: Configuration containing P2D2 reference data
        """
        self.deconv_config = deconv_config or DeconvolutionConfig()
        self.build_config = build_config or BuildConfig()
        self.deconv_tools = LCMSDeconvolutionTools(self.deconv_config)
        
        # Load reference data from build config
        self.ref_data = None
        self.nl_refs = None
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load P2D2 reference data from build config."""
        try:
            # Get reference data directly from build config
            self.ref_data = self.build_config.ref
            self.nl_refs = self.build_config.ref2
            
            if self.ref_data is None or self.nl_refs is None:
                raise ValueError("Reference data not loaded in BuildConfig")
            
            # Clean and prepare the neutral loss data (similar to original code)
            self.nl_refs = self.nl_refs.copy()
            self.nl_refs.drop_duplicates('formula', inplace=True)
            self.nl_refs = self.nl_refs[self.nl_refs['precursor_mz'] > 0]
            
            # Filter by number of ions
            num_ions = self.nl_refs['nl_spectrum'].apply(lambda x: x.shape[1])
            self.nl_refs['num_ions'] = num_ions
            self.nl_refs = self.nl_refs[self.nl_refs['num_ions'] > 3]
            self.nl_refs.reset_index(inplace=True, drop=True)
            
            print(f"Loaded {len(self.ref_data)} reference spectra")
            print(f"Loaded {len(self.nl_refs)} neutral loss reference spectra")
            
        except Exception as e:
            print(f"Error loading reference data: {e}")
            print("Make sure BuildConfig is properly initialized")
            self.ref_data = None
            self.nl_refs = None
    
    def make_merged_spectra(self, 
                           df: pd.DataFrame, 
                           intensity_threshold: float = 0.01,
                           spectrum_key: str = 'spectrum',
                           identifier_key: str = 'original_p2d2_index') -> Optional[np.ndarray]:
        """Create merged chimeric spectra from reference spectra.
        
        Combines multiple reference spectra into a single chimeric spectrum
        for validation testing.
        
        Args:
            df: DataFrame containing reference spectra
            intensity_threshold: Minimum intensity threshold for peak inclusion
            spectrum_key: Column name containing spectral data
            identifier_key: Column name for spectrum identification
            
        Returns:
            Array containing merged spectral data or None if empty
        """
        merged_spectra = []
        
        for i, row in df.iterrows():
            mz = row[spectrum_key][0]
            intensity = row[spectrum_key][1]
            
            if len(intensity) == 0 or len(mz) == 0:
                continue
                
            # Create identifier and precursor arrays
            my_identifier = np.array([row[identifier_key]] * len(mz), dtype=object)
            my_precursor = np.array([row['precursor_mz']] * len(mz))
            
            # Normalize intensity and apply threshold
            intensity = intensity / intensity.max()
            idx = np.argwhere(intensity > intensity_threshold).flatten()
            
            # Filter arrays
            mz = mz[idx]
            intensity = intensity[idx]
            my_identifier = my_identifier[idx]
            my_precursor = my_precursor[idx]
            
            # Stack into single array
            s = np.vstack((my_identifier, mz, intensity, my_precursor))
            merged_spectra.append(s)
        
        if len(merged_spectra) == 0:
            return None
            
        return np.hstack(merged_spectra)
    
    def setup_validation_references(self, 
                                   refs: List[pd.DataFrame], 
                                   noise_filter: float = 0.01,
                                   spectrum_key: str = 'spectrum', 
                                   identifier_key: str = 'original_p2d2_index') -> pd.DataFrame:
        """Setup DataFrame for validation from mixed reference spectra.
        
        Creates synthetic chimeric spectra by combining reference spectra
        and formats them for deconvolution testing.
        
        Args:
            refs: List of reference DataFrames
            noise_filter: Intensity threshold for noise filtering
            spectrum_key: Column containing spectral data
            identifier_key: Column containing spectrum identifiers
            
        Returns:
            DataFrame formatted for deconvolution validation
        """
        my_spectra = []
        counter = 0
        
        for r in refs:
            if r.shape[0] == 0:
                continue
                
            s = self.make_merged_spectra(
                r, 
                intensity_threshold=noise_filter,
                spectrum_key=spectrum_key,
                identifier_key=identifier_key
            )
            
            if s is None:
                continue
                
            # Convert to DataFrame
            s = pd.DataFrame(s.T, columns=[identifier_key, 'mz', 'i', 'precursor_mz'])
            
            # Add synthetic metadata
            s['isolated_precursor_mz'] = r['precursor_mz'].mean()  # Simulate wide isolation window
            s['rt'] = counter
            my_spectra.append(s)
            counter += 1
        
        if not my_spectra:
            return pd.DataFrame()
            
        my_spectra = pd.concat(my_spectra)
        my_spectra.reset_index(inplace=True, drop=True)
        return my_spectra
    
    def chunk_refs_by_precursor(self, ref: pd.DataFrame) -> List[pd.DataFrame]:
        """Chunk reference DataFrame by precursor m/z for validation.
        
        Groups reference spectra by rounded precursor m/z to create
        realistic chimeric spectrum scenarios.
        
        Args:
            ref: Reference DataFrame containing spectral data
            
        Returns:
            List of DataFrames grouped by precursor mass
        """
        ref = ref.copy()
        ref['round_mz'] = ref['precursor_mz'].round(0)
        g = ref.groupby('round_mz')
        refs = [group for name, group in g]
        
        # Sort by number of rows and filter
        refs = sorted(refs, key=lambda x: x.shape[0], reverse=True)
        refs = [r for r in refs if r.shape[0] > 1]
        
        return refs
    
    def calc_precursor_precision_recall(self, 
                                       original_df: pd.DataFrame, 
                                       deconvoluted_df: pd.DataFrame,
                                       ref_group: pd.DataFrame,
                                       mz_tolerance: float = 0.01) -> Dict:
        """Calculate precision and recall for precursor and fragment recovery.
        
        Measures both precursor-level and fragment-level accuracy by comparing
        original vs deconvoluted assignments using m/z matching.
        
        Args:
            original_df: Original synthetic chimeric spectra
            deconvoluted_df: Results from deconvolution algorithm
            ref_group: Reference group containing original precursor m/z values
            mz_tolerance: Tolerance for m/z matching in Daltons
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        # Get original precursor m/z values (ground truth)
        original_precursor_mz = list(ref_group['precursor_mz'].unique())
        
        # Calculate m/z-based precursor metrics
        if deconvoluted_df.empty:
            deconvoluted_mz = []
        else:
            # Each cluster represents a recovered precursor - use cluster mean m/z
            deconvoluted_mz = []
            if 'cluster' in deconvoluted_df.columns:
                for cluster_id in deconvoluted_df['cluster'].unique():
                    cluster_mean_mz = deconvoluted_df[deconvoluted_df['cluster'] == cluster_id]['value'].mean()
                    deconvoluted_mz.append(cluster_mean_mz)
        
        # Match deconvoluted m/z to original m/z within tolerance
        matched_deconvoluted = []
        matched_original = []
        
        for deconv_mz in deconvoluted_mz:
            for orig_mz in original_precursor_mz:
                if abs(deconv_mz - orig_mz) <= mz_tolerance:
                    matched_deconvoluted.append(deconv_mz)
                    matched_original.append(orig_mz)
                    break  # Each deconvoluted m/z can only match one original
        
        # Calculate precision/recall metrics
        tp = len(matched_deconvoluted)  # Deconvoluted m/z that match original
        fp = len(deconvoluted_mz) - len(matched_deconvoluted)  # Deconvoluted m/z with no original match
        fn = len(original_precursor_mz) - len(matched_original)  # Original m/z with no deconvoluted match
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result = {
            "original_precursors": original_precursor_mz,
            "deconvoluted_precursors": deconvoluted_mz,
            "tp": tp, "fp": fp, "fn": fn, "tn": 0,
            "precision": precision, "recall": recall, "f1_score": f1_score
        }
        
        # Calculate fragment-level metrics (if we have deconvolution results)
        if not deconvoluted_df.empty and 'cluster' in deconvoluted_df.columns:
            # Prepare data for fragment-level analysis - use available columns
            merge_columns = ['mz']
            if 'i' in deconvoluted_df.columns:
                merge_columns.append('i')
            elif 'intensity' in deconvoluted_df.columns:
                merge_columns.append('intensity')
            
            deconv_subset = deconvoluted_df[merge_columns + ['rt', 'cluster', 'original_p2d2_index']].rename(
                columns={'rt': 'rt_deconvoluted', 'i': 'intensity', 'original_p2d2_index': 'original_p2d2_index_deconvoluted'}
            )
            
            # Match intensity column name in original_df if needed
            original_merge_cols = ['mz']
            if 'intensity' in original_df.columns and len(merge_columns) > 1:
                original_merge_cols.append('intensity')
            
            df = original_df.merge(
                deconv_subset,
                on=original_merge_cols,
                how='left'
            )
            
            # Calculate most common p2d2_index for each cluster
            g = df.groupby(['rt_deconvoluted', 'cluster'])['original_p2d2_index_deconvoluted'].apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            )
            df = pd.merge(
                df, 
                g.rename('most_common_p2d2_index'), 
                left_on=['rt_deconvoluted', 'cluster'], 
                right_index=True, 
                how='outer'
            )
            
            df['frag_correct'] = df['original_p2d2_index_deconvoluted'] == df['most_common_p2d2_index']
            df.loc[pd.isna(df['cluster']), 'frag_correct'] = None
            
            frag_tp = (df['frag_correct'] == True).sum()
            frag_fp = (df['frag_correct'] == False).sum()
            frag_fn = (df['frag_correct'].isna()).sum()
            
            frag_precision = frag_tp / (frag_tp + frag_fp) if (frag_tp + frag_fp) > 0 else 0
            frag_recall = frag_tp / (frag_tp + frag_fn) if (frag_tp + frag_fn) > 0 else 0
            frag_f1_score = 2 * (frag_precision * frag_recall) / (frag_precision + frag_recall) if (frag_precision + frag_recall) > 0 else 0
            
            result.update({
                'frag_tp': frag_tp, 'frag_fp': frag_fp, 'frag_fn': frag_fn, 'frag_tn': 0,
                'frag_precision': frag_precision, 'frag_recall': frag_recall, 'frag_f1_score': frag_f1_score
            })
        else:
            # No deconvolution results - all fragments are false negatives
            result.update({
                'frag_tp': 0, 'frag_fp': 0, 'frag_fn': len(original_df), 'frag_tn': 0,
                'frag_precision': 0, 'frag_recall': 0, 'frag_f1_score': 0
            })
        
        return result
    
    def run_single_validation(self, ref_group: pd.DataFrame) -> Optional[Dict]:
        """Run validation on a single group of reference spectra.
        
        Args:
            ref_group: DataFrame containing reference spectra to combine
            
        Returns:
            Validation metrics dictionary or None if validation failed
        """
        if ref_group.shape[0] < 2:
            return None
        
        try:
            # Create synthetic chimeric spectrum
            original_df = self.setup_validation_references(
                [ref_group], 
                noise_filter=0.0, 
                spectrum_key='nl_spectrum',
                identifier_key='original_p2d2_index'
            )
            
            if original_df.empty:
                return None
            
            # Add mass difference data
            ms2_df = self.deconv_tools.algorithms.add_mass_difference_data(
                original_df, 
                self.deconv_config.mdm_deltas
            )
            
            # Perform deconvolution clustering
            delta_keys = list(self.deconv_config.mdm_deltas.keys())
            ms2_df = ms2_df.groupby('rt').apply(
                lambda x: self.deconv_tools.algorithms.perform_deconvolution_clustering(x, delta_keys)
            )
            ms2_df.reset_index(inplace=True, drop=True)
            
            if ms2_df.empty:
                return None
            
            # Calculate precision and recall
            metrics = self.calc_precursor_precision_recall(original_df, ms2_df, ref_group)
            return metrics
            
        except Exception as e:
            print(f"Validation failed for ref group: {e}")
            return None
    
    def run_comprehensive_validation(self, max_groups: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """Run comprehensive validation across multiple reference groups.
        
        Args:
            max_groups: Maximum number of reference groups to test
            
        Returns:
            Tuple of (detailed results DataFrame, summary metrics dict)
        """
        if self.nl_refs is None:
            raise ValueError("No reference data loaded. Cannot run validation.")
        
        # Get reference groups
        ref_groups = self.chunk_refs_by_precursor(self.nl_refs)
        ref_groups = ref_groups[:max_groups]  # Limit for practical runtime
        
        print(f"Running validation on {len(ref_groups)} reference groups...")
        
        validation_results = []
        for i, ref_group in enumerate(ref_groups):
            if i % 10 == 0:
                print(f"Processing group {i+1}/{len(ref_groups)}")
                
            metrics = self.run_single_validation(ref_group)
            if metrics is not None:
                validation_results.append(metrics)
        
        if not validation_results:
            raise ValueError("No successful validations completed")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(validation_results)
        
        # Calculate overall metrics
        summary_metrics = self.calculate_summary_metrics(results_df)
        
        print(f"Validation completed on {len(results_df)} groups")
        return results_df, summary_metrics
    
    def calculate_summary_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate summary metrics from validation results.
        
        Args:
            results_df: DataFrame containing individual validation results
            
        Returns:
            Dictionary of summary metrics
        """
        # Overall fragment metrics
        frag_tp = results_df['frag_tp'].sum()
        frag_fp = results_df['frag_fp'].sum()
        frag_fn = results_df['frag_fn'].sum()
        
        frag_precision = frag_tp / (frag_tp + frag_fp) if (frag_tp + frag_fp) > 0 else 0
        frag_recall = frag_tp / (frag_tp + frag_fn) if (frag_tp + frag_fn) > 0 else 0
        frag_f1 = 2 * (frag_precision * frag_recall) / (frag_precision + frag_recall) if (frag_precision + frag_recall) > 0 else 0
        
        # Overall precursor metrics (same approach as fragments)
        precursor_tp = results_df['tp'].sum()
        precursor_fp = results_df['fp'].sum()
        precursor_fn = results_df['fn'].sum()
        
        overall_precursor_precision = precursor_tp / (precursor_tp + precursor_fp) if (precursor_tp + precursor_fp) > 0 else 0
        overall_precursor_recall = precursor_tp / (precursor_tp + precursor_fn) if (precursor_tp + precursor_fn) > 0 else 0
        overall_precursor_f1 = 2 * (overall_precursor_precision * overall_precursor_recall) / (overall_precursor_precision + overall_precursor_recall) if (overall_precursor_precision + overall_precursor_recall) > 0 else 0
        
        # Average metrics (for comparison)
        avg_precursor_precision = results_df['precision'].mean()
        avg_precursor_recall = results_df['recall'].mean()
        avg_precursor_f1 = results_df['f1_score'].mean()
        
        avg_fragment_precision = results_df['frag_precision'].mean()
        avg_fragment_recall = results_df['frag_recall'].mean()
        avg_fragment_f1 = results_df['frag_f1_score'].mean()
        
        return {
            'overall_fragment_precision': frag_precision,
            'overall_fragment_recall': frag_recall,
            'overall_fragment_f1': frag_f1,
            'overall_precursor_precision': overall_precursor_precision,
            'overall_precursor_recall': overall_precursor_recall,
            'overall_precursor_f1': overall_precursor_f1,
            'avg_precursor_precision': avg_precursor_precision,
            'avg_precursor_recall': avg_precursor_recall,
            'avg_precursor_f1': avg_precursor_f1,
            'avg_fragment_precision': avg_fragment_precision,
            'avg_fragment_recall': avg_fragment_recall,
            'avg_fragment_f1': avg_fragment_f1,
            'total_groups_tested': len(results_df)
        }


class ValidationPlotter:
    """Creates visualization plots for validation results."""
    
    @staticmethod
    def plot_precision_recall_histograms(results_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create histograms of precision and recall distributions.
        
        Args:
            results_df: DataFrame containing validation results
            save_path: Optional path to save the figure
        """
        # Set larger font sizes for journal publication
        plt.rcParams.update({'font.size': 18})  # Base font size increased
        
        edges = np.linspace(0, 1, 20)
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), sharex=True, sharey=True)
        ax = ax.flatten()
        
        # Panel labels
        panel_labels = ['a', 'b', 'c', 'd']
        
        results_df['frag_precision'].hist(bins=edges, ax=ax[0], alpha=0.7)
        ax[0].set_title('Fragment Ion Assignment Precision', fontsize=20)
        ax[0].set_ylabel('Frequency', fontsize=18)
        ax[0].text(0.05, 0.95, panel_labels[0], transform=ax[0].transAxes, fontsize=24, 
                   verticalalignment='top', horizontalalignment='left')
        
        results_df['frag_recall'].hist(bins=edges, ax=ax[1], alpha=0.7)
        ax[1].set_title('Fragment Ion Assignment Recall', fontsize=20)
        ax[1].text(0.05, 0.95, panel_labels[1], transform=ax[1].transAxes, fontsize=24, 
                   verticalalignment='top', horizontalalignment='left')
        
        results_df['precision'].hist(bins=edges, ax=ax[2], alpha=0.7)
        ax[2].set_title('Precursor Ion Assignment Precision', fontsize=20)
        ax[2].set_xlabel('Score', fontsize=18)
        ax[2].set_ylabel('Frequency', fontsize=18)
        ax[2].text(0.05, 0.95, panel_labels[2], transform=ax[2].transAxes, fontsize=24, 
                   verticalalignment='top', horizontalalignment='left')
        
        results_df['recall'].hist(bins=edges, ax=ax[3], alpha=0.7)
        ax[3].set_title('Precursor Ion Assignment Recall', fontsize=20)
        ax[3].set_xlabel('Score', fontsize=18)
        ax[3].text(0.05, 0.95, panel_labels[3], transform=ax[3].transAxes, fontsize=24, 
                   verticalalignment='top', horizontalalignment='left')
        
        # Increase tick label font size
        for axis in ax:
            axis.tick_params(labelsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def make_validation_plot(deconvoluted_df: pd.DataFrame, 
                            ref_df: pd.DataFrame, 
                            ax: plt.Axes,
                            panel_label: str = None) -> plt.Figure:
        """Create validation plot showing original vs deconvoluted spectra.
        
        Args:
            deconvoluted_df: DataFrame with deconvolution results
            ref_df: DataFrame with reference spectra
            ax: Matplotlib axes to plot on
            panel_label: Optional panel label (a, b, c, etc.) for top-right corner
            
        Returns:
            Figure object
        """
        # Get deconvoluted precursor masses
        decon_vals = deconvoluted_df.groupby('cluster')['value'].mean().sort_values(ascending=True).tolist()
        
        # Plot original MS2 fragments
        x, y, s = [], [], []
        for i, row in ref_df.iterrows():
            ms2_mz = row['nl_spectrum'][0]
            ms2_intensity = row['nl_spectrum'][1]
            for j in range(len(ms2_mz)):
                x.append(ms2_mz[j])
                y.append(row['precursor_mz'])
                s.append(20 * ms2_intensity[j]**0.5)
        
        s = [min(ss, 200) for ss in s]  # Cap size
        ax.scatter(x, y, s=s, alpha=0.6, label='Original MS2', color='blue')
        
        # Plot deconvoluted precursor lines
        for val in decon_vals:
            ax.axhline(val, linestyle=':', label=f'Decon. Prec.: {val:.4f}', color='red')
        
        # Plot fragment assignments
        x, y = [], []
        for i, row in deconvoluted_df.iterrows():
            x.append(row['mz'])
            y.append(row['value'])
        
        ax.plot(x, y, 'k.', alpha=1, label='MDM recovered values')
        
        # Add panel label in top-left corner if provided
        if panel_label:
            ax.text(0.05, 0.95, panel_label, transform=ax.transAxes, fontsize=30, 
                   verticalalignment='top', horizontalalignment='left')
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylabel('Precursor m/z')
        ax.set_xlabel('Fragment m/z')
        
        return ax.figure
    
    @staticmethod
    def plot_validation_examples(validator: DeconvolutionValidator, 
                                num_examples: int = 4, 
                                save_path: Optional[str] = None):
        """Create example validation plots showing deconvolution performance.
        
        Args:
            validator: DeconvolutionValidator instance
            num_examples: Number of example plots to create
            save_path: Optional path to save the figure
        """
        ref_groups = validator.chunk_refs_by_precursor(validator.nl_refs)
        ref_groups = ref_groups[:num_examples]

        fig, ax = plt.subplots(figsize=(22, 12), ncols=2, nrows=2)
        ax = ax.flatten()
        
        # Panel labels for validation examples
        panel_labels = ['a', 'b', 'c', 'd']
        
        for i, ref_group in enumerate(ref_groups):
            # Create synthetic chimeric spectrum
            original_df = validator.setup_validation_references(
                [ref_group], 
                noise_filter=0,
                spectrum_key='nl_spectrum',
                identifier_key='original_p2d2_index'
            )
            
            # Run deconvolution
            ms2_df = validator.deconv_tools.algorithms.add_mass_difference_data(
                original_df, 
                validator.deconv_config.mdm_deltas
            )
            
            delta_keys = list(validator.deconv_config.mdm_deltas.keys())
            deconvoluted_df = ms2_df.groupby('rt').apply(
                lambda x: validator.deconv_tools.algorithms.perform_deconvolution_clustering(x, delta_keys)
            )
            deconvoluted_df.reset_index(inplace=True, drop=True)
            
            # Create plot with panel label
            panel_label = panel_labels[i] if i < len(panel_labels) else None
            ValidationPlotter.make_validation_plot(deconvoluted_df, ref_group, ax[i], panel_label)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main execution functions
def main():
    """Main validation execution function."""
    print("Starting LCMS Deconvolution Validation...")
    
    # Import necessary modules
    from pathlib import Path
    from ..build.reference import load_p2d2_reference_data
    from ..config import DeconvolutionConfig, BuildConfig
    
    # Initialize validator with proper configuration (like in your notebook)
    print("Initializing validator with BuildConfig...")
    
    # Set up paths
    PYTHONPATH = "/global/homes/b/bpb/repos/envnet"
    
    # 1. Load the MDM neutral loss data required by the p2d2 loader
    print("Loading MDM neutral loss data...")
    mdm_path = Path(PYTHONPATH) / "envnet" / "data" / "mdm_neutral_losses.csv"
    mdm_df = pd.read_csv(mdm_path)

    # 2. Load the P2D2 reference library using the correct function
    print("Loading P2D2 reference spectral library...")
    ref_df, _ = load_p2d2_reference_data(deltas=mdm_df)

    # 3. Manually populate the BuildConfig with the loaded P2D2 spectra
    spectra_path = Path(PYTHONPATH) / "scripts" / "build_files.csv"
    build_config = BuildConfig(file_metadata_path=str(spectra_path))
    build_config.ref_spec = ref_df['spectrum'].tolist()
    build_config.ref_pmz = ref_df['precursor_mz'].tolist()
    ref_df = ref_df.reset_index().rename(columns={'index': 'ref_spec_index'})
    build_config.ref_nodes = ref_df

    # 4. Configure deconvolution parameters
    deconv_config = DeconvolutionConfig(
        z_score_threshold=2,
        num_points=3,
        isolation_tolerance=0.75
    )

    # 5. Initialize validator with proper configs
    validator = DeconvolutionValidator(deconv_config=deconv_config, build_config=build_config)
    
    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run comprehensive validation
    print("Running comprehensive validation...")
    results_df, summary_metrics = validator.run_comprehensive_validation(max_groups=50)
    
    # Save results
    results_path = output_dir / "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    # Print summary metrics
    print("\n=== VALIDATION SUMMARY ===")
    for metric, value in summary_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Create and save plots
    print("\nGenerating validation plots...")
    
    # Precision/recall histograms
    ValidationPlotter.plot_precision_recall_histograms(
        results_df, 
        save_path=output_dir / "precision_recall_histograms.pdf"
    )
    
    # Example validation plots
    ValidationPlotter.plot_validation_examples(
        validator,
        num_examples=4,
        save_path=output_dir / "validation_examples.pdf"
    )
    
    # Save summary metrics
    summary_path = output_dir / "summary_metrics.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Summary metrics saved to: {summary_path}")
    
    print(f"\nValidation complete! Results saved in: {output_dir}")
    

if __name__ == "__main__":
    main()