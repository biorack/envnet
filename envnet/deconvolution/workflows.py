"""End-to-end workflow orchestration and command-line interface.

This module provides high-level workflows that combine multiple processing
steps and a command-line interface for batch processing of LCMS data files.

Usage from the command line


# Basic deconvolution
python -m envnet.deconvolution.workflows my_file.h5

# With MS-Buddy analysis
python -m envnet.deconvolution.workflows --do_buddy my_file.h5

# Custom parameters
python -m envnet.deconvolution.workflows \
    --mz_tol 0.005 \
    --max_rt 10 \
    --do_buddy \
    --polarity negative \
    --output results.parquet \
    my_file.h5


"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from .core import LCMSDeconvolutionTools, MSBuddyAnalyzer
from ..config.deconvolution_config import DeconvolutionConfig
from ..config.buddy_config import BuddyConfig

class LCMSWorkflow:
    """Combined workflow class that integrates deconvolution and MS-Buddy analysis."""
    
    def __init__(self, 
                 deconv_config: Optional[DeconvolutionConfig] = None,
                 buddy_config: Optional[BuddyConfig] = None,  # Fixed parameter type
                 do_buddy: bool = True):
        """Initialize the workflow with configuration objects."""
        self.deconv_config = deconv_config or DeconvolutionConfig()
        self.buddy_config = buddy_config or BuddyConfig()  # Fixed instantiation
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


# Convenience functions for easy access
def deconvolute_lcms_file(file_path: str, deltas: Optional[Dict[str, float]] = None, 
                         config: Optional[DeconvolutionConfig] = None) -> Optional[pd.DataFrame]:
    """Convenience function to run just deconvolution with default or custom config.
    
    Args:
        file_path: Path to LCMS data file
        deltas: Optional mass differences for neutral loss analysis
        config: Optional custom deconvolution configuration
        
    Returns:
        Deconvoluted DataFrame or None if no data
    """
    if config is None:
        config = DeconvolutionConfig()

    deconv_tools = LCMSDeconvolutionTools(config)
    return deconv_tools.deconvolute_lcms_file(file_path, deltas)


def run_lcms_workflow_with_defaults(file_path: str, 
                                   deconv_config: Optional[DeconvolutionConfig] = None,
                                   buddy_config: Optional[BuddyConfig] = None, 
                                   do_buddy: bool = True) -> Optional[pd.DataFrame]:
    """Run complete LCMS workflow using default MDM deltas from config.
    
    Args:
        file_path: Path to LCMS data file
        deconv_config: Optional custom deconvolution configuration
        buddy_config: Optional custom MS-Buddy configuration  
        do_buddy: Whether to run MS-Buddy analysis
        
    Returns:
        Processed DataFrame with results or None if no data
    """
    workflow = LCMSWorkflow(deconv_config, buddy_config, do_buddy)
    return workflow.run_full_workflow(file_path)


def run_lcms_workflow(file_path: str, 
                     deltas: Optional[Dict[str, float]] = None,
                     deconv_config: Optional[DeconvolutionConfig] = None,
                     buddy_config: Optional[BuddyConfig] = None, 
                     do_buddy: bool = True) -> Optional[pd.DataFrame]:
    """Convenience function to run complete workflow with custom parameters.
    
    Args:
        file_path: Path to LCMS data file
        deltas: Optional custom mass differences for neutral loss analysis
        deconv_config: Optional custom deconvolution configuration
        buddy_config: Optional custom MS-Buddy configuration
        do_buddy: Whether to run MS-Buddy analysis
        
    Returns:
        Processed DataFrame with results or None if no data
    """
    workflow = LCMSWorkflow(deconv_config, buddy_config, do_buddy)
    return workflow.run_full_workflow(file_path, deltas)


# Command-line interface
def main(args):
    """Main function for command-line execution."""
    
    # Create configuration objects from command-line arguments
    deconv_config = DeconvolutionConfig(
        mz_tol=args.mz_tol,
        isolation_tolerance=args.isolation_tol,
        max_rt=args.max_rt,
        min_rt=args.min_rt,
        min_library_match_score=args.min_library_match_score,
        min_intensity_ratio=args.min_intensity_ratio,
        filter_percent=args.filter_percent,
        z_score_threshold=args.z_score_threshold,
        num_points=args.num_points,
        instrument_precision=args.instrument_precision,
        file_key=args.file_key
    )
    
    buddy_config = BuddyConfig(  # Fixed class name
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
            # Save results - convert spectral arrays to separate columns for parquet
            os.makedirs(os.path.dirname(out_file), exist_ok=True) if os.path.dirname(out_file) else None
            
            # Convert spectral arrays to separate m/z and intensity columns
            cols = [c for c in df.columns if c.endswith('_spectrum')]
            for my_col in cols:
                df[f'{my_col}_mz_vals'] = df[my_col].apply(lambda x: x[0])
                df[f'{my_col}_intensity_vals'] = df[my_col].apply(lambda x: x[1])
            df.drop(columns=cols, inplace=True)

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


def setup_argparse():
    """Set up command-line argument parsing.
    
    Returns:
        Configured ArgumentParser object
    """
    description_string = (
        "Run LCMS deconvolution with optional MS-Buddy analysis on H5/mzML files.\n\n"
        "Examples:\n"
        "  # Basic deconvolution only\n"
        "  python -m envnet.deconvolution.workflows my_file.h5\n\n"
        "  # With MS-Buddy analysis\n"
        "  python -m envnet.deconvolution.workflows --do_buddy my_file.h5\n\n"
        "  # Custom parameters\n"
        "  python -m envnet.deconvolution.workflows --mz_tol 0.005 --max_rt 10 --do_buddy --polarity positive my_file.h5\n\n"
        "  # Specify output file\n"
        "  python -m envnet.deconvolution.workflows --output /path/to/results.parquet --do_buddy my_file.h5\n"
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
    parser.add_argument('--min_library_match_score', type=float, default=0.8, help='Similarity cutoff for spectral comparison')
    parser.add_argument('--min_intensity_ratio', type=float, default=2.0, help='Minimum intensity ratio threshold')
    parser.add_argument('--max_rt', type=float, default=30.0, help='Maximum retention time for filtering')
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
    
    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Handle parallel flag (convert --no_parallel to parallel=False)
    args.parallel = not args.no_parallel
    
    main(args)


# ===================================================================
# USAGE EXAMPLES
# ===================================================================

"""
Example 1: Basic deconvolution only (no MS-Buddy)
"""
# from envnet.deconvolution.workflows import deconvolute_lcms_file
# from envnet.config.deconvolution_config import DeconvolutionConfig
# 
# # Use default settings
# df = deconvolute_lcms_file('/path/to/your/file.h5')
# 
# # Or with custom config
# config = DeconvolutionConfig(
#     mz_tol=0.005,
#     max_rt=10.0,
#     isolation_tolerance=0.8
# )
# df = deconvolute_lcms_file('/path/to/your/file.h5', config=config)

"""
Example 2: Full workflow with MS-Buddy analysis
"""
# from envnet.deconvolution.workflows import LCMSWorkflow
# from envnet.config.deconvolution_config import DeconvolutionConfig
# from envnet.config.buddy_config import BuddyAnalysisConfig
# 
# # Use default settings for both deconvolution and MS-Buddy
# workflow = LCMSWorkflow(do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.h5')
# 
# # Custom configurations
# deconv_config = DeconvolutionConfig(
#     mz_tol=0.005,
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
# from envnet.deconvolution.workflows import run_lcms_workflow_with_defaults, run_lcms_workflow
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
# from envnet.deconvolution.core import MSBuddyAnalyzer
# from envnet.config.buddy_config import BuddyAnalysisConfig
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
# from envnet.deconvolution.workflows import LCMSWorkflow
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
# from envnet.deconvolution.workflows import LCMSWorkflow
# from envnet.config.deconvolution_config import DeconvolutionConfig
# 
# # For mzML files, you might need different file_key
# config = DeconvolutionConfig(file_key='ms2_neg')  # for negative mode
# workflow = LCMSWorkflow(deconv_config=config, do_buddy=True)
# df = workflow.run_full_workflow('/path/to/your/file.mzML')

"""
Example 7: Accessing intermediate results and engine objects
"""
# from envnet.deconvolution.workflows import LCMSWorkflow
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
# from envnet.deconvolution.workflows import LCMSWorkflow
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
# from envnet.deconvolution.workflows import LCMSWorkflow
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
# from envnet.deconvolution import LCMSDeconvolutionTools, MSBuddyAnalyzer
# from envnet.config import DeconvolutionConfig, BuddyAnalysisConfig
# 
# # Process in steps to manage memory
# deconv_config = DeconvolutionConfig()
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
# # Step 2: MS-Buddy analysis on saved data
# # df = pd.read_parquet('/tmp/deconvoluted_intermediate.parquet')
# # buddy_analyzer = MSBuddyAnalyzer(buddy_config)
# # df_final = buddy_analyzer.analyze_spectra(df)