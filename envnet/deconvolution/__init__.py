"""ENVnet deconvolution module for processing dissolved organic matter mass spectrometry data.

This module provides tools for deconvoluting chimeric LCMS/MSMS spectra using
neutral loss patterns and clustering algorithms specifically designed for DOM analysis.
"""

# Import main classes and functions for public API
from .core import LCMSDeconvolutionTools, MSBuddyAnalyzer
from .preprocessing import LCMSDataProcessor
from .algorithms import DeconvolutionAlgorithms
from .postprocessing import DeconvolutionPostProcessor
from .workflows import (
    LCMSWorkflow,
    deconvolute_lcms_file,
    run_lcms_workflow_with_defaults,
    run_lcms_workflow
)

# Define what gets imported with "from envnet.deconvolution import *"
__all__ = [
    # Main workflow classes
    'LCMSDeconvolutionTools',
    'MSBuddyAnalyzer', 
    'LCMSWorkflow',
    
    # Component classes
    'LCMSDataProcessor',
    'DeconvolutionAlgorithms', 
    'DeconvolutionPostProcessor',
    
    # Convenience functions
    'deconvolute_lcms_file',
    'run_lcms_workflow_with_defaults',
    'run_lcms_workflow'
]