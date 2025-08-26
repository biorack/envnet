"""
ENVnet build module for molecular network construction.

This module provides tools for building molecular networks from deconvoluted
mass spectrometry data using library matching, spectral clustering, and 
REM-BLINK networking.
"""

# Core interface classes
from .core import ENVnetBuilder, build_envnet, quick_envnet
from .workflows import ENVnetWorkflows

# Individual module classes  
from .dataloading import SpectraLoader
from .library_matching import LibraryMatcher
from .clustering import SpectraClusterer
from .network import NetworkBuilder
from .mgf_tools import MGFGenerator
from .integration import SiriusIntegrator
from .visualization import NetworkVisualizer

# Utility functions
from .formula_tools import (
    formula_to_dict, calculate_mass, get_formula_props,
    calc_dbe, aromaticity_index, calc_nosc
)
from .mgf_tools import create_mgf_files_for_sirius
from .integration import integrate_sirius_results
from .visualization import plot_network_classes, create_visualization_report

# Main exports
__all__ = [
    # Core interfaces
    'ENVnetBuilder',
    'build_envnet', 
    'quick_envnet',
    'ENVnetWorkflows',
    
    # Module classes
    'SpectraLoader',
    'LibraryMatcher', 
    'SpectraClusterer',
    'NetworkBuilder',
    'MGFGenerator',
    'SiriusIntegrator',
    'NetworkVisualizer',
    
    # Utility functions
    'formula_to_dict',
    'calculate_mass',
    'get_formula_props',
    'calc_dbe',
    'aromaticity_index', 
    'calc_nosc',
    'create_mgf_files_for_sirius',
    'integrate_sirius_results',
    'plot_network_classes',
    'create_visualization_report'
]

# Version info
__version__ = '1.0.0'