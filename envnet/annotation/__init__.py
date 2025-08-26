"""
ENVnet annotation module for mapping experimental data to ENVnet reference database.

This module provides tools for annotating experimental LCMS data using the ENVnet
reference database through MS1 precursor matching and MS2 spectral matching.
"""

from .core import AnnotationEngine
# from .workflows import AnnotationWorkflow
from .dataloading import ENVnetLoader, ExperimentalDataLoader
from .ms1_matching import MS1Matcher
from .ms2_matching import MS2Matcher
from .preprocessing import AnnotationPreprocessor
from .postprocessing import AnnotationPostprocessor

__all__ = [
    'AnnotationEngine',
    'AnnotationWorkflow', 
    'ENVnetLoader',
    'ExperimentalDataLoader',
    'MS1Matcher',
    'MS2Matcher',
    'AnnotationPreprocessor',
    'AnnotationPostprocessor'
]