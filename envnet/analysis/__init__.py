"""
ENVnet analysis module for downstream analysis of annotation results.

This module provides tools for comprehensive analysis of ENVnet annotation data including:
- Statistical analysis (t-tests, fold change)
- Multivariate analysis (PCA)
- Compound class enrichment analysis
- Visualization (UpSet plots, set cover analysis)
- Data integration and quality control
"""

from .core import AnalysisEngine
from .workflows import AnalysisWorkflow
from .dataloading import AnnotationDataLoader
from .statistics import StatisticalAnalyzer
from .multivariate import MultivariateAnalyzer
from .visualization import AnalysisVisualizer
from .enrichment import EnrichmentAnalyzer
from .utils import AnalysisUtils

__all__ = [
    'AnalysisEngine',
    'AnalysisWorkflow',
    'AnnotationDataLoader',
    'StatisticalAnalyzer',
    'MultivariateAnalyzer',
    'AnalysisVisualizer',
    'EnrichmentAnalyzer',
    'AnalysisUtils'
]