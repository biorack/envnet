"""
Core analysis functionality for ENVnet annotation results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from pathlib import Path
import os

from ..config.analysis_config import AnalysisConfig
from .dataloading import AnnotationDataLoader
from .statistics import StatisticalAnalyzer
from .multivariate import MultivariateAnalyzer
from .visualization import AnalysisVisualizer
from .enrichment import EnrichmentAnalyzer
from .utils import AnalysisUtils


class AnalysisEngine:
    """
    Main engine for analyzing ENVnet annotation results.
    
    This class orchestrates comprehensive analysis workflows including:
    1. Data integration and filtering
    2. Statistical analysis (t-tests, fold change)
    3. Multivariate analysis (PCA)
    4. Compound class enrichment
    5. Visualization (UpSet plots, set cover, etc.)
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize analysis engine with configuration."""
        self.config = config or AnalysisConfig()
        
        # Initialize components
        self.data_loader = AnnotationDataLoader(self.config)
        self.stats_analyzer = StatisticalAnalyzer(self.config)
        self.multivariate = MultivariateAnalyzer(self.config)
        self.visualizer = AnalysisVisualizer(self.config)
        self.enrichment = EnrichmentAnalyzer(self.config)
        self.utils = AnalysisUtils(self.config)
        
        # Data storage
        self.ms1_data: Optional[pd.DataFrame] = None
        self.ms2_data: Optional[Dict[str, pd.DataFrame]] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.file_metadata: Optional[pd.DataFrame] = None
        
    def load_annotation_results(self, 
                               ms1_file: Optional[str] = None,
                               ms2_deconv_file: Optional[str] = None,
                               ms2_original_file: Optional[str] = None,
                               file_metadata: Optional[Union[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load annotation results from parquet files.
        
        Args:
            ms1_file: Path to MS1 annotation results
            ms2_deconv_file: Path to MS2 deconvoluted annotation results  
            ms2_original_file: Path to MS2 original annotation results
            file_metadata: File metadata (DataFrame or path to CSV/Google Sheets)
            
        Returns:
            Dict containing loaded data
        """
        print("Loading annotation results...")
        
        # Use config defaults if not provided
        ms1_file = ms1_file or self.config.ms1_results_file
        ms2_deconv_file = ms2_deconv_file or self.config.ms2_deconv_results_file
        ms2_original_file = ms2_original_file or self.config.ms2_original_results_file
        
        # Load data
        data = self.data_loader.load_annotation_data(
            ms1_file, ms2_deconv_file, ms2_original_file, file_metadata
        )
        
        self.ms1_data = data['ms1']
        self.ms2_data = data['ms2']
        self.file_metadata = data['metadata']
        
        print(f"Loaded MS1 data: {len(self.ms1_data) if self.ms1_data is not None else 0} records")
        print(f"Loaded MS2 deconv data: {len(self.ms2_data.get('deconvoluted', [])) if self.ms2_data else 0} records")
        print(f"Loaded MS2 original data: {len(self.ms2_data.get('original', [])) if self.ms2_data else 0} records")
        
        return data
    
    def prepare_analysis_data(self, require_ms2_support: Optional[bool] = None) -> pd.DataFrame:
        """
        Prepare and filter data for analysis.
        
        Args:
            require_ms2_support: Whether to require MS2 support (overrides config)
            
        Returns:
            pd.DataFrame: Prepared analysis-ready data
        """
        if self.ms1_data is None:
            raise ValueError("Must load annotation results first")
            
        require_ms2 = require_ms2_support if require_ms2_support is not None else self.config.require_ms2_support
        
        print("Preparing analysis data...")
        self.merged_data = self.data_loader.prepare_analysis_data(
            self.ms1_data, self.ms2_data, self.file_metadata, require_ms2
        )
        
        print(f"Prepared data: {len(self.merged_data)} records")
        return self.merged_data
    
    def run_pca_analysis(self, grouping_column: str = 'specific_environment',
                        output_file: Optional[str] = None) -> Dict:
        """
        Run PCA analysis on the data.
        
        Args:
            grouping_column: Column to group/color samples by
            output_file: Optional file to save plot
            
        Returns:
            Dict with PCA results and plot
        """
        if self.merged_data is None:
            raise ValueError("Must prepare analysis data first")
            
        print("Running PCA analysis...")
        results = self.multivariate.run_pca(
            self.merged_data, grouping_column, output_file
        )
        
        return results
    
    def run_statistical_analysis(self, 
                                control_group: str,
                                treatment_group: str,
                                output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Run statistical analysis (t-tests, fold change).
        
        Args:
            control_group: Name of control group
            treatment_group: Name of treatment group  
            output_file: Optional file to save results
            
        Returns:
            pd.DataFrame: Statistical results
        """
        if self.merged_data is None:
            raise ValueError("Must prepare analysis data first")
            
        print(f"Running statistical analysis: {treatment_group} vs {control_group}")
        results = self.stats_analyzer.run_analysis(
            self.merged_data, self.file_metadata, control_group, treatment_group
        )
        
        if output_file:
            results.to_csv(output_file)
            print(f"Statistical results saved to {output_file}")
            
        return results
    
    def run_enrichment_analysis(self, 
                               stats_results: pd.DataFrame,
                               envnet_data: Optional[pd.DataFrame] = None,
                               output_dir: Optional[str] = None) -> Dict:
        """
        Run compound class enrichment analysis.
        
        Args:
            stats_results: Statistical analysis results
            envnet_data: ENVnet node data with compound classes
            output_dir: Directory to save results
            
        Returns:
            Dict with enrichment results
        """
        print("Running compound class enrichment analysis...")
        results = self.enrichment.run_enrichment_analysis(
            stats_results, envnet_data, output_dir
        )
        
        return results
    
    def generate_upset_plot(self, 
                           grouping_column: str = 'specific_environment',
                           output_file: Optional[str] = None) -> None:
        """
        Generate UpSet plot showing compound overlap across groups.
        
        Args:
            grouping_column: Column to group by
            output_file: Optional file to save plot
        """
        if self.merged_data is None:
            raise ValueError("Must prepare analysis data first")
            
        print("Generating UpSet plot...")
        self.visualizer.create_upset_plot(
            self.merged_data, grouping_column, output_file
        )
    
    def generate_set_cover_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """
        Generate set cover analysis and plots.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dict with set cover results
        """
        if self.merged_data is None:
            raise ValueError("Must prepare analysis data first")
            
        print("Running set cover analysis...")
        results = self.visualizer.create_set_cover_plots(
            self.merged_data, output_dir
        )
        
        return results
    
    def run_full_analysis(self, 
                         annotation_files: Dict[str, str],
                         file_metadata: Union[str, pd.DataFrame],
                         control_group: str,
                         treatment_group: str,
                         output_dir: str,
                         envnet_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete analysis workflow.
        
        Args:
            annotation_files: Dict with 'ms1', 'ms2_deconv', 'ms2_original' file paths
            file_metadata: File metadata 
            control_group: Control group name
            treatment_group: Treatment group name
            output_dir: Output directory
            envnet_data: Optional ENVnet node data
            
        Returns:
            Dict with all analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Load data
        print("=== Step 1: Loading Data ===")
        self.load_annotation_results(
            ms1_file=annotation_files.get('ms1'),
            ms2_deconv_file=annotation_files.get('ms2_deconv'),
            ms2_original_file=annotation_files.get('ms2_original'),
            file_metadata=file_metadata
        )
        
        # 2. Prepare analysis data
        print("\n=== Step 2: Preparing Analysis Data ===")
        self.prepare_analysis_data()
        
        # 3. PCA Analysis
        print("\n=== Step 3: PCA Analysis ===")
        pca_file = output_dir / "pca_analysis.png"
        results['pca'] = self.run_pca_analysis(output_file=str(pca_file))
        
        # 4. Statistical Analysis
        print("\n=== Step 4: Statistical Analysis ===")
        stats_file = output_dir / "statistical_results.csv"
        results['statistics'] = self.run_statistical_analysis(
            control_group, treatment_group, str(stats_file)
        )
        
        # 5. UpSet Plot
        print("\n=== Step 5: UpSet Plot ===")
        upset_file = output_dir / "upset_plot.png"
        self.generate_upset_plot(output_file=str(upset_file))
        
        # 6. Set Cover Analysis
        print("\n=== Step 6: Set Cover Analysis ===")
        results['set_cover'] = self.generate_set_cover_analysis(str(output_dir))
        
        # 7. Enrichment Analysis (if ENVnet data provided)
        if envnet_data is not None:
            print("\n=== Step 7: Enrichment Analysis ===")
            results['enrichment'] = self.run_enrichment_analysis(
                results['statistics'], envnet_data, str(output_dir)
            )
        
        # 8. Generate summary report
        self._generate_analysis_summary(results, output_dir)
        
        print("\nFull analysis workflow complete!")
        return results
    
    def _generate_analysis_summary(self, results: Dict, output_dir: Path):
        """Generate summary report of analysis results."""
        summary = {
            'data_summary': {
                'total_ms1_features': len(self.merged_data) if self.merged_data is not None else 0,
                'unique_compounds': self.merged_data['original_index'].nunique() if self.merged_data is not None else 0,
                'unique_samples': self.merged_data['lcmsrun_observed'].nunique() if self.merged_data is not None else 0,
                'environments': list(self.merged_data['specific_environment'].unique()) if self.merged_data is not None else []
            }
        }
        
        # Add statistical summary
        if 'statistics' in results:
            stats = results['statistics']
            summary['statistical_summary'] = {
                'significant_features': len(stats[stats['p_value'] < self.config.max_pvalue]),
                'upregulated_features': len(stats[stats['log2_foldchange'] > 0]),
                'downregulated_features': len(stats[stats['log2_foldchange'] < 0])
            }
        
        # Save summary
        import json
        summary_file = output_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Analysis summary saved to {summary_file}")