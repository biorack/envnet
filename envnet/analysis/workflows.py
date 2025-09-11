"""
End-to-end analysis workflows and CLI interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Union, List
import json
import pandas as pd

from ..config.analysis_config import AnalysisConfig
from .core import AnalysisEngine


class AnalysisWorkflow:
    """High-level workflow orchestrator for ENVnet analysis."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize workflow with configuration."""
        self.config = config or AnalysisConfig()
        self.engine = AnalysisEngine(self.config)
        
    def run_pca_workflow(self, 
                        annotation_files: Dict[str, str],
                        file_metadata: Union[str, pd.DataFrame],
                        output_dir: str,
                        grouping_column: str = 'specific_environment') -> str:
        """
        Run PCA analysis workflow.
        
        Args:
            annotation_files: Dict with annotation file paths
            file_metadata: File metadata
            output_dir: Output directory
            grouping_column: Column for grouping/coloring
            
        Returns:
            str: Path to PCA plot
        """
        print("=== PCA Analysis Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self.engine.load_annotation_results(
            ms1_file=annotation_files.get('ms1'),
            ms2_deconv_file=annotation_files.get('ms2_deconv'),
            ms2_original_file=annotation_files.get('ms2_original'),
            file_metadata=file_metadata
        )
        
        self.engine.prepare_analysis_data()
        
        # Run PCA
        pca_file = output_dir / "pca_analysis.png"
        results = self.engine.run_pca_analysis(
            grouping_column=grouping_column,
            output_file=str(pca_file)
        )
        
        print(f"PCA analysis complete. Plot saved to {pca_file}")
        return str(pca_file)
    
    def run_statistical_workflow(self, 
                                annotation_files: Dict[str, str],
                                file_metadata: Union[str, pd.DataFrame],
                                control_group: str,
                                treatment_group: str,
                                output_dir: str) -> str:
        """
        Run statistical analysis workflow.
        
        Args:
            annotation_files: Dict with annotation file paths
            file_metadata: File metadata
            control_group: Control group name
            treatment_group: Treatment group name
            output_dir: Output directory
            
        Returns:
            str: Path to statistical results
        """
        print("=== Statistical Analysis Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self.engine.load_annotation_results(
            ms1_file=annotation_files.get('ms1'),
            ms2_deconv_file=annotation_files.get('ms2_deconv'),
            ms2_original_file=annotation_files.get('ms2_original'),
            file_metadata=file_metadata
        )
        
        self.engine.prepare_analysis_data()
        
        # Run statistical analysis
        stats_file = output_dir / "statistical_results.csv"
        results = self.engine.run_statistical_analysis(
            control_group, treatment_group, str(stats_file)
        )
        
        print(f"Statistical analysis complete. Results saved to {stats_file}")
        return str(stats_file)
    
    def run_visualization_workflow(self, 
                                  annotation_files: Dict[str, str],
                                  file_metadata: Union[str, pd.DataFrame],
                                  output_dir: str) -> Dict[str, str]:
        """
        Run visualization workflow (UpSet plots, set cover).
        
        Args:
            annotation_files: Dict with annotation file paths
            file_metadata: File metadata
            output_dir: Output directory
            
        Returns:
            Dict with paths to generated plots
        """
        print("=== Visualization Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self.engine.load_annotation_results(
            ms1_file=annotation_files.get('ms1'),
            ms2_deconv_file=annotation_files.get('ms2_deconv'),
            ms2_original_file=annotation_files.get('ms2_original'),
            file_metadata=file_metadata
        )
        
        self.engine.prepare_analysis_data()
        
        output_files = {}
        
        # UpSet plot
        upset_file = output_dir / "upset_plot.png"
        self.engine.generate_upset_plot(output_file=str(upset_file))
        output_files['upset_plot'] = str(upset_file)
        
        # Set cover analysis
        set_cover_results = self.engine.generate_set_cover_analysis(str(output_dir))
        output_files['set_cover_pdf'] = str(output_dir / "set_cover_results.pdf")
        
        print("Visualization workflow complete!")
        return output_files
    
    def run_enrichment_workflow(self, 
                               stats_results: Union[str, pd.DataFrame],
                               envnet_data: Union[str, pd.DataFrame],
                               output_dir: str) -> Dict[str, str]:
        """
        Run compound class enrichment workflow.
        
        Args:
            stats_results: Statistical results (file path or DataFrame)
            envnet_data: ENVnet node data (file path or DataFrame)
            output_dir: Output directory
            
        Returns:
            Dict with paths to enrichment results
        """
        print("=== Compound Class Enrichment Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if isinstance(stats_results, str):
            stats_data = pd.read_csv(stats_results, index_col=0)
        else:
            stats_data = stats_results
            
        if isinstance(envnet_data, str):
            if envnet_data.endswith('.parquet'):
                envnet_df = pd.read_parquet(envnet_data)
            else:
                envnet_df = pd.read_csv(envnet_data)
        else:
            envnet_df = envnet_data
        
        # Run enrichment analysis
        results = self.engine.enrichment.run_enrichment_analysis(
            stats_data, envnet_df, str(output_dir)
        )
        
        output_files = {
            'enrichment_pdf': str(output_dir / "compound_class_enrichment.pdf")
        }
        
        # Add individual CSV files
        for class_col in results.keys():
            csv_file = output_dir / f"{class_col}_enrichment.csv"
            if csv_file.exists():
                output_files[f'{class_col}_csv'] = str(csv_file)
        
        print("Enrichment analysis workflow complete!")
        return output_files
    
    def run_full_analysis_workflow(self, 
                                  annotation_files: Dict[str, str],
                                  file_metadata: Union[str, pd.DataFrame],
                                  control_group: str,
                                  treatment_group: str,
                                  output_dir: str,
                                  envnet_data: Optional[Union[str, pd.DataFrame]] = None) -> Dict[str, str]:
        """
        Run complete analysis workflow.
        
        Args:
            annotation_files: Dict with annotation file paths
            file_metadata: File metadata
            control_group: Control group name
            treatment_group: Treatment group name
            output_dir: Output directory
            envnet_data: Optional ENVnet node data for enrichment
            
        Returns:
            Dict with paths to all output files
        """
        print("=== Full ENVnet Analysis Workflow ===")
        
        # Use the engine's full analysis method
        results = self.engine.run_full_analysis(
            annotation_files=annotation_files,
            file_metadata=file_metadata,
            control_group=control_group,
            treatment_group=treatment_group,
            output_dir=output_dir,
            envnet_data=envnet_data
        )
        
        # Convert results to file paths
        output_dir = Path(output_dir)
        output_files = {
            'pca_plot': str(output_dir / "pca_analysis.png"),
            'statistical_results': str(output_dir / "statistical_results.csv"),
            'upset_plot': str(output_dir / "upset_plot.png"),
            'set_cover_pdf': str(output_dir / "set_cover_results.pdf"),
            'analysis_summary': str(output_dir / "analysis_summary.json")
        }
        
        if envnet_data is not None:
            output_files['enrichment_pdf'] = str(output_dir / "compound_class_enrichment.pdf")
        
        return output_files


def main():
    """Command-line interface for ENVnet analysis."""
    parser = argparse.ArgumentParser(description='ENVnet Analysis Workflows')
    parser.add_argument('workflow', 
                       choices=['pca', 'stats', 'viz', 'enrichment', 'full'],
                       help='Analysis workflow to run')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Output directory for results')
    
    # Input files
    parser.add_argument('--ms1-file', 
                       help='MS1 annotation results file (parquet)')
    parser.add_argument('--ms2-deconv-file',
                       help='MS2 deconvoluted annotation results file (parquet)')
    parser.add_argument('--ms2-original-file',
                       help='MS2 original annotation results file (parquet)')
    parser.add_argument('--file-metadata',
                       help='File metadata (CSV file or "google_sheets")')
    
    # Statistical analysis parameters
    parser.add_argument('--control-group', 
                       help='Control group name for statistical analysis')
    parser.add_argument('--treatment-group',
                       help='Treatment group name for statistical analysis')
    
    # Enrichment analysis
    parser.add_argument('--envnet-data',
                       help='ENVnet node data file for enrichment analysis')
    parser.add_argument('--stats-results',
                       help='Statistical results file for enrichment analysis')
    
    # Configuration options
    parser.add_argument('--config-file',
                       help='Custom configuration file (JSON)')
    parser.add_argument('--peak-value', 
                       choices=['peak_area', 'peak_height'],
                       default='peak_area',
                       help='Peak value to use for analysis')
    parser.add_argument('--require-ms2-support', action='store_true',
                       help='Require MS2 support for MS1 features')
    parser.add_argument('--normalize-data', action='store_true', default=True,
                       help='Normalize data by total signal per sample')
    parser.add_argument('--max-pvalue', type=float, default=0.05,
                       help='Maximum p-value for significance')
    parser.add_argument('--grouping-column', default='specific_environment',
                       help='Column for grouping samples in PCA')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        with open(args.config_file) as f:
            config_dict = json.load(f)
        config = AnalysisConfig(**config_dict)
    else:
        config = AnalysisConfig(
            peak_value=args.peak_value,
            require_ms2_support=args.require_ms2_support,
            normalize_data=args.normalize_data,
            max_pvalue=args.max_pvalue
        )
    
    # Prepare annotation files dict
    annotation_files = {}
    if args.ms1_file:
        annotation_files['ms1'] = args.ms1_file
    if args.ms2_deconv_file:
        annotation_files['ms2_deconv'] = args.ms2_deconv_file
    if args.ms2_original_file:
        annotation_files['ms2_original'] = args.ms2_original_file
    
    # Initialize workflow
    workflow = AnalysisWorkflow(config)
    
    try:
        # Run requested workflow
        if args.workflow == 'pca':
            if not annotation_files or not args.file_metadata:
                print("Error: PCA workflow requires annotation files and file metadata")
                sys.exit(1)
            
            output_file = workflow.run_pca_workflow(
                annotation_files, args.file_metadata, args.output_dir, 
                args.grouping_column
            )
            print(f"PCA results: {output_file}")
            
        elif args.workflow == 'stats':
            if not all([annotation_files, args.file_metadata, args.control_group, args.treatment_group]):
                print("Error: Stats workflow requires annotation files, metadata, and group names")
                sys.exit(1)
                
            output_file = workflow.run_statistical_workflow(
                annotation_files, args.file_metadata, 
                args.control_group, args.treatment_group, args.output_dir
            )
            print(f"Statistical results: {output_file}")
            
        elif args.workflow == 'viz':
            if not annotation_files or not args.file_metadata:
                print("Error: Visualization workflow requires annotation files and metadata")
                sys.exit(1)
                
            output_files = workflow.run_visualization_workflow(
                annotation_files, args.file_metadata, args.output_dir
            )
            print(f"Visualization results: {output_files}")
            
        elif args.workflow == 'enrichment':
            if not args.stats_results or not args.envnet_data:
                print("Error: Enrichment workflow requires statistical results and ENVnet data")
                sys.exit(1)
                
            output_files = workflow.run_enrichment_workflow(
                args.stats_results, args.envnet_data, args.output_dir
            )
            print(f"Enrichment results: {output_files}")
            
        elif args.workflow == 'full':
            if not all([annotation_files, args.file_metadata, args.control_group, args.treatment_group]):
                print("Error: Full workflow requires annotation files, metadata, and group names")
                sys.exit(1)
                
            output_files = workflow.run_full_analysis_workflow(
                annotation_files, args.file_metadata,
                args.control_group, args.treatment_group, args.output_dir,
                args.envnet_data
            )
            print(f"Full analysis results: {output_files}")
            
    except Exception as e:
        print(f"Error running analysis workflow: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()