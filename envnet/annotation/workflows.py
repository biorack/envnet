"""
End-to-end annotation workflows and CLI interface.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import json

from ..config.annotation_config import AnnotationConfig
from .core import AnnotationEngine


class AnnotationWorkflow:
    """High-level workflow orchestrator for ENVnet annotation."""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        """Initialize workflow with configuration."""
        self.config = config or AnnotationConfig()
        self.engine = AnnotationEngine(self.config)
        
    def run_ms1_annotation(self, file_source: Dict, output_dir: str,
                          envnet_files: Optional[Dict] = None) -> str:
        """
        Run MS1-only annotation workflow.
        
        Args:
            file_source: Dict specifying file source (google_sheets, csv, or file_list)
            output_dir: Output directory path
            envnet_files: Dict with graphml_file and mgf_base_name (optional)
            
        Returns:
            str: Path to output file
        """
        print("=== MS1 Annotation Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._dump_config_to_json(str(output_dir))

        # Load reference data
        if envnet_files:
            self.engine.load_envnet_reference(**envnet_files)
        else:
            self.engine.load_envnet_reference()
        
        # Load experimental files
        self.engine.load_experimental_files(**file_source)
        
        # Run MS1 annotation
        output_path = Path(output_dir) / "ms1_annotations.parquet"
        results = self.engine.annotate_ms1_features(str(output_path))
        
        print(f"MS1 annotation complete. Results saved to {output_path}")
        return str(output_path)
    
    def run_ms2_annotation(self, file_source: Dict, output_dir: str,
                          spectrum_types: List[str] = ['deconvoluted', 'original'],
                          envnet_files: Optional[Dict] = None) -> List[str]:
        """
        Run MS2-only annotation workflow.
        
        Args:
            file_source: Dict specifying file source
            output_dir: Output directory path
            spectrum_types: List of spectrum types to process
            envnet_files: Dict with graphml_file and mgf_base_name (optional)
            
        Returns:
            List[str]: Paths to output files
        """
        print("=== MS2 Annotation Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._dump_config_to_json(str(output_dir))

        # Load reference data
        if envnet_files:
            self.engine.load_envnet_reference(**envnet_files)
        else:
            self.engine.load_envnet_reference()
        
        # Load experimental files
        self.engine.load_experimental_files(**file_source)
        
        output_files = []
        for spectrum_type in spectrum_types:
            print(f"\nProcessing {spectrum_type} spectra...")
            output_path = Path(output_dir) / f"ms2_{spectrum_type}_annotations.parquet"
            
            results = self.engine.annotate_ms2_spectra(
                spectrum_type=spectrum_type,
                output_file=str(output_path)
            )
            
            output_files.append(str(output_path))
            print(f"{spectrum_type.title()} MS2 annotation complete.")
        
        return output_files
    
    def run_full_annotation(self, file_source: Dict, output_dir: str,
                           envnet_files: Optional[Dict] = None) -> Dict[str, str]:
        """
        Run complete annotation workflow (MS1 + MS2).
        
        Args:
            file_source: Dict specifying file source
            output_dir: Output directory path
            envnet_files: Dict with graphml_file and mgf_base_name (optional)
            
        Returns:
            Dict[str, str]: Mapping of annotation type to output file path
        """
        print("=== Full ENVnet Annotation Workflow ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._dump_config_to_json(str(output_dir))

        # Load reference data first
        if envnet_files:
            self.engine.load_envnet_reference(**envnet_files)
        else:
            self.engine.load_envnet_reference()
            
        # Load experimental files
        self.engine.load_experimental_files(**file_source)
        
        # Run individual annotation steps
        results = {}
        
        # MS1 annotation
        print("\n--- MS1 Annotation ---")
        ms1_output = output_dir / "ms1_annotations.parquet"
        results['ms1'] = self.engine.annotate_ms1_features(str(ms1_output))
        
        # MS2 annotation - deconvoluted spectra
        print("\n--- MS2 Deconvoluted Spectra Annotation ---")
        ms2_deconv_output = output_dir / "ms2_deconvoluted_annotations.parquet"
        results['ms2_deconvoluted'] = self.engine.annotate_ms2_spectra(
            'deconvoluted', str(ms2_deconv_output)
        )
        
        # MS2 annotation - original spectra  
        print("\n--- MS2 Original Spectra Annotation ---")
        ms2_orig_output = output_dir / "ms2_original_annotations.parquet"
        results['ms2_original'] = self.engine.annotate_ms2_spectra(
            'original', str(ms2_orig_output)
        )
        
        # Convert to file paths
        output_files = {
            'ms1': str(ms1_output),
            'ms2_deconvoluted': str(ms2_deconv_output),
            'ms2_original': str(ms2_orig_output)
        }
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        print("\nFull annotation workflow complete!")
        return output_files
    
    def _generate_summary_report(self, results: Dict, output_dir: Path):
        """Generate a summary report of annotation results."""
        summary = {}

        for annotation_type, data in results.items():
            if data is not None and len(data) > 0:
                summary[annotation_type] = {
                    'total_annotations': len(data),
                    'unique_files': data['lcmsrun_observed'].nunique() if 'lcmsrun_observed' in data.columns else 0,
                    'unique_envnet_matches': data['original_index'].nunique() if 'original_index' in data.columns else 0
                }
                
                # Add confidence stats for MS2
                if 'confidence_level' in data.columns:
                    summary[annotation_type]['confidence_breakdown'] = data['confidence_level'].value_counts().to_dict()
        
        # Save summary
        summary_file = output_dir / "annotation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary report saved to {summary_file}")
    
    def _dump_config_to_json(self, output_dir: str):
        """
        Save configuration parameters to JSON file in output directory.
        
        Args:
            output_dir: Output directory path
        """
        import json
        from datetime import datetime
        from pathlib import Path
        import dataclasses
        
        output_path = Path(output_dir)
        config_file = output_path / "annotation_config.json"

        # Convert config to dictionary
        if dataclasses.is_dataclass(self.config):
            config_dict = dataclasses.asdict(self.config)
        else:
            config_dict = vars(self.config)
        
        # Add workflow metadata
        config_data = {
            "workflow_info": {
                "timestamp": datetime.now().isoformat(),
                "envnet_version": getattr(self.config, 'version', 'unknown')
            },
            "configuration": config_dict
        }
        
        # Handle non-serializable objects
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            else:
                return str(obj)
        
        # Save to file
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=json_serializer)
            print(f"Configuration saved to: {config_file}")
        except Exception as e:
            print(f"Warning: Could not save configuration to JSON: {e}")


def main():
    """Command-line interface for ENVnet annotation."""
    parser = argparse.ArgumentParser(description='ENVnet Annotation Workflows')
    parser.add_argument('workflow', choices=['ms1', 'ms2', 'full'],
                       help='Annotation workflow to run')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Output directory for results')
    
    # ENVnet reference file options
    envnet_group = parser.add_argument_group('ENVnet Reference Files')
    envnet_group.add_argument('--envnet-graphml', 
                             default='envnet.graphml',
                             help='ENVnet GraphML file (default: <module>/data/envnet.graphml)')
    envnet_group.add_argument('--envnet-mgf-base', 
                             default='envnet',
                             help='Base name for ENVnet MGF files (default: envnet, expects envnet_deconvoluted_spectra.mgf and envnet_original_spectra.mgf)')
    envnet_group.add_argument('--envnet-deconv-mgf',
                             help='Path to deconvoluted spectra MGF file (overrides --envnet-mgf-base)')
    envnet_group.add_argument('--envnet-original-mgf',
                             help='Path to original spectra MGF file (overrides --envnet-mgf-base)')
    
    # File source options
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('--google-sheets', action='store_true',
                           help='Load files from Google Sheets (default config)')
    file_group.add_argument('--csv-file', 
                           help='CSV file with file metadata')
    file_group.add_argument('--file-list', nargs='+',
                           help='List of file paths')
    
    # MS2-specific options
    parser.add_argument('--spectrum-types', nargs='+', 
                       choices=['deconvoluted', 'original'],
                       default=['deconvoluted', 'original'],
                       help='Spectrum types for MS2 annotation')
    
    # Configuration options
    parser.add_argument('--config-file', 
                       help='Custom configuration file (JSON)')
    parser.add_argument('--mz-tol', type=float, default=0.01,
                       help='m/z tolerance for matching')
    parser.add_argument('--min-library-match-score', type=float, default=0.5,
                       help='Minimum score for MS2 matches')
    parser.add_argument('--min-matches', type=int, default=3,
                       help='Minimum number of matching peaks for MS2')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Chunk size for MS2 processing')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        # Load custom config from JSON
        with open(args.config_file) as f:
            config_dict = json.load(f)
        config = AnnotationConfig(**config_dict)
    else:
        config = AnnotationConfig(
            mz_tol=args.mz_tol, 
            min_library_match_score=args.min_library_match_score,
            min_matches=args.min_matches,
            chunk_size=args.chunk_size
        )
    
    # Determine file source
    if args.google_sheets:
        file_source = {
            'google_sheet_config': {
                'notebook_name': 'Supplementary Tables',
                'file_sheet': 'Table 1a',
                'envo_sheet': 'Table 1b'
            }
        }
    elif args.csv_file:
        file_source = {'csv_file': args.csv_file}
    else:
        file_source = {'file_list': args.file_list}
    
    # Determine ENVnet reference files
    envnet_files = None
    if (args.envnet_graphml != 'envnet.graphml' or 
        args.envnet_mgf_base != 'envnet' or
        args.envnet_deconv_mgf or 
        args.envnet_original_mgf):
        
        envnet_files = {
            'graphml_file': args.envnet_graphml,
            'mgf_base_name': args.envnet_mgf_base
        }
        
        # Handle custom MGF file paths if provided
        if args.envnet_deconv_mgf or args.envnet_original_mgf:
            envnet_files['custom_mgf_paths'] = {
                'deconvoluted': args.envnet_deconv_mgf,
                'original': args.envnet_original_mgf
            }
    
    # Print configuration summary
    print("=== ENVnet Annotation Configuration ===")
    print(f"Workflow: {args.workflow}")
    print(f"Output directory: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"ENVnet GraphML: {args.envnet_graphml}")
    if args.envnet_deconv_mgf:
        print(f"Deconvoluted MGF: {args.envnet_deconv_mgf}")
    else:
        print(f"Deconvoluted MGF: <module>/data/{args.envnet_mgf_base}_deconvoluted_spectra.mgf")
    if args.envnet_original_mgf:
        print(f"Original MGF: {args.envnet_original_mgf}")
    else:
        print(f"Original MGF: <module>/data/{args.envnet_mgf_base}_original_spectra.mgf")
    print(f"m/z tolerance: {args.mz_tol}")
    print(f"Min score: {args.min_library_match_score}")
    if args.workflow in ['ms2', 'full']:
        print(f"Spectrum types: {args.spectrum_types}")
    print()
    
    # Initialize workflow
    workflow = AnnotationWorkflow(config)
    
    # try:
        # Run requested workflow
    if args.workflow == 'ms1':
        output_file = workflow.run_ms1_annotation(file_source, args.output_dir, envnet_files)
        print(f"MS1 annotation results: {output_file}")
        
    elif args.workflow == 'ms2':
        output_files = workflow.run_ms2_annotation(
            file_source, args.output_dir, args.spectrum_types, envnet_files
        )
        print(f"MS2 annotation results: {output_files}")
        
    elif args.workflow == 'full':
        output_files = workflow.run_full_annotation(file_source, args.output_dir, envnet_files)
        print(f"Full annotation results: {output_files}")
    
    # except Exception as e:
        # print(f"Error running annotation workflow: {e}")
        # sys.exit(1)


if __name__ == '__main__':
    main()