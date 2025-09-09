#!/usr/bin/env python3
"""
Command line interface for ENVnet molecular network construction.

Usage examples:
    # Quick test build
    python build_envnet.py quick --max-spectra 500 --output data/test

    # Full build
    python build_envnet.py build --output data/full_build

    # Library matching only
    python build_envnet.py library-matching --output data/library_analysis

    # With custom config
    python build_envnet.py build --config my_config.yaml --output data/custom_build
"""

import argparse
import sys
import traceback
from pathlib import Path
import time

# Add the parent directory to Python path so we can import envnet
sys.path.insert(0, str(Path(__file__).parent.parent))

from envnet.build import ENVnetBuilder, build_envnet, quick_envnet
from envnet.config import BuildConfig
from envnet.build.workflows import ENVnetWorkflows

#!/usr/bin/env python3
"""
Command line interface for ENVnet molecular network construction.
...
"""

print("DEBUG: Script starting...")
print(f"DEBUG: Python executable: {sys.executable}")
print(f"DEBUG: Script path: {__file__}")

import argparse
import sys
import traceback
from pathlib import Path
import time

print("DEBUG: Basic imports successful...")

# Add the parent directory to Python path so we can import envnet
sys.path.insert(0, str(Path(__file__).parent.parent))
print(f"DEBUG: Added to path: {Path(__file__).parent.parent}")

try:
    from envnet.build import ENVnetBuilder, build_envnet, quick_envnet
    from envnet.config import BuildConfig
    print("DEBUG: ENVnet imports successful!")
except Exception as e:
    print(f"DEBUG: Import error: {e}")
    traceback.print_exc()
    sys.exit(1)
    

def setup_logging(verbose: bool = True):
    """Set up basic logging."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def quick_build_command(args):
    """Run quick build for testing."""
    print("=" * 60)
    print("ENVnet Quick Build")
    print("=" * 60)
    print(f"Max spectra: {args.max_spectra}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output}")
    print()
    
    start_time = time.time()
    
    try:
        # Load config if provided
        config = None
        if args.config:
            config = BuildConfig.from_file(args.config)
            print(f"Loaded configuration from: {args.config}")
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run quick build
        builder = quick_envnet(
            max_spectra=args.max_spectra,
            max_files=args.max_files,
            output_dir=output_dir,
            config=config,
            verbose=args.verbose
        )
        

        
        builder.save_network(output_dir / "quick_network.graphml")
        builder.save_node_data(output_dir / "quick_node_data.csv")
        
        # Print summary
        summary = builder.get_network_summary()
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("QUICK BUILD COMPLETE")
        print("=" * 60)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Network nodes: {summary.get('num_nodes', 0)}")
        print(f"Network edges: {summary.get('num_edges', 0)}")
        print(f"Connected components: {summary.get('num_connected_components', 0)}")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in quick build: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def full_build_command(args):
    """Run full ENVnet build."""
    print("=" * 60)
    print("ENVnet Full Build")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print()
    
    start_time = time.time()
    
    # try:
    # Load config if provided
    config = None
    if args.config:
        config = BuildConfig.from_file(args.config)
        print(f"Loaded configuration from: {args.config}")
    
    # Run full build
    builder = build_envnet(
        config=config,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Save results
    output_dir = Path(args.output)
    builder.save_network(output_dir / "envnet_network.graphml")
    builder.save_node_data(output_dir / "envnet_node_data.csv")
    builder.save_library_matches(args.output)
    
    # Print summary
    summary = builder.get_network_summary()
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("FULL BUILD COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Network nodes: {summary.get('num_nodes', 0)}")
    print(f"Network edges: {summary.get('num_edges', 0)}")
    print(f"Connected components: {summary.get('num_connected_components', 0)}")
    print(f"Results saved to: {output_dir}")
    
    return True
        
    # except Exception as e:
    #     print(f"\nERROR in full build: {e}")
    #     if args.verbose:
    #         traceback.print_exc()
    #     return False

def library_matching_command(args):
    """Run library matching analysis."""
    print("=" * 60)
    print("ENVnet Library Matching Analysis")
    print("=" * 60)
    
    try:
        config = None
        if args.config:
            config = BuildConfig.from_file(args.config)
        
        # Import here to avoid circular dependency
        from envnet.build.workflows import ENVnetWorkflows
        workflows = ENVnetWorkflows(config)
        results = workflows.library_matching_workflow()
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results['deconvoluted_matches'].to_csv(output_dir / "deconvoluted_matches.csv", index=False)
        results['original_matches'].to_csv(output_dir / "original_matches.csv", index=False)
        
        # Print summary
        stats = results['statistics']
        print(f"\nLibrary Matching Results:")
        print(f"Total spectra analyzed: {stats['total_spectra']}")
        print(f"Deconvoluted match rate: {stats['deconv_match_rate']:.1%}")
        print(f"Original match rate: {stats['orig_match_rate']:.1%}")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in library matching: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def clustering_command(args):
    """Run clustering analysis."""
    print("=" * 60)
    print("ENVnet Clustering Analysis")
    print("=" * 60)
    
    try:
        config = None
        if args.config:
            config = BuildConfig.from_file(args.config)
        
        workflows = ENVnetWorkflows(config)
        results = workflows.clustering_workflow()
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results['clustered_spectra'].to_csv(output_dir / "clustered_spectra.csv", index=False)
        
        # Print summary
        stats = results['statistics']
        print(f"\nClustering Results:")
        print(f"Total spectra: {stats['total_spectra']}")
        print(f"Unique clusters: {stats['unique_clusters']}")
        print(f"Reduction factor: {stats['reduction_factor']:.1f}x")
        print(f"Mean cluster size: {stats['mean_cluster_size']:.1f}")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in clustering: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def integrate_sirius_command(args):
    """Integrate SIRIUS results."""
    print("=" * 60)
    print("ENVnet SIRIUS Integration")
    print("=" * 60)
    
    try:
        # Load existing network
        if not args.network_file:
            print("ERROR: --network-file is required for SIRIUS integration")
            return False
        
        import networkx as nx
        network = nx.read_graphml(args.network_file)
        print(f"Loaded network from: {args.network_file}")
        
        config = None
        if args.config:
            config = BuildConfig.from_file(args.config)
        
        workflows = ENVnetWorkflows(config)
        results = workflows.sirius_integration_workflow(args.sirius_dir, network)
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        nx.write_graphml(results['network'], output_dir / "network_with_sirius.graphml")
        results['formula_results'].to_csv(output_dir / "sirius_formulas.csv", index=False)
        results['class_results'].to_csv(output_dir / "sirius_classes.csv", index=False)
        
        print(f"\nSIRIUS Integration Complete:")
        print(f"Formula predictions: {len(results['formula_results'])}")
        print(f"Class predictions: {len(results['class_results'])}")
        print(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in SIRIUS integration: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="ENVnet Molecular Network Construction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global arguments
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', default='data', help='Output directory (default: data)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick build
    quick_parser = subparsers.add_parser('quick', help='Quick build for testing')
    quick_parser.add_argument('--max-spectra', type=int, default=1000, 
                            help='Maximum spectra to process (default: 1000)')
    quick_parser.add_argument('--max-files', type=int, default=5,          # Add this
                            help='Maximum files to load (default: 5)')
    
    # Full build
    build_parser = subparsers.add_parser('build', help='Full ENVnet build')
    
    # Library matching
    library_parser = subparsers.add_parser('library-matching', help='Library matching analysis')
    
    # Clustering
    cluster_parser = subparsers.add_parser('clustering', help='Spectral clustering analysis')
    
    # SIRIUS integration
    sirius_parser = subparsers.add_parser('sirius', help='Integrate SIRIUS results')
    sirius_parser.add_argument('--sirius-dir', required=True, help='SIRIUS results directory')
    sirius_parser.add_argument('--network-file', required=True, help='Existing network GraphML file')
    sirius_parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory to save the updated network file.'
    ) 
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    verbose = args.verbose and not args.quiet
    setup_logging(verbose)
    
    # Route to appropriate command
    success = False
    if args.command == 'quick':
        success = quick_build_command(args)
    elif args.command == 'build':
        success = full_build_command(args)
    elif args.command == 'library-matching':
        success = library_matching_command(args)
    elif args.command == 'clustering':
        success = clustering_command(args)
    elif args.command == 'sirius':
        success = integrate_sirius_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())