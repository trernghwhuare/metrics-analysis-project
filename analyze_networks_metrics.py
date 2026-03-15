#!/usr/bin/env python3
"""
Comprehensive Network Metrics Analysis Script

This script provides a complete workflow for analyzing network metrics:
1. Computes comprehensive graph theory metrics
2. Generates professional visualizations  
3. Produces analysis-ready output files

Usage:
    python analyze_networks_metrics.py --graph path/to/network.gt --output-dir results/
"""

import argparse
import logging
import os
import sys
import tempfile

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import graph_tool.all as gt
    from network_metrics_package.metrics.generator import compute_and_save_metrics
    from network_metrics_package.plotting.compare_plots import main as plot_main
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPENDENCIES = False


def setup_logging(verbose=False):
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_network_metrics(graph_path, output_dir, prefix="network", normalize=True, threads=8, plots=None):
    """
    Complete network analysis pipeline.
    
    Args:
        graph_path (str): Path to input graph file
        output_dir (str): Directory for output files
        prefix (str): Prefix for output files
        normalize (bool): Whether to normalize metrics
        threads (int): Number of OpenMP threads
        plots (list): List of plot types to generate
        
    Returns:
        tuple: (metrics_dict, npz_path, csv_path)
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    try:
        G = gt.load_graph(graph_path)
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        raise
    
    logger.info(f"Graph loaded: {G.num_vertices()} vertices, {G.num_edges()} edges")
    
    # Compute metrics
    logger.info("Computing network metrics...")
    metrics, npz_path, csv_path = compute_and_save_metrics(
        G, 
        out_dir=output_dir, 
        prefix=prefix, 
        normalize=normalize, 
        nthreads=threads, 
        save_files=True
    )
    
    logger.info(f"Metrics computed and saved to {npz_path} and {csv_path}")
    
    # Generate plots
    if plots is not None:
        logger.info(f"Generating plots: {', '.join(plots)}")
        try:
            plot_main(npz_path=npz_path, out_dir=output_dir, plots=plots)
            logger.info("Plots generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    return metrics, npz_path, csv_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Network Metrics Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with all plots
  python analyze_networks_metrics.py --graph network.gt --output-dir results/
  
  # Analysis without normalization and specific plots only
  python analyze_networks_metrics.py --graph network.gt --output-dir results/ --no-normalize --plots violin,box
  
  # High-performance analysis with more threads
  python analyze_networks_metrics.py --graph network.gt --output-dir results/ --threads 16
        """
    )
    
    parser.add_argument("--graph", required=True, help="Path to graph-tool graph file (graphml/gt)")
    parser.add_argument("--output-dir", default="metrics_analysis_outputs", help="Output directory for results")
    parser.add_argument("--prefix", default="network", help="Prefix for output files")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable min-max normalization")
    parser.add_argument("--threads", type=int, default=8, help="OpenMP threads for graph-tool")
    parser.add_argument("--plots", nargs="+", choices=["violin", "box", "heatmap", "clustermap"], 
                       default=["violin", "box", "heatmap", "clustermap"],
                       help="Specific plot types to generate (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if not HAS_DEPENDENCIES:
        print("Required dependencies not available. Please install requirements.txt")
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        analyze_network_metrics(
            graph_path=args.graph,
            output_dir=args.output_dir,
            prefix=args.prefix,
            normalize=args.normalize,
            threads=args.threads,
            plots=args.plots
        )
        print(f"Analysis complete! Results in {args.output_dir}/")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()