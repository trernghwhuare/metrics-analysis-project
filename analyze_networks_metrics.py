#!/usr/bin/env python3
"""
Script to analyze all your network files with the network_metrics_package.
Author: Hua Cheng <trernghwhuare@aliyun.com>
"""

import sys
import os
import glob

# Add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import graph_tool.all as gt
from network_metrics_package.metrics.generator import compute_and_save_metrics
from network_metrics_package.plotting.compare_plots import (
    load_metrics, plot_violin, plot_box, plot_heatmap_corr, plot_clustermap
)

# List of your network files
network_files = [
    "iT_max_plus.gt",
    "max_CTC_plus.gt", 
    "max_M2M1S1_plus.gt",
    "optimus_CTC_plus.gt",
    "optimus_M2M1S1_plus.gt"
]

def analyze_network_file(filepath, output_base_dir="your_networks_analysis"):
    """
    Analyze a single network file.
    
    Parameters:
    filepath (str): Path to the network file
    output_base_dir (str): Base directory for all outputs
    """
    # Extract prefix from filename
    prefix = os.path.splitext(os.path.basename(filepath))[0]
    
    # Create output directory for this network
    output_dir = os.path.join(output_base_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing {filepath}...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load the network
        g = gt.load_graph(filepath)
        print(f"  Network loaded: {g.num_vertices()} vertices, {g.num_edges()} edges")
        
        # Compute metrics
        print("  Computing metrics...")
        metrics, npz_path, csv_path = compute_and_save_metrics(
            g,
            out_dir=output_dir,
            prefix=prefix,
            normalize=True,
            save_files=True
        )
        print(f"  Metrics saved to {npz_path} and {csv_path}")
        
        # Load metrics for plotting
        loaded_metrics = load_metrics(npz_path)
        
        # Generate plots
        print("  Generating visualizations...")
        
        # Violin plot
        violin_file = os.path.join(output_dir, f"{prefix}_violin.png")
        plot_violin(loaded_metrics, out=violin_file)
        print(f"  Violin plot saved to {violin_file}")
        
        # Box plot
        box_file = os.path.join(output_dir, f"{prefix}_box.png")
        plot_box(loaded_metrics, out=box_file)
        print(f"  Box plot saved to {box_file}")
        
        # Correlation heatmap
        heatmap_file = os.path.join(output_dir, f"{prefix}_heatmap_corr.png")
        plot_heatmap_corr(loaded_metrics, out=heatmap_file)
        print(f"  Correlation heatmap saved to {heatmap_file}")
        
        # Clustermap
        clustermap_file = os.path.join(output_dir, f"{prefix}_clustermap.png")
        plot_clustermap(loaded_metrics, out=clustermap_file)
        print(f"  Clustermap saved to {clustermap_file}")
        
        print(f"  Analysis complete for {filepath}!")
        return True
        
    except Exception as e:
        print(f"  Error analyzing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to analyze all network files.
    """
    print("Starting analysis of your network files...")
    print("=" * 50)
    
    # Create base output directory
    base_output_dir = "your_networks_analysis"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Counter for successful analyses
    success_count = 0
    
    # Analyze each network file
    for network_file in network_files:
        if os.path.exists(network_file):
            print(f"\nProcessing {network_file}...")
            if analyze_network_file(network_file, base_output_dir):
                success_count += 1
        else:
            print(f"\nWarning: {network_file} not found, skipping...")
    
    print("\n" + "=" * 50)
    print(f"Analysis complete! {success_count} out of {len(network_files)} networks processed successfully.")
    print(f"Results are in the '{base_output_dir}' directory, with each network in its own subdirectory.")

if __name__ == "__main__":
    main()