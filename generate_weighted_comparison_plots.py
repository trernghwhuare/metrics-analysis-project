#!/usr/bin/env python3
"""
Generate box, violin, and heatmap plots for weighted networks.
This script directly uses the src/network_metrics_package directory.
"""

import sys
import os

# Add the src directory to Python path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

from network_metrics_package.plotting.compare_plots import (
    load_metrics, plot_violin, plot_box, plot_heatmap_corr, plot_clustermap
)

def generate_weighted_comparison_plots():
    """Generate comparison plots for all weighted networks."""
    weighted_metrics_dir = "metrics_out_weighted_random_uniform"
    output_dir = "results_weighted"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files
    npz_files = [f for f in os.listdir(weighted_metrics_dir) if f.endswith('.npz')]
    
    for npz_file in npz_files:
        base_name = npz_file.replace('_weighted_random_uniform_metrics.npz', '')
        full_path = os.path.join(weighted_metrics_dir, npz_file)
        
        print(f"Generating comparison plots for {base_name}...")
        
        # Load metrics
        metrics = load_metrics(full_path)
        
        # Box plot
        box_file = os.path.join(output_dir, f"{base_name}_box.png")
        plot_box(metrics, out=box_file)
        print(f"  Box plot saved to {box_file}")
        
        # Violin plot  
        violin_file = os.path.join(output_dir, f"{base_name}_violin.png")
        plot_violin(metrics, out=violin_file)
        print(f"  Violin plot saved to {violin_file}")
        
        # Correlation heatmap
        heatmap_file = os.path.join(output_dir, f"{base_name}_corr_heatmap.png")
        plot_heatmap_corr(metrics, out=heatmap_file)
        print(f"  Correlation heatmap saved to {heatmap_file}")
        
        # Clustermap (only if we have enough valid metrics)
        try:
            clustermap_file = os.path.join(output_dir, f"{base_name}_clustermap.png")
            plot_clustermap(metrics, out=clustermap_file)
            print(f"  Clustermap saved to {clustermap_file}")
        except Exception as e:
            print(f"  Warning: Could not generate clustermap for {base_name}: {e}")

if __name__ == "__main__":
    generate_weighted_comparison_plots()