#!/usr/bin/env python3
"""
Generate proper comparison plots (box, violin, heatmap, clustermap) for real neuronal circuit models.
This script uses the plotting functions from network_metrics_package.plotting.compare_plots
to create the exact plot types needed for Figure 4 in the paper.
"""

import os
import numpy as np
import pandas as pd
from network_metrics_package.plotting.compare_plots import (
    load_metrics, plot_violin, plot_box, plot_heatmap_corr, plot_clustermap
)

def generate_comparison_plots_for_network(npz_file, output_dir="results"):
    """Generate box, violin, heatmap, and clustermap plots for a single network."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics from NPZ file
    metrics = load_metrics(npz_file)
    
    # Get base name from filename
    base_name = os.path.splitext(os.path.basename(npz_file))[0].replace('_metrics', '')
    
    # Generate all four plot types
    print(f"Generating comparison plots for {base_name}...")
    
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
    
    return {
        'box': box_file,
        'violin': violin_file, 
        'heatmap': heatmap_file,
        'clustermap': clustermap_file if 'clustermap_file' in locals() else None
    }

def main():
    """Generate comparison plots for all four real neuronal circuit models."""
    networks = [
        "metrics_out/TC2CT_metrics.npz",
        "metrics_out/TC2IT2PTCT_metrics.npz", 
        "metrics_out/TC2IT4_IT2CT_metrics.npz",
        "metrics_out/TC2PT_metrics.npz"
    ]
    
    all_plots = {}
    
    for network_file in networks:
        if os.path.exists(network_file):
            plots = generate_comparison_plots_for_network(network_file, "results")
            network_name = os.path.splitext(os.path.basename(network_file))[0].replace('_metrics', '')
            all_plots[network_name] = plots
        else:
            print(f"Warning: {network_file} not found")
    
    print("\nAll comparison plots generated successfully!")
    print("Files are available in the 'results/' directory:")
    for network, plots in all_plots.items():
        print(f"\n{network}:")
        for plot_type, plot_file in plots.items():
            if plot_file and os.path.exists(plot_file):
                print(f"  {plot_type}: {os.path.basename(plot_file)}")

if __name__ == "__main__":
    main()