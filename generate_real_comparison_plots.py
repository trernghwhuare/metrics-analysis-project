#!/usr/bin/env python3
"""
Generate proper comparison plots (box, strip, heatmap, clustermap) for real neuronal circuit models.
This script should be run with: PYTHONPATH=/home/leo520/my/metrics-analysis-project/src pixi run python generate_real_comparison_plots.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Add the src directory to Python path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from network_metrics_package.plotting.compare_plots import (
    load_metrics, plot_strip, plot_box, plot_heatmap_corr
)

def generate_comparison_plots_for_network(npz_file, output_dir="results"):
    """Generate box, strip, heatmap, and clustermap plots for a single network."""
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
    
    # Strip plot (replacing violin plot)
    strip_file = os.path.join(output_dir, f"{base_name}_strip.png")
    plot_strip(metrics, out=strip_file)
    print(f"  Strip plot saved to {strip_file}")
    
    # Correlation heatmap
    heatmap_file = os.path.join(output_dir, f"{base_name}_corr_heatmap.png")
    plot_heatmap_corr(metrics, out=heatmap_file)
    print(f"  Correlation heatmap saved to {heatmap_file}")
    

def main():
    """Generate comparison plots for all networks in metrics_out directory."""
    metrics_dir = "metrics_out"
    output_dir = "results"
    
    # Get all NPZ files
    npz_files = [f for f in os.listdir(metrics_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No NPZ files found in {metrics_dir}")
        return
    
    for npz_file in npz_files:
        full_path = os.path.join(metrics_dir, npz_file)
        generate_comparison_plots_for_network(full_path, output_dir)

if __name__ == "__main__":
    main()