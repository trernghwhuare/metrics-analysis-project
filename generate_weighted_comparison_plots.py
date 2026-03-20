#!/usr/bin/env python3
"""
Generate box, strip, and heatmap plots for weighted networks.
This script directly uses the src/network_metrics_package directory.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

from network_metrics_package.plotting.compare_plots import (
    load_metrics, plot_strip, plot_box, plot_heatmap_corr
)

def generate_weighted_comparison_plots():
    """Generate comparison plots for all weighted networks."""
    # Use absolute paths based on script location
    weighted_metrics_dir = os.path.join(script_dir, "metrics_out")
    output_dir = os.path.join(script_dir, "results_weighted")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files
    npz_files = [f for f in os.listdir(weighted_metrics_dir) if f.endswith('.npz')]
    
    for npz_file in npz_files:
        base_name = npz_file.replace('_metrics.npz', '')
        full_path = os.path.join(weighted_metrics_dir, npz_file)
        
        print(f"Generating comparison plots for {base_name}...")
        
        # Load metrics
        metrics = load_metrics(full_path)
        
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

if __name__ == "__main__":
    generate_weighted_comparison_plots()