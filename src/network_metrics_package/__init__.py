"""
Network Metrics Analysis Package
================================

A package for analyzing neural network structures using graph theory metrics.
"""

__version__ = "0.1.0"
__author__ = "Hua Cheng"

# Import main functions for easy access
from .metrics.generator import compute_and_save_metrics
from .plotting.compare_plots import load_metrics, plot_violin, plot_box, plot_heatmap_corr, plot_clustermap

__all__ = [
    "compute_and_save_metrics",
    "load_metrics", 
    "plot_violin",
    "plot_box",
    "plot_heatmap_corr",
    "plot_clustermap"
]