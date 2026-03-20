"""
Network Metrics Analysis Package
"""

from .metrics.generator import compute_and_save_metrics
from .metrics.utils import sanitize_array, minmax_normalize, analyze_network_structure
from .plotting.compare_plots import (
    load_metrics,
    plot_strip,
    plot_box,
    plot_heatmap_corr,
    plot_clustermap,
    main as plot_main
)
from .gt_generator import generate_graph

__version__ = "0.1.0"
__author__ = "Hua Cheng <trernghwhuare@aliyun.com>"

__all__ = [
    'compute_and_save_metrics',
    'sanitize_array', 
    'minmax_normalize',
    'analyze_network_structure',
    'load_metrics',
    'plot_strip',
    'plot_box', 
    'plot_heatmap_corr',
    'plot_clustermap',
    'plot_main',
    'generate_graph'
]