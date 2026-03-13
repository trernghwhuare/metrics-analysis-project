#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.plotting.compare_plots import load_metrics, plot_violin

if __name__ == '__main__':
    # Load metrics
    metrics_path = os.path.join('..', 'metrics_out', 'max_CTC_plus_metrics.npz')
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
        
    metrics = load_metrics(metrics_path)
    print(f"Available metrics: {list(metrics.keys())}")
    
    # Print shape and type information for each metric
    for k, v in metrics.items():
        arr = np.array(v)
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Try to create violin plot
    try:
        output_path = os.path.join('..', 'metrics_out', 'test_violin_fixed.png')
        plot_violin(metrics, out=output_path)
        print(f"Violin plot created successfully: {output_path}")
    except Exception as e:
        print(f"Error creating violin plot: {e}")
        import traceback
        traceback.print_exc()