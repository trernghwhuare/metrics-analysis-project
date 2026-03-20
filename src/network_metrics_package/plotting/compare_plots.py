"""
Network Metrics Visualization
Author: Hua Cheng <trernghwhuare@aliyun.com>
"""

import os
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cbook
import logging


sns.set(style="whitegrid")

def load_metrics(npz_path):
    data = np.load(npz_path)
    # Create writeable copies of all arrays to avoid read-only issues
    return {k: np.array(data[k], copy=True) for k in data.files}


def plot_strip(metrics_dict, metric_names=None, title="Metric correlation", out=None, figsize=(10,8)):
    """
    Create strip plots (scatter plots) for network metrics.
    Strip plots are better suited for sparse, skewed data with many zeros.
    """
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    
    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
    
    # Filter out metrics with no valid data and ensure all arrays are proper 1D arrays
    valid_metrics = {}
    valid_metric_names = []
    
    for name in metric_names:
        try:
            # Get the data and convert to numpy array
            data = np.array(metrics_dict[name])
            
            # Handle object arrays (arrays with inhomogeneous shapes)
            if data.dtype == object:
                # Flatten all elements and combine into a single 1D array
                flattened = []
                for item in data.flat:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        flattened.extend(np.asarray(item).flatten())
                    else:
                        flattened.append(item)
                data = np.array(flattened, dtype=float)
            else:
                # For regular arrays, just flatten to 1D
                data = data.flatten()
            
            # Remove NaN values
            data = data[~np.isnan(data)]
            
            # Only include metrics with at least one valid value
            if len(data) > 0:
                valid_metrics[name] = data
                valid_metric_names.append(name)
        except Exception as e:
            print(f"Warning: Could not process metric {name}: {e}")
            continue
    
    # Check if we have any valid metrics
    if not valid_metric_names:
        print("No valid data for strip plot")
        return
    
    # Prepare data for strip plot - ensure each dataset is a proper 1D array
    data_list = []
    for name in valid_metric_names:
        metric_data = valid_metrics[name]
        # Make sure it's a proper 1D array of finite values
        if metric_data.ndim > 1:
            metric_data = metric_data.flatten()
        
        # Filter out infinite values as well as NaN
        metric_data = metric_data[np.isfinite(metric_data)]
        
        if len(metric_data) > 0:
            data_list.append(metric_data)
        else:
            print(f"Warning: No finite data for metric {name}")
    
    # If no data left after filtering, exit
    if not data_list:
        print("No valid finite data for strip plot")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # Create strip plot (scatter plot) for each metric
    for i, (data, label) in enumerate(zip(data_list, valid_metric_names)):
        x_positions = np.full(len(data), i + 1)
        # Add small random jitter to x positions to avoid overlapping points
        x_jitter = x_positions + np.random.normal(0, 0.05, len(x_positions))
        ax.scatter(x_jitter, data, alpha=0.6, s=10, color='#D43F3A', edgecolors='black', linewidth=0.5)
    
    set_axis_style(ax, valid_metric_names)
    ax.set_title(title)
    plt.tight_layout()
    if out:
        plt.savefig(out)
    plt.close(fig)


def plot_box(metrics_dict, metric_names=None, title="Metric Box Plot", out=None, figsize=(10,8)):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    
    valid_metrics = {}
    valid_metric_names = []
    
    for name in metric_names:
        try:
            data = np.array(metrics_dict[name])
            if data.dtype == object:
                flattened = []
                for item in data.flat:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        flattened.extend(np.asarray(item).flatten())
                    else:
                        flattened.append(item)
                data = np.array(flattened, dtype=float)
            else:
                data = data.flatten()
            
            data = data[~np.isnan(data)]
            if len(data) > 0:
                valid_metrics[name] = data
                valid_metric_names.append(name)
        except Exception as e:
            print(f"Warning: Could not process metric {name}: {e}")
            continue
    
    if not valid_metric_names:
        print("No valid data for box plot")
        return
    
    # Prepare data for box plot
    data_list = []
    for name in valid_metric_names:
        metric_data = valid_metrics[name]
        if metric_data.ndim > 1:
            metric_data = metric_data.flatten()
        metric_data = metric_data[np.isfinite(metric_data)]
        if len(metric_data) > 0:
            data_list.append(metric_data)
        else:
            print(f"Warning: No finite data for metric {name}")
    
    if not data_list:
        print("No valid finite data for box plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data_list, labels=valid_metric_names)
    ax.set_title(title)
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if out:
        plt.savefig(out)
    plt.close(fig)


def plot_heatmap_corr(metrics_dict, metric_names=None, title="Metric Correlation Heatmap", out=None, annot=True, figsize=(10,8)):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    
    # Create DataFrame with all metrics
    df_data = {}
    for name in metric_names:
        try:
            data = np.array(metrics_dict[name])
            if data.dtype == object:
                # Handle object arrays by taking first element or flattening
                if len(data) > 0:
                    if isinstance(data[0], (list, tuple, np.ndarray)):
                        data = np.array([item[0] if len(item) > 0 else np.nan for item in data if isinstance(item, (list, tuple, np.ndarray))])
                    else:
                        data = data.astype(float)
            df_data[name] = data
        except Exception as e:
            print(f"Warning: Could not process metric {name} for correlation: {e}")
            continue
    
    if not df_data:
        print("No valid data for correlation heatmap")
        return
    
    df = pd.DataFrame(df_data)
    
    # Remove rows with all NaN
    df = df.dropna(how='all')
    
    if len(df) == 0:
        print("No valid rows for correlation heatmap")
        return
    
    # Compute correlation matrix
    try:
        corr = df.corr(method='spearman')
    except Exception as e:
        print(f"Error computing correlation: {e}")
        return
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if out:
        plt.savefig(out)
    plt.close(fig)


def plot_clustermap(metrics_dict, metric_names=None, title="Metric Clustermap", out=None, figsize=(10,8)):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    df = pd.DataFrame({k: metrics_dict[k] for k in metric_names})
    
    # Filter out columns with no valid data
    valid_cols = [c for c in df.columns if df[c].notna().any() and (df[c].std() > 0 or np.nanstd(df[c]) > 0)]
    if len(valid_cols) == 0:
        print("No valid columns for clustermap")
        return
    elif len(valid_cols) == 1:
        print("Only one valid column for clustermap, skipping")
        return
        
    df = df.loc[:, valid_cols]
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    if len(df) == 0:
        print("No valid rows for clustermap")
        return
    
    # Create a completely new DataFrame with writeable arrays to avoid read-only issues
    df_writeable = pd.DataFrame()
    for col in df.columns:
        # Ensure each column is a writeable numpy array
        col_data = np.array(df[col].values, copy=True)
        col_data.flags.writeable = True
        df_writeable[col] = col_data
    
    # Compute correlation matrix, handling NaN values
    corr = df.corr(method='spearman')
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2.0
    
    # Make sure the correlation matrix values are writeable before filling diagonal
    corr_values = np.array(corr.values, copy=True)
    corr_values.flags.writeable = True
    np.fill_diagonal(corr_values, 1.0)
    
    # Create a new DataFrame with the fixed values
    corr_fixed = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    
    try:
        cg = sns.clustermap(corr_fixed, cmap="vlag", figsize=figsize, annot=True)
        plt.suptitle(title)
        if out:
            cg.savefig(out)
            plt.close()
    except Exception as e:
        plt.figure(figsize=figsize)
        sns.heatmap(corr_fixed, annot=True, cmap="vlag", center=0)
        plt.title(title + " (fallback heatmap)")
        if out:
            plt.savefig(out)
            plt.close()


def plot_facetgrid_degree_centrality_from_npz(npz_path, base_name, out_dir=".", figsize=(12, 8)):
    """
    Generate facet grid plots from NPZ file containing metrics.
    This assumes the NPZ file contains both 'degrees' and centrality metrics.
    """
    metrics = load_metrics(npz_path)
    
    # Check if degrees are available
    if 'degrees' not in metrics:
        logging.warning("No 'degrees' found in metrics file. Skipping facet grid plot.")
        return
    
    degrees = metrics['degrees']
    nc = len(degrees)
    logging.info(f"Using all components: total vertices = {nc}")
    
    if nc == 0:
        logging.warning("Largest component empty — skipping all metric plots.")
        return
    
    # Define the metrics to plot
    metric_mapping = {
        'bt': 'betweenness',
        'pr': 'pagerank', 
        'V': 'eigenvector',
        'katz': 'katz',
        'hitsX': 'hits_authority',
        'hitsY': 'hits_hub',
        't': 'eigentrust',
        'tt': 'trust_transitivity',
        'c': 'closeness'
    }
    
    # Helper function to align metric to degree indices and filter valid values
    def _align_mask(metric_arr):
        arr = np.array(metric_arr, dtype=float)
        if len(arr) != len(degrees):
            logging.warning(f"Metric array length {len(arr)} doesn't match degrees length {len(degrees)}")
            return None
            
        valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
        if not np.any(valid_mask):
            return None
        return degrees[valid_mask], arr[valid_mask]
    
    # Collect per-metric aligned pairs (degree, metric)
    aligned_metric_arrays = {}
    for short_name, full_name in metric_mapping.items():
        if full_name in metrics:
            arr = metrics[full_name]
            pair = _align_mask(arr)
            if pair is not None and len(pair[0]) > 0:
                aligned_metric_arrays[short_name] = pair
    
    logging.info(f"Number of valid metric series: {len(aligned_metric_arrays)}")
    logging.info(f"Total vertices considered: {nc}")

    # Create facet plot
    data_list = []
    for metric_name, (deg_arr, metric_arr) in aligned_metric_arrays.items():
        df = pd.DataFrame({'Degree': deg_arr, 'Centrality': metric_arr, 'Metric': metric_name})
        data_list.append(df)

    if not data_list:
        logging.warning("No valid metric data for faceted plots.")
        return
        
    combined_df = pd.concat(data_list, ignore_index=True)
    sns.set_style('whitegrid')
    g = sns.FacetGrid(combined_df, col='Metric', hue='Metric',
                      palette='tab20', sharey=False, height=3, aspect=1.2, col_wrap=3)
    g.map(sns.scatterplot, 'Degree', 'Centrality', alpha=0.7, s=20)
    try:
        g.map(sns.kdeplot, 'Degree', 'Centrality', fill=True, levels=10, alpha=0.3, warn_singular=False)
    except Exception:
        pass
    g.add_legend()
    g.set_axis_labels('Degree', 'Centrality')
    g.fig.suptitle(f"{base_name}", fontsize=12)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{base_name}_facetgrid.png")
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    logging.info(f"Saved facet grid plot to {output_path}")


def main(npz_path=None, out_dir="metrics_out", plots=None):
    """
    Main function to generate various plots from metrics files.
    
    Parameters:
    -----------
    npz_path : str or None
        Path to a single .npz file, or None to process all in metrics_out/
    out_dir : str
        Output directory for plots
    plots : list or None
        List of plot types to generate: ['strip', 'box', 'heatmap', 'clustermap', 'facetgrid']
    """
    if plots is None:
        plots = ["strip", "box", "heatmap", "clustermap"]
    
    if npz_path is None:
        # Process all .npz files in metrics_out directory
        metrics_dir = "metrics_out"
        if not os.path.exists(metrics_dir):
            print(f"Metrics directory {metrics_dir} not found")
            return
            
        files = [f for f in os.listdir(metrics_dir) if f.endswith('.npz')]
        if not files:
            print(f"No .npz files found in {metrics_dir}")
            return
            
        for filename in files:
            npz_path = os.path.join(os.getcwd(), "metrics_out", filename)
            base_name = os.path.splitext(os.path.basename(npz_path))[0]
            os.makedirs(out_dir, exist_ok=True)
            try:
                metrics = load_metrics(npz_path)
            except Exception as e:
                print(f"Error loading metrics from {npz_path}: {e}")
                continue
                
            if "strip" in plots: 
                try:
                    plot_strip(metrics, out=os.path.join(out_dir, f"{base_name}_strip.png"))
                except Exception as e:
                    print(f"Error creating strip plot for {base_name}: {e}")
            if "box" in plots:
                try:
                    plot_box(metrics, out=os.path.join(out_dir, f"{base_name}_box.png"))
                except Exception as e:
                    print(f"Error creating box plot for {base_name}: {e}")
            if "heatmap" in plots:
                try:
                    plot_heatmap_corr(metrics, out=os.path.join(out_dir, f"{base_name}_corr_heatmap.png"), annot=True)
                except Exception as e:
                    print(f"Error creating heatmap for {base_name}: {e}")
            if "clustermap" in plots:
                try:
                    plot_clustermap(metrics, out=os.path.join(out_dir, f"{base_name}_clustermap.png"))
                except Exception as e:
                    print(f"Error creating clustermap for {base_name}: {e}")
            if "facetgrid" in plots:
                try:
                    plot_facetgrid_degree_centrality_from_npz(npz_path, base_name, out_dir)
                except Exception as e:
                    print(f"Error creating facetgrid plot for {base_name}: {e}")
    else:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        os.makedirs(out_dir, exist_ok=True)
        try:
            metrics = load_metrics(npz_path)
        except Exception as e:
            print(f"Error loading metrics from {npz_path}: {e}")
            return
            
        if "strip" in plots:
            try:
                plot_strip(metrics, out=os.path.join(out_dir, f"{base_name}_strip.png"))
            except Exception as e:
                print(f"Error creating strip plot for {base_name}: {e}")
        if "box" in plots:
            try:
                plot_box(metrics, out=os.path.join(out_dir, f"{base_name}_box.png"))
            except Exception as e:
                print(f"Error creating box plot for {base_name}: {e}")
        if "heatmap" in plots:
            try:
                plot_heatmap_corr(metrics, out=os.path.join(out_dir, f"{base_name}_corr_heatmap.png"), annot=True)
            except Exception as e:
                print(f"Error creating heatmap for {base_name}: {e}")
        if "clustermap" in plots:
            try:
                plot_clustermap(metrics, out=os.path.join(out_dir, f"{base_name}_clustermap.png"))
            except Exception as e:
                print(f"Error creating clustermap for {base_name}: {e}")
        if "facetgrid" in plots:
            try:
                plot_facetgrid_degree_centrality_from_npz(npz_path, base_name, out_dir)
            except Exception as e:
                print(f"Error creating facetgrid plot for {base_name}: {e}")


if __name__ == "__main__":
    # simple CLI: set METRICS_NPZ and comma-separated PLOTS env vars or edit defaults here
    npz = os.environ.get("METRICS_NPZ", None)
    plots_env = os.environ.get("PLOTS", "strip,box,heatmap,clustermap,facetgrid")
    plots_list = [p.strip() for p in plots_env.split(",") if p.strip()]
    main(npz_path=npz, out_dir="metrics_out", plots=plots_list)