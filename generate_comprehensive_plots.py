#!/usr/bin/env python3
"""
Generate comprehensive plots including individual joint plots for each metric,
facet grid plots, and multivariate plots as specified in the requirements.
Modified to work without graph-tool by loading degrees and metrics directly from CSV data.
"""
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def plot_individual_joint_plots(aligned_degrees, aligned_metric_arrays, base_name, out_dir="."):
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Create degrees dictionary for compatibility with the original function signature
    aligned_degrees_dict = {i: aligned_degrees[i] for i in range(len(aligned_degrees))}
    
    metric_list = list(aligned_metric_arrays.keys())
    # Use proper colormap access
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(metric_list)))
    metric_colors = {name: c for name, c in zip(metric_list, colors)}
    
    def plot_metric_distribution(degrees_dict, metric_array, metric_name, base_name):
        """Plot individual joint distribution of degree vs metric."""
        degrees_array = np.array([degrees_dict[v] for v in sorted(degrees_dict.keys(), key=int)])
        assert len(degrees_array) == len(metric_array), f"Length mismatch in {metric_name}"

        df = pd.DataFrame({'Degree': degrees_array, metric_name: metric_array})
        
        metric_list = list(aligned_metric_arrays.keys())
        colors = [plt.cm.tab20(i) for i in range(len(metric_list))]  
        metric_colors = {name: tuple(c) for name, c in zip(metric_list, colors)}
        color = metric_colors.get(metric_name, "#000000")  

        valid_data = df.dropna()
        if len(valid_data) < 2:
            logging.warning(f"Skipping {metric_name}: insufficient valid data points ({len(valid_data)})")
            return
            
        # Check if metric has variation
        unique_vals = valid_data[metric_name].unique()
        if len(unique_vals) <= 1:
            logging.warning(f"Skipping {metric_name}: no variation in metric values")
            return
        
        color = metric_colors.get(metric_name, "#000000")  # Default to black if not found

        try:
            # Create joint KDE plot matching seaborn example style with fill=False for better distinction
            g = sns.jointplot(data=valid_data, x='Degree', y=metric_name, kind="kde", 
                              levels=10, color=color, fill=False, alpha=0.7, height=6)
            
            # Use rug plots for marginals to show individual data points
            g.plot_marginals(sns.rugplot, height=0.5, color=color, alpha=0.7)
            
            g.set_axis_labels('Degree', metric_name.replace('_', ' ').title())
            g.fig.suptitle(f"Joint KDE Distribution: Degree vs {metric_name.replace('_', ' ').title()} - {base_name}", y=1.02)
            plt.tight_layout()
            output_path = os.path.join(out_dir, f"{base_name}_jointplot_degree_{metric_name}.png")
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            logging.info(f"Saved individual plot for {metric_name}")
        except Exception as e:
            logging.warning(f"Failed to create plot for {metric_name}: {e}")
            
    for metric_name, metric_values in aligned_metric_arrays.items():
        logging.info(f"{metric_name}: aligned metric length = {len(metric_values)}, degree length = {len(aligned_degrees)}")
        plot_metric_distribution(
            {i: aligned_degrees[i] for i in range(len(aligned_degrees))},
            metric_values,
            metric_name,
            base_name
        )

def plot_facet_grid(aligned_degrees, aligned_metric_arrays, base_name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    
    degrees_array = np.array(aligned_degrees)
    data_list = []

    metric_list = list(aligned_metric_arrays.keys())
    # Use proper colormap access
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(metric_list)))
    metric_colors = {
        name: c for name, c in zip(metric_list, colors)
    }
    
    for metric_name, metric_array in aligned_metric_arrays.items():
        if len(metric_array) != len(degrees_array):
            logging.warning(f"Skipping {metric_name}: length mismatch")
            continue
        
        df = pd.DataFrame({
            'Degree': degrees_array,
            'Centrality': metric_array,
            'Metric': metric_name
        })
        
        # Filter valid data
        valid_df = df.dropna()
        if len(valid_df) < 2:
            logging.warning(f"Skipping {metric_name} in facet grid: insufficient valid data")
            continue
            
        # Check for variation
        unique_vals = valid_df['Centrality'].unique()
        if len(unique_vals) <= 1:
            logging.warning(f"Skipping {metric_name} in facet grid: no variation")
            continue
            
        data_list.append(valid_df)

    if not data_list:
        logging.warning("No valid metric data for faceted plots.")
        return
        
    combined_df = pd.concat(data_list, ignore_index=True)
    
    try:
        g = sns.FacetGrid(
            combined_df,
            col="Metric",
            hue="Metric",
            palette=metric_colors,
            col_wrap=3,
            sharey=False,
            height=4,
            aspect=1.2
        )

        g.map(sns.kdeplot, "Degree", "Centrality", fill=True, alpha=0.5)
        g.map(sns.kdeplot, "Degree", "Centrality", alpha=0.8, levels=15)
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.tight_layout()
        output_path = os.path.join(out_dir, f"{base_name}_facetgrid.png")
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        logging.info(f"Saved facet grid plot")
    except Exception as e:
        logging.warning(f"Failed to create facet grid: {e}")
        
def plot_multivariate_metrics(metrics_data, metric_names, base_name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        df = pd.DataFrame(metrics_data, columns=metric_names)
        
        valid_columns = []
        for col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0 and len(valid_data.unique()) > 1:
                valid_columns.append(col)
                
        if not valid_columns:
            logging.warning("No valid columns with variation for plotting.")
            return
        
        df_valid = df[valid_columns]
        
        if len(df_valid) < 2:
            logging.warning("Insufficient data points for multivariate plot.")
            return
        
        sns.set_theme(style="white")
        # Create PairGrid - attempt KDE plots with scatter fallback for problematic data
        g = sns.PairGrid(df_valid, diag_sharey=False)
        
        def safe_kdeplot(x, y, **kwargs):
            """Attempt KDE plot, fall back to scatter if it fails."""
            try:
                # Check if we have enough unique values for meaningful KDE
                x_unique = len(np.unique(x.dropna()))
                y_unique = len(np.unique(y.dropna()))
                if x_unique <= 2 or y_unique <= 2:
                    # Use scatter plot for low-variation data
                    sns.scatterplot(x=x, y=y, **kwargs)
                else:
                    # Use KDE for data with sufficient variation
                    sns.kdeplot(x=x, y=y, fill=False, alpha=0.7, warn_singular=False, **kwargs)
            except Exception as e:
                # Fall back to scatter plot if KDE fails
                logging.warning(f"KDE failed, using scatter plot: {e}")
                sns.scatterplot(x=x, y=y, **kwargs)
        
        g.map_upper(safe_kdeplot)
        g.map_lower(safe_kdeplot)
        
        # Enhanced diagonal plots with distinct colors and improved clarity
        colors = sns.color_palette("husl", len(valid_columns))
        for i, col in enumerate(valid_columns):
            def diag_plot(x, color=None, label=None, **kwargs):
                # Use distinct color for each histogram
                hist_color = colors[i] if i < len(colors) else None
                sns.histplot(x=x, element="step", linewidth=2, kde=True, 
                           color=hist_color, alpha=0.7, **kwargs)
            
            g.axes[i, i].clear()
            valid_data = df_valid[col].dropna()
            if len(valid_data) > 0:
                diag_plot(valid_data)
                g.axes[i, i].set_title(col, fontsize=10, fontweight='bold')
        
        g.fig.suptitle(f"Multivariate View of Centrality Metrics - {base_name}", y=1.02)
        plt.tight_layout()
        output_path = os.path.join(out_dir, f"{base_name}_multivariate_centralities.png")
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        logging.info(f"Saved multivariate plot")
    except Exception as e:
        logging.error(f"Failed to create multivariate plot: {e}", exc_info=True)

def load_metrics_from_csv(csv_file):
    logging.info(f"Loading metrics from CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    degrees = df['vertex_id'].values
    metrics_dict = {}
    for col in df.columns:
        if col != 'vertex_id':
            metrics_dict[col] = df[col].values
    return degrees, metrics_dict

def generate_all_plots_simple(degrees, metrics, base_name, out_dir="."):
    logging.info("Starting simplified plot generation...")
    logging.info(f"Degrees shape: {degrees.shape}")
    for name, values in metrics.items():
        valid_count = np.sum(~np.isnan(values))
        unique_count = len(np.unique(values[~np.isnan(values)])) if valid_count > 0 else 0
        logging.info(f"{name}: {valid_count} valid values, {unique_count} unique values")
    
    # Create simplified metric arrays for plotting functions
    plot_metric_arrays = {
        'betweenness': metrics.get('betweenness', np.full(len(degrees), np.nan)),
        'pagerank': metrics.get('pagerank', np.full(len(degrees), np.nan)),
        'eigenvector': metrics.get('eigenvector', np.full(len(degrees), np.nan)),
        'closeness': metrics.get('closeness', np.full(len(degrees), np.nan)),
        'hits_authority': metrics.get('hits_authority', np.full(len(degrees), np.nan)),
        'hits_hub': metrics.get('hits_hub', np.full(len(degrees), np.nan)),
        'eigentrust': metrics.get('eigentrust', np.full(len(degrees), np.nan)),
        'katz': metrics.get('katz', np.full(len(degrees), np.nan)),
        'trust_transitivity': metrics.get('trust_transitivity', np.full(len(degrees), np.nan))
    }
    plot_individual_joint_plots(degrees, plot_metric_arrays, base_name, out_dir)
    plot_facet_grid(degrees, plot_metric_arrays, base_name, out_dir)
    
    # Prepare data for multivariate plot
    multivariate_data = []
    multivariate_columns = ['Degree']
    
    # Add Degree as first column
    multivariate_data.append(degrees)
    
    # Add all available metrics
    for metric_name in ['betweenness', 'pagerank', 'eigenvector', 'closeness', 
                       'hits_authority', 'hits_hub', 'eigentrust', 'katz', 'trust_transitivity']:
        if metric_name in metrics:
            multivariate_data.append(metrics[metric_name])
            multivariate_columns.append(metric_name)
    
    # Convert to numpy array with proper shape (rows=vertices, cols=metrics+degree)
    multivariate_array = np.column_stack(multivariate_data)
    
    plot_multivariate_metrics(
        multivariate_array,
        multivariate_columns,
        base_name,
        out_dir
    )

def main():
    """Main function for standalone execution."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate comprehensive plots from metrics files.')
    parser.add_argument('--csv', '-c', type=str, 
                       help='Path to CSV metrics file (e.g., metrics_out/TC2CT_metrics.csv)')
    parser.add_argument('--output-dir', '-o', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--base-name', '-b', type=str,
                       help='Base name for output files (default: derived from CSV filename)')
    
    args = parser.parse_args()
    if args.csv:
        csv_file = args.csv
        if not args.base_name:
            base_name = os.path.splitext(os.path.basename(csv_file))[0].replace('_metrics', '')
        else:
            base_name = args.base_name
    else:
        csv_file = "metrics_out/TC2CT_metrics.csv"
        base_name = "TC2CT"
        if not os.path.exists(csv_file):
            logging.error(f"Default CSV file {csv_file} not found.")
            logging.info("Please provide --csv argument.")
            logging.info("Available CSV files: metrics_out/*.csv")
            return
    try:
        degrees, metrics = load_metrics_from_csv(csv_file)
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return
    
    generate_all_plots_simple(degrees, metrics, base_name, args.output_dir)
    # Resolve the actual output directory path for clearer logging
    resolved_output_dir = os.path.abspath(args.output_dir)
    logging.info(f"All plots generated successfully in {resolved_output_dir}/")


if __name__ == "__main__":
    main()