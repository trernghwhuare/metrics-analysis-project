import os
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cbook


sns.set(style="whitegrid")

def load_metrics(npz_path):
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def plot_violin(metrics_dict, metric_names=None, title="Metric correlation", out=None, figsize=(10,8)):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
        
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
        
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
        print("No valid data for violin plot")
        return
    
    # Prepare data for violin plot - ensure each dataset is a proper 1D array
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
        print("No valid finite data for violin plot")
        return
    
    # Check if we should use log scale based on data range
    try:
        # Concatenate all data to check range
        all_data = np.concatenate(data_list)
        # Only consider positive values for log scaling
        positive_data = all_data[all_data > 0]
        use_log_scale = False
        if len(positive_data) > 0:
            data_range = np.log10(np.max(positive_data)) - np.log10(np.min(positive_data))
            use_log_scale = data_range > 2  # Use log scale if data spans more than 2 orders of magnitude
        
        # Transform data for log scale if needed
        if use_log_scale:
            transformed_data = []
            for d in data_list:
                positive_d = d[d > 0]
                if len(positive_d) > 0:
                    # Use log1p instead of log10 to avoid negative values for small positive numbers
                    transformed_data.append(np.log1p(positive_d))
                else:
                    # If no positive values, create a small array with a default value
                    transformed_data.append(np.array([0]))
            data_list = transformed_data
    except Exception as e:
        print(f"Warning: Could not determine log scaling: {e}")
        use_log_scale = False

    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plots for each metric individually
    try:
        parts = ax.violinplot(data_list, positions=range(1, len(data_list) + 1), 
                              showmeans=True, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('pink')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        # Calculate quartiles and whiskers for each dataset individually
        quartile1, medians, quartile3 = [], [], []
        whiskers = []
        
        for d in data_list:
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
            
            # Calculate whiskers
            lower_whisker, upper_whisker = adjacent_values(np.sort(d), q1, q3)
            whiskers.append([lower_whisker, upper_whisker])
            
        quartile1 = np.array(quartile1)
        medians = np.array(medians)
        quartile3 = np.array(quartile3)
        whiskers = np.array(whiskers)
        
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # Set axis style and labels
        set_axis_style(ax, valid_metric_names)
        ax.set_xticklabels(valid_metric_names, rotation=45, ha='right')
        
        # Set y-axis scale and label
        if use_log_scale:
            ax.set_ylabel('Value (log1p scale)')
            # Add information text about scaling
            ax.text(0.02, 0.98, 'Log1p scale', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.set_ylabel('Value')
        
        ax.set_title('Violin plot')
        
        plt.tight_layout()
        if out: plt.savefig(out); plt.close()
    except Exception as e:
        print(f"Error creating violin plot: {e}")
        import traceback
        traceback.print_exc()
        plt.close()

def plot_box(metrics_dict, metric_names=None, title="Metric correlation (boxplot)", out=None, figsize=(10,8)):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    
    df = pd.DataFrame({k: metrics_dict[k] for k in metric_names})
    np.random.seed(19680801)
    
    # Process data to ensure all arrays have the same length
    used_data = []
    valid_metric_names = []
    for k in metric_names:
        series = df[k].dropna()
        if len(series) > 0:
            used_data.append(np.array(series))
            valid_metric_names.append(k)
    
    if not used_data:
        print("No valid data for box plot")
        return
    
    # Check if we should use log scale based on data range
    all_data = np.concatenate(used_data)
    # Only consider positive values for log scaling
    positive_data = all_data[all_data > 0]
    use_log_scale = False
    if len(positive_data) > 0:
        data_range = np.log10(np.max(positive_data)) - np.log10(np.min(positive_data))
        use_log_scale = data_range > 2  # Use log scale if data spans more than 2 orders of magnitude
    
    # Transform data for log scale if needed
    if use_log_scale:
        transformed_data = []
        for data in used_data:
            positive_data = data[data > 0]
            if len(positive_data) > 0:
                transformed_data.append(np.log10(positive_data))
            else:
                transformed_data.append(np.array([0]))
        used_data = transformed_data
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Metric distributions")
    ax.set_xlabel("Metric")
    
    # Create box plots for each metric
    bp = ax.boxplot(used_data, positions=range(1, len(used_data) + 1),
                    patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red'), 
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), 
                    flierprops=dict(markerfacecolor='gray', marker='o', markersize=5, 
                                  linestyle='none', markeredgecolor='black'))
    
    # Set y-axis scale and label
    if use_log_scale:
        ax.set_ylabel('Value (log10 scale)')
        # Add information text about scaling
        ax.text(0.02, 0.98, 'Log10 scale', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.set_ylabel('Value')
    
    ax.set_xticklabels(valid_metric_names, rotation=45, ha='right')
    plt.tight_layout()
    if out: plt.savefig(out); plt.close()
    
def plot_heatmap_corr(metrics_dict, metric_names=None, title="Metric correlation (heatmap)", out=None, figsize=(8,8), annot=False):
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    df = pd.DataFrame({k: metrics_dict[k] for k in metric_names})
    
    # Filter out columns with no valid data
    valid_cols = [c for c in df.columns if df[c].notna().any() and (df[c].std() > 0 or np.nanstd(df[c]) > 0)]
    if len(valid_cols) == 0:
        print("No valid columns for heatmap")
        return
    elif len(valid_cols) == 1:
        print("Only one valid column for heatmap")
        # Create a 1x1 heatmap
        df_single = df[[valid_cols[0]]].to_frame()
        corr = df_single.corr(method='spearman')
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=annot, cmap="vlag", center=0, square=True, linewidths=.5)
        plt.title(title)
        plt.tight_layout()
        if out: plt.savefig(out); plt.close()
        return
        
    df = df.loc[:, valid_cols]
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    if len(df) == 0:
        print("No valid rows for heatmap")
        return
    
    corr = df.corr(method='spearman')
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr.values, 1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap="vlag", center=0, square=True, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    if out: plt.savefig(out); plt.close()


def plot_clustermap(metrics_dict, metric_names=None, title="Metric clustermap", out=None, figsize=(10,10)):
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
    
    # Compute correlation matrix, handling NaN values
    corr = df.corr(method='spearman')
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr.values, 1.0)
    
    try:
        cg = sns.clustermap(corr, cmap="vlag", figsize=figsize, annot=True)
        plt.suptitle(title)
        if out:
            cg.savefig(out)
            plt.close()
    except Exception as e:
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap="vlag", center=0)
        plt.title(title + " (fallback heatmap)")
        if out:
            plt.savefig(out)
            plt.close()


def save_plot(filename):
    plt.savefig(filename)
    plt.close()

def main(npz_path=None, out_dir="metrics_out", plots=None):
    if npz_path is None:
        files = [f for f in os.listdir(os.path.join(os.getcwd(), "metrics_out")) if f.endswith(".npz")]
        print(files)
        if not files:
            raise FileNotFoundError("No .npz files found in metrics_out directory.")
        for filename in files:
            npz_path = os.path.join(os.getcwd(), "metrics_out", filename)
            base_name = os.path.splitext(os.path.basename(npz_path))[0]
            os.makedirs(out_dir, exist_ok=True)
            try:
                metrics = load_metrics(npz_path)
            except Exception as e:
                print(f"Error loading metrics from {npz_path}: {e}")
                continue
                
            if plots is None:
                plots = ["violin", "box", "heatmap", "clustermap"]

            if "violin" in plots:
                try:
                    plot_violin(metrics, out=os.path.join(out_dir, f"{base_name}_violin.png"))
                except Exception as e:
                    print(f"Error creating violin plot for {base_name}: {e}")
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
    else:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        os.makedirs(out_dir, exist_ok=True)
        try:
            metrics = load_metrics(npz_path)
        except Exception as e:
            print(f"Error loading metrics from {npz_path}: {e}")
            return
            
        if plots is None:
            plots = ["violin", "box", "heatmap", "clustermap"]

        if "violin" in plots:
            try:
                plot_violin(metrics, out=os.path.join(out_dir, f"{base_name}_violin.png"))
            except Exception as e:
                print(f"Error creating violin plot for {base_name}: {e}")
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

if __name__ == "__main__":
    # simple CLI: set METRICS_NPZ and comma-separated PLOTS env vars or edit defaults here
    npz = os.environ.get("METRICS_NPZ", None)
    plots_env = os.environ.get("PLOTS", "violin,box,heatmap,clustermap")
    plots_list = [p.strip() for p in plots_env.split(",") if p.strip()]
    main(npz_path=npz, out_dir="metrics_out", plots=plots_list)
