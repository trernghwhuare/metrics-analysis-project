import numpy as np
import logging

logger = logging.getLogger(__name__)

def sanitize_array(arr):
    """Convert input to numpy float array and replace +/-inf with nan.
    Accept graph-tool vertex property objects (has .get_array or .a)."""
    try:
        if hasattr(arr, "get_array"):
            a = np.asarray(arr.get_array(), dtype=float)
        elif hasattr(arr, "a"):
            a = np.asarray(arr.a, dtype=float)
        else:
            a = np.asarray(arr, dtype=float)
    except Exception:
        # fallback to best-effort conversion
        a = np.asarray(arr, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a

def minmax_normalize(arr):
    """Min-max normalize 1D array, preserving nan values."""
    a = sanitize_array(arr)
    valid = ~np.isnan(a)
    if not np.any(valid):
        return a
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if mx == mn:
        a[valid] = 0.0
        return a
    a[valid] = (a[valid] - mn) / (mx - mn)
    return a

def analyze_network_structure(metrics_dict):
    """
    Analyze network structure based on computed metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metric arrays from compute_and_save_metrics
        
    Returns:
        dict: Structural analysis results including correlations, distributions, and insights
    """
    import pandas as pd
    
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics_dict)
    
    # Remove rows with all NaN values
    df_clean = df.dropna(how='all')
    
    if len(df_clean) == 0:
        return {
            'correlations': {},
            'summary_stats': {},
            'insights': ['No valid data available for analysis']
        }
    
    # Compute correlations between metrics
    correlations = df_clean.corr(method='spearman').to_dict()
    
    # Compute summary statistics
    summary_stats = {}
    for column in df_clean.columns:
        series = df_clean[column].dropna()
        if len(series) > 0:
            summary_stats[column] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'count': int(len(series)),
                'nan_count': int(len(df[column]) - len(series))
            }
        else:
            summary_stats[column] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0,
                'nan_count': int(len(df[column]))
            }
    
    # Generate basic insights
    insights = []
    if len(correlations) > 1:
        # Find highly correlated metrics using numpy for safer handling
        try:
            corr_matrix = np.array([[correlations[col1][col2] for col2 in correlations.keys()] 
                                   for col1 in correlations.keys()])
            corr_cols = list(correlations.keys())
            high_corr_pairs = []
            for i in range(len(corr_cols)):
                for j in range(i+1, len(corr_cols)):
                    corr_val = corr_matrix[i, j]
                    if isinstance(corr_val, (int, float)) and not np.isnan(corr_val):
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append((corr_cols[i], corr_cols[j], corr_val))
            
            if high_corr_pairs:
                insights.append(f"Found {len(high_corr_pairs)} pairs of highly correlated metrics (|r| > 0.7)")
        except Exception as e:
            logging.warning(f"Failed to compute correlation pairs: {e}")
            pass
    
    # Check for metrics with low variance
    low_variance_metrics = []
    for metric, stats in summary_stats.items():
        if stats['std'] < 0.1 and stats['count'] > 0:
            low_variance_metrics.append(metric)
    
    if low_variance_metrics:
        insights.append(f"Metrics with low variance (< 0.1): {', '.join(low_variance_metrics)}")
    
    # Check for metrics with high NaN counts
    high_nan_metrics = []
    total_vertices = len(df)
    for metric, stats in summary_stats.items():
        if stats['nan_count'] > 0:
            nan_ratio = stats['nan_count'] / total_vertices
            if nan_ratio > 0.5:  # More than 50% NaN
                high_nan_metrics.append(f"{metric} ({nan_ratio:.1%} NaN)")
    
    if high_nan_metrics:
        insights.append(f"Metrics with high NaN ratios: {', '.join(high_nan_metrics)}")
    
    if not insights:
        insights.append("No specific structural patterns detected")
    
    return {
        'correlations': correlations,
        'summary_stats': summary_stats,
        'insights': insights
    }

def diagnose_centrality_issues(metrics_dict, graph_info=None):
    """
    Diagnose potential issues with centrality metrics that result in NaN or zero values.
    
    Args:
        metrics_dict (dict): Dictionary of metric arrays
        graph_info (dict): Optional graph information (vertices, edges, components)
        
    Returns:
        dict: Diagnosis results with recommendations
    """
    diagnosis = {
        'issues': [],
        'recommendations': [],
        'graph_summary': graph_info or {}
    }
    
    total_vertices = None
    for metric_name, metric_array in metrics_dict.items():
        if total_vertices is None:
            total_vertices = len(metric_array)
        nan_count = np.sum(np.isnan(metric_array))
        zero_count = np.sum(metric_array == 0)
        finite_count = np.sum(np.isfinite(metric_array))
        
        if nan_count > 0:
            diagnosis['issues'].append(f"{metric_name}: {nan_count}/{total_vertices} NaN values")
            
        if finite_count == 0:
            diagnosis['issues'].append(f"{metric_name}: All values are NaN or infinite")
            
        if finite_count > 0 and np.allclose(metric_array[np.isfinite(metric_array)], 0):
            diagnosis['issues'].append(f"{metric_name}: All finite values are zero (possible normalization issue)")
    
    # Add recommendations based on common issues
    if any('closeness' in issue for issue in diagnosis['issues']):
        diagnosis['recommendations'].append(
            "Closeness centrality often produces NaN in disconnected graphs. "
            "Consider using harmonic closeness or analyzing per connected component."
        )
        
    if any('hits' in issue for issue in diagnosis['issues']):
        diagnosis['recommendations'].append(
            "HITS algorithm requires sufficient linking structure. "
            "Consider using PageRank or degree centrality for sparse neural networks."
        )
        
    if any('eigenvector' in issue for issue in diagnosis['issues']):
        diagnosis['recommendations'].append(
            "Eigenvector centrality may fail to converge on disconnected components. "
            "Consider using Katz centrality as a more robust alternative."
        )
        
    if not diagnosis['issues']:
        diagnosis['recommendations'].append("All centrality metrics appear to be computed successfully.")
        
    return diagnosis
