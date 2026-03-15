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
                'count': int(len(series))
            }
        else:
            summary_stats[column] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
    
    # Generate basic insights
    insights = []
    if len(correlations) > 1:
        # Find highly correlated metrics
        corr_df = pd.DataFrame(correlations)
        high_corr_pairs = []
        for i, col1 in enumerate(corr_df.columns):
            for col2 in corr_df.columns[i+1:]:
                corr_val = corr_df.loc[col1, col2]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            insights.append(f"Found {len(high_corr_pairs)} pairs of highly correlated metrics (|r| > 0.7)")
    
    # Check for metrics with low variance
    low_variance_metrics = []
    for metric, stats in summary_stats.items():
        if stats['std'] < 0.1 and stats['count'] > 0:
            low_variance_metrics.append(metric)
    
    if low_variance_metrics:
        insights.append(f"Metrics with low variance (< 0.1): {', '.join(low_variance_metrics)}")
    
    if not insights:
        insights.append("No specific structural patterns detected")
    
    return {
        'correlations': correlations,
        'summary_stats': summary_stats,
        'insights': insights
    }