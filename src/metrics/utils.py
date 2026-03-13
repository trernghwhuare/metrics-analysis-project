import numpy as np
import logging

logger = logging.getLogger(__name__)

def sanitize_array(arr):
    """Convert input to numpy float array and replace +/-inf with nan.
    Accept graph-tool vertex property objects (has .get_array or .a)."""
    try:
        # Handle None case
        if arr is None:
            return np.array([])
            
        # Handle scalar values
        if np.isscalar(arr):
            return np.array([float(arr)])
            
        if hasattr(arr, "get_array"):
            a = np.asarray(arr.get_array(), dtype=float)
        elif hasattr(arr, "a"):
            a = np.asarray(arr.a, dtype=float)
        else:
            a = np.asarray(arr, dtype=float)
    except Exception:
        # fallback to best-effort conversion
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            # If all else fails, return empty array
            a = np.array([])
            
    # Handle case where a might not be an array
    if not hasattr(a, '__len__'):
        a = np.array([float(a)]) if not np.isscalar(a) else np.array([float(a)])
        
    a[~np.isfinite(a)] = np.nan
    return a

def minmax_normalize(arr):
    """Min-max normalize 1D array, preserving nan values."""
    a = sanitize_array(arr)
    # Check if array is empty
    if len(a) == 0:
        return a
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