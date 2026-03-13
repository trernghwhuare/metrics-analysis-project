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