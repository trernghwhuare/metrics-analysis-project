#!/usr/bin/env python3

import numpy as np

# Load the metrics data
data = np.load('../metrics_out/max_CTC_plus_metrics.npz')

print('Available metrics:', list(data.keys()))
print()

# Examine each metric
for key in data.files:
    arr = data[key]
    print(f'Metric: {key}')
    print(f'  Shape: {arr.shape}')
    print(f'  Data type: {arr.dtype}')
    print(f'  NaN count: {np.count_nonzero(np.isnan(arr))}')
    print(f'  Non-NaN count: {np.count_nonzero(~np.isnan(arr))}')
    
    # Show some sample values
    non_nan_values = arr[~np.isnan(arr)]
    if len(non_nan_values) > 0:
        print(f'  Min value: {np.min(non_nan_values)}')
        print(f'  Max value: {np.max(non_nan_values)}')
        print(f'  Sample values: {non_nan_values[:min(5, len(non_nan_values))]}')
    else:
        print('  No valid (non-NaN) values found')
    print()

data.close()