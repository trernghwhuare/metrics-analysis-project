# Analyzing Your Neural Network Files

This document provides specific instructions for analyzing your neural network files with the network_metrics_package.

## Your Network Files

You have the following network files that you want to analyze:
- iT_max_plus.gt
- max_CTC_plus.gt
- max_M2M1S1_plus.gt
- optimus_CTC_plus.gt
- optimus_M2M1S1_plus.gt

## How to Analyze Your Networks

### 1. Place Your Network Files

First, make sure all your network files (.gt files) are in the same directory as this README.

### 2. Check Your Networks

To verify that your network files can be loaded correctly, run:

```bash
python check_networks.py
```

This will show basic information about each network (number of vertices and edges).

### 3. Analyze a Single Network

To analyze a single network, you can use:

```bash
python src/network_metrics_package/main.py --graph <network_file.gt> --out <output_directory> --prefix <network_name>
```

For example:
```bash
python src/network_metrics_package/main.py --graph max_CTC_plus.gt --out max_CTC_analysis --prefix max_CTC
```

### 4. Understanding the Output

For each network, you'll get:

1. **Metrics Files**:
   - `<network_name>_metrics.npz`: All metrics in NumPy format for programmatic access
   - `<network_name>_metrics.csv`: All metrics in CSV format for easy viewing

2. **Visualizations**:
   - `<network_name>_violin.png`: Distribution of each metric
   - `<network_name>_box.png`: Quartiles and outliers of each metric
   - `<network_name>_heatmap_corr.png`: Correlations between metrics
   - `<network_name>_clustermap.png`: Grouped patterns in the metrics

### 5. Metrics Computed

The package computes these graph centrality metrics for each network:
1. **PageRank**: Measures node importance based on link analysis
2. **Betweenness**: Measures how often a node lies on shortest paths
3. **Closeness**: Measures how close a node is to all other nodes
4. **Eigenvector**: Measures node importance based on neighbor importance
5. **Katz**: Measures node influence considering path distances
6. **HITS Hub**: Measures how well a node points to authoritative nodes
7. **HITS Authority**: Measures how well a node is pointed to by hubs

### 6. Working with Results

After analysis, you can load and work with the results:

```python
# Load from .npz file
import numpy as np
data = np.load('max_CTC_analysis/max_CTC_metrics.npz')
pagerank = data['pagerank']

# Or load from CSV
import pandas as pd
df = pd.read_csv('max_CTC_analysis/max_CTC_metrics.csv')
```

### 7. Comparing Networks

Since all networks are analyzed with the same metrics, you can easily compare them:
- Compare the distributions of metrics using the violin or box plots
- Examine the correlation patterns in the heatmaps
- Look for clustering patterns in the clustermaps

## Troubleshooting

If you encounter issues:

1. **File not found errors**: Make sure all .gt files are in the current directory
2. **Import errors**: Make sure you've installed the package with `pip install -e . --no-deps`
3. **Memory errors**: For very large networks, consider running on a machine with more RAM
4. **graph-tool errors**: Make sure graph-tool is properly installed with `sudo apt-get install python3-graph-tool`

## Getting Help

If you need help:
1. Run `python check_networks.py` to verify your files can be loaded
2. Check the terminal output for specific error messages
3. Make sure all dependencies are properly installed