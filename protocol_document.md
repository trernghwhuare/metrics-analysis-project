# Analyzing Neural Network Files

This document provides specific instructions for analyzing neural network files with the network_metrics_package.

## 🧠 Neuroscience Context

This framework is designed for analyzing **brain connectivity networks** and **neural circuit data**. The included `.gt` files represent:
- **Structural brain networks** derived from neuroimaging or electrophysiology data
- **Functional connectivity patterns** between brain regions
- **Synthetic neural circuits** for methodological validation

The computed graph theory metrics (PageRank, Betweenness, Closeness, etc.) are standard measures used in **connectomics** and **computational neuroscience** to understand brain network organization, information flow, and functional specialization.

## Network Files

The following network files available for analyzeing in this project:
- iT_max_plus.gt
- max_CTC_plus.gt
- max_M2M1S1_plus.gt
- optimus_CTC_plus.gt
- optimus_M2M1S1_plus.gt

## How to Analyze Networks

### 1. Analyze a Single Network

To analyze a single network:

```bash
python src/network_metrics_package/main.py --graph <network_file.gt> --out <output_directory> --prefix <network_name>
```

For example:
```bash
python src/network_metrics_package/main.py --graph max_CTC_plus.gt --out max_CTC_analysis --prefix max_CTC
```

### 2. Understanding the Output

1. **Metrics Files**:
For each network analised, the package generates two files
   - `<network_name>_metrics.npz`: All metrics in NumPy format for programmatic access
   - `<network_name>_metrics.csv`: All metrics in CSV format for easy viewing

2. **Visualizations**:
The package generates four types of visualizations plots:
   - `<network_name>_violin.png`: Distribution of each metric
   - `<network_name>_box.png`: Quartiles and outliers of each metric
   - `<network_name>_heatmap_corr.png`: Correlations between metrics
   - `<network_name>_clustermap.png`: Grouped patterns in the metrics

### 3. Metrics Computed

The package computes these graph centrality metrics for each network:
1. **PageRank**: Measures node importance based on link analysis
2. **Betweenness**: Measures how often a node lies on shortest paths
3. **Closeness**: Measures how close a node is to all other nodes
4. **Eigenvector**: Measures node importance based on neighbor importance
5. **Katz**: Measures node influence considering path distances
6. **HITS Hub**: Measures how well a node points to authoritative nodes
7. **HITS Authority**: Measures how well a node is pointed to by hubs

### 4. Load and work with the results

```python
# Load from .npz file
import numpy as np
data = np.load('max_CTC_analysis/max_CTC_metrics.npz')
pagerank = data['pagerank']

# Or load from CSV
import pandas as pd
df = pd.read_csv('max_CTC_analysis/max_CTC_metrics.csv')
```

### 5. Comparing Networks based on the metrics analized

- Compare the distributions of metrics using the violin or box plots
- Examine the correlation patterns in the heatmaps
- Look for clustering patterns in the clustermaps

