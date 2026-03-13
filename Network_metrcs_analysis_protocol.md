# Using the Network Metrics Package with Your Own Networks

This document provides a step-by-step guide on how to use the network_metrics_package with your own network data.

## 1. Preparing Your Network Data

The package works with networks in several formats. If your network is not already in graph-tool format (.gt), you'll need to convert it first.

### Supported Formats
- GraphML (.graphml)
- GML (.gml)
- Edge lists (.edges, .txt)
- Adjacency matrices (.csv)
- Graph-tool native format (.gt) - Most efficient

### Converting Your Network

If your network is in a different format, use the provided conversion script:

```bash
# For edge list format
python load_network.py your_network.edgelist edgelist

# For GraphML format
python load_network.py your_network.graphml graphml

# For GML format
python load_network.py your_network.gml gml

# For adjacency matrix
python load_network.py your_network.csv adjacency
```

This will create a file named `your_network.gt` which can be used with the analysis tools.

## 2. Analyzing Your Network

Once you have your network in .gt format, you can analyze it in several ways:

### Command Line Approach

```bash
python src/network_metrics_package/main.py \
  --graph your_network.gt \
  --out results_directory \
  --prefix your_analysis_name
```

Options:
- `--no-normalize`: Disable min-max normalization of metrics
- `--threads N`: Set the number of threads for computation (default: 8)

### Python Script Approach

For more control, use the Python API:

```python
import graph_tool.all as gt
from network_metrics_package.metrics.generator import compute_and_save_metrics

# Load your network
g = gt.load_graph("your_network.gt")

# Compute metrics
metrics, npz_path, csv_path = compute_and_save_metrics(
    g,
    out_dir="results_directory",
    prefix="your_analysis_name",
    normalize=True,  # Enable min-max normalization
    nthreads=8,      # Number of threads
    save_files=True  # Save to .npz and .csv files
)

# Metrics is a dictionary with keys like:
# 'pagerank', 'betweenness', 'closeness', 'eigenvector', 'katz', 'hits_hub', 'hits_authority'
```

## 3. Understanding the Output

The analysis produces .npz and .csv files containing the computed metrics.

### Metrics Included
1. **PageRank**: Measures node importance based on link analysis
2. **Betweenness**: Measures how often a node lies on shortest paths
3. **Closeness**: Measures how close a node is to all other nodes
4. **Eigenvector**: Measures node importance based on neighbor importance
5. **Katz**: Measures node influence considering path distances
6. **HITS Hub**: Measures how well a node points to authoritative nodes
7. **HITS Authority**: Measures how well a node is pointed to by hubs

## 4. Working with the Results

### Loading Metrics in Python

```python
import numpy as np

# Load from .npz file
data = np.load('your_analysis_name_metrics.npz')
pagerank = data['pagerank']
betweenness = data['betweenness']
# ... and so on for other metrics

# Or load from CSV
import pandas as pd
df = pd.read_csv('your_analysis_name_metrics.csv')
```

### Further Analysis

The metrics can be used for:
- Identifying important nodes in your network
- Comparing different networks
- Correlating network structure with function
- Clustering nodes based on their metrics
- Building machine learning models with network features

## 5. Best Practices

1. **Normalization**: Enable normalization (`normalize=True`) to make metrics comparable across networks of different sizes.

2. **Thread Count**: Adjust the number of threads based on your CPU. More threads can speed up computation but may impact other processes.

3. **Large Networks**: For very large networks (>100,000 nodes), consider:
   - Using fewer metrics
   - Increasing available memory
   - Running on a machine with more CPU cores

4. **Memory Usage**: The package can be memory-intensive for large networks. Ensure you have sufficient RAM.

5. **Reproducibility**: Save your networks in .gt format for reproducible results.

## 6. Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the package with:
   ```bash
   pip install -e . --no-deps
   ```

2. **Missing graph-tool**: Install graph-tool with:
   ```bash
   sudo apt-get install python3-graph-tool
   ```

3. **Memory Errors**: For large networks, try:
   - Reducing the number of threads
   - Using a machine with more RAM
   - Analyzing network components separately

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your network file is in the correct format
3. Ensure you have write permissions to the output directory
4. Check the terminal output for specific error messages

## 7. Extending the Package

You can extend the package by:
1. Adding new metrics in `src/network_metrics_package/metrics/`
2. Adding new analysis methods in `src/network_metrics_package/analysis/`

For contributions, please follow the existing code style and add appropriate tests in the `tests/` directory.