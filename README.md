# Metrics Analysis Project

A Python package for analyzing graph metrics in neural network structures.

## Overview

The Metrics Analysis Project is designed to generate metric values and aligned arrays for statistical analysis. It provides a modular structure for metric generation, utility functions, and visualization of metric comparisons.

## Features

- Compute multiple graph metrics (PageRank, Betweenness, Closeness, Eigenvector, Katz, HITS, etc.)
- Generate statistical visualizations (violin plots, box plots, heatmaps, clustermaps)
- Save metrics in multiple formats (NPZ, CSV)
- Normalize metrics for easier comparison

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

For computing metrics:
```bash
python src/metrics/main.py --graph path/to/graph.gt --out output_directory --prefix my_network
```

For plotting existing metrics:
```bash
python src/plotting/compare_plots.py
```

### As a Library

```python
import graph_tool.all as gt
from src.metrics.generator import compute_and_save_metrics

# Load your graph
g = gt.load_graph("path/to/your/graph.gt")

# Compute metrics
metrics, npz_path, csv_path = compute_and_save_metrics(
    g, 
    out_dir="output", 
    prefix="my_network",
    normalize=True
)
```

## Project Structure

```
metrics_analysis_project/
├── src/
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── main.py
│   │   └── utils.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   └── compare_plots.py
│   └── main.py
├── tests/
│   ├── test_generator.py
│   └── test_plotting.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.