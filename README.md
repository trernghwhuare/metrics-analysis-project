# Network Metrics Analysis Framework

A comprehensive Python framework for analyzing complex networks using graph theory metrics with built-in visualization and reproducible research capabilities.

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/trernghwhuare/metrics-analysis-project/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/trernghwhuare/metrics-analysis-project/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/trernghwhuare/metrics-analysis-project)
![GitHub Pages](https://img.shields.io/badge/Documentation-GitHub_Pages-success)

## 📊 Features

- **Comprehensive Graph Theory Metrics**: Calculate degree distribution, clustering coefficient, path length, and other network properties
- **Advanced Network Structure Analysis**: Built-in correlation analysis and structural insights generation
- **Professional Visualizations**: Built-in plotting tools for comparing different network structures (violin plots, box plots, heatmaps, clustermaps)
- **Graph Generation**: Create various graph models (scale-free, small-world, random geometric, etc.) and save as .gt files
- **Reproducible Research**: Integrated Jupyter Book support for generating academic-quality documentation
- **Modular & Extensible Architecture**: Easy to add new metrics and analysis methods
- **Comprehensive Testing**: Full test coverage for reliable results
- **Production Ready**: Proper package structure with versioning and dependency management
- **Continuous Integration**: Automated testing and documentation deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/trernghwhuare/metrics-analysis-project.git
cd metrics-analysis-project

# Install dependencies (including graph-tool)
pip install -r requirements.txt

# Install documentation dependencies (optional)
pip install jupyter-book myst-parser
```

### Basic Usage

```python
from network_metrics_package import compute_and_save_metrics, analyze_network_structure

# Compute all available metrics for your network data
metrics, npz_path, csv_path = compute_and_save_metrics(
    your_graph_tool_graph,
    out_dir="results/",
    prefix="my_network"
)

# Analyze network structure and get insights
structure_analysis = analyze_network_structure(metrics)
```

### Graph Generation

Generate synthetic networks for analysis or testing:

```python
from network_metrics_package.gt_generator import generate_graph

# Generate a scale-free Price network
g = generate_graph('price', n_vertices=1000, c=0.8, m=2, directed=True)
g.save("scale_free_network.gt")

# Generate a small-world network
g = generate_graph('small_world', n_vertices=500, k=4, p=0.1)
g.save("small_world_network.gt")
```

### Comprehensive Analysis Script

Use the built-in analysis script for complete end-to-end analysis:

```bash
# Complete analysis with all visualizations
python analyze_networks_metrics.py --graph your_network.gt --output-dir results/

# Custom analysis with specific options
python analyze_networks_metrics.py --graph your_network.gt --output-dir results/ --no-normalize --plots violin box --threads 16
```

### Graph Generation Script

Use the command-line interface to generate various graph models:

```bash
# Generate a Price network (scale-free)
python -m network_metrics_package.gt_generator --model price --vertices 1000 --output price_1000.gt

# Generate a random graph
python -m network_metrics_package.gt_generator --model random --vertices 500 --edges 2000 --output random_500_2000.gt

# Generate a small-world network
python -m network_metrics_package.gt_generator --model small_world --vertices 100 --k 4 --p 0.1 --output sw_100_4_0.1.gt

# Generate example neuroscience networks
python generate_example_networks.py
```

### Interactive Notebook

Explore the framework interactively using the provided Jupyter notebook:

```bash
jupyter notebook Network_Metrics_Analysis.ipynb
```


## 📚 Documentation & Reproducible Research

This project includes integrated documentation generation using Jupyter Book:

```bash
# Build HTML documentation
jupyter-book build .

# Build PDF documentation  
jupyter-book build . --builder pdflatex
```

The generated documentation provides:
- Complete API reference
- Usage examples and tutorials
- Theoretical background on network metrics
- Reproducible analysis workflows
- Best practices for network analysis

## 🧪 Testing

Run the comprehensive test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run with verbose output
python -m unittest discover tests/ -v
```

## 📈 Example Output

The framework generates professional-quality outputs including:
- **Metric Files**: NPZ and CSV files with all computed metrics
- **Visualizations**: 
  - Degree distribution violin plots
  - Box plots for metric comparisons
  - Correlation heatmaps
  - Hierarchical clustermaps
- **Structural Analysis**: JSON-compatible analysis results with correlations and insights
- **Graph Files**: .gt files for generated networks that can be loaded and analyzed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- New network metrics
- Additional visualization types
- Performance improvements
- Documentation enhancements
- Bug fixes and feature requests
- New graph generation models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with** ❤️ **for network science researchers, data scientists, and computational biologists**

*This framework enables systematic extraction of complex network structural features and supports complete reproducible research workflows from data to publication-ready results.*