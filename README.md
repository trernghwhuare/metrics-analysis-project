# Network Metrics Analysis Framework

A comprehensive Python framework for analyzing complex networks using graph theory metrics with built-in visualization and reproducible research capabilities.

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/trernghwhuare/metrics-analysis-project/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/trernghwhuare/metrics-analysis-project/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/trernghwhuare/metrics-analysis-project)
![GitHub Pages](https://img.shields.io/badge/Documentation-GitHub_Pages-success)

## 📊 Features

- **Comprehensive Graph Theory Metrics**: Calculate degree distribution, clustering coefficient, path length, and other network properties
- **Network Structure Analysis**: Advanced analysis of computed metrics for deeper insights
- **Interactive Visualizations**: Built-in plotting tools for comparing different network structures
- **Reproducible Research**: Integrated Jupyter Book support for generating academic-quality documentation
- **Modular Architecture**: Extensible design for adding new metrics and analysis methods
- **Unit Tested**: Comprehensive test coverage for reliable results
- **Continuous Integration**: Automated testing and documentation deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/trernghwhuare/metrics-analysis-project.git
cd metrics-analysis-project

# Install dependencies
pip install -r requirements.txt

# Install documentation dependencies (optional)
pip install jupyter-book myst-parser
```

### Basic Usage

```python
from src.metrics.generator import NetworkMetricsGenerator
from src.metrics.utils import analyze_network_structure

# Initialize the metrics generator
generator = NetworkMetricsGenerator()

# Compute all available metrics for your network data
metrics = generator.compute_all_metrics(your_network_data)

# Analyze network structure
structure_analysis = analyze_network_structure(metrics)

# Generate visualizations
from src.plotting.compare_plots import create_comparison_plots
plots = create_comparison_plots(metrics)
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

## 🧪 Testing

Run the comprehensive test suite to ensure everything works correctly:

```bash
pytest tests/
```

## 📈 Example Output

The framework generates professional-quality visualizations including:
- Degree distribution plots
- Clustering coefficient comparisons
- Path length distributions
- Network structure comparison charts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- New network metrics
- Additional visualization types
- Performance improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with** ❤️ **for network science researchers, data scientists, and computational biologists**

*This framework enables systematic extraction of complex network structural features and supports complete reproducible research workflows from data to publication-ready results.*