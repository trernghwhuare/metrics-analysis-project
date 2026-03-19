# Network Metrics Analysis Package

A comprehensive Python framework for analyzing complex networks using graph theory metrics with built-in visualization and reproducible research capabilities.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trernghwhuare/metrics-analysis-project/main?urlpath=lab)

## 📊 Features

- **Advanced Network Metrics**: Comprehensive suite of graph theory metrics including centrality measures, clustering coefficients, path analysis, and community detection
- **Graph-tool Integration**: High-performance network analysis leveraging the powerful graph-tool library
- **Reproducible Research**: Complete computational environment with MyBinder compatibility
- **Interactive Analysis**: Jupyter Notebook interface for exploratory data analysis
- **Professional Documentation**: MyST-powered documentation with Jupyter Book integration

## 🚀 Quick Start

### Local Development with Pixi (Recommended)

This project uses [pixi](https://pixi.sh/) for dependency management, providing a reproducible environment across platforms.

```bash
# Clone the repository
git clone https://github.com/trernghwhuare/metrics-analysis-project.git
cd metrics-analysis-project

# Install the pixi environment (requires Node.js >= 20 for documentation)
pixi install

# Launch Jupyter Notebook
pixi run notebook

# Build documentation (view at http://localhost:3000)
pixi run build-docs

# Run analysis script
pixi run analyze

# Run tests
pixi run test
```

### Requirements

- **Python**: >= 3.8, < 3.12
- **Node.js**: >= 20 (required for documentation building)
- **Platforms**: Linux (x86_64), macOS (x86_64, arm64)

> **Note**: The `graph-tool` dependency is only available on Linux and macOS platforms through conda-forge.

## 📚 Documentation

The project includes comprehensive documentation built with MyST and Jupyter Book:

- **Main Analysis Notebook**: `Network_Metrics_Analysis.ipynb`
- **Protocol Documentation**: `protocol_document.md` 
- **API Reference**: Generated from source code

To build and view documentation locally:
```bash
pixi run build-docs
# Then open http://localhost:3000 in your browser
```

## 🔬 Neuroscience Context

This framework is designed for computational neuroscience applications, enabling researchers to:
- Analyze brain connectivity networks from neuroimaging data
- Compute graph-theoretical metrics for neural circuits
- Compare network properties across experimental conditions
- Generate publication-ready visualizations of complex networks

The protocol documentation provides detailed methodology for neuroscience-specific applications.

## 🧪 Testing

The package includes comprehensive unit tests to ensure reliability:

```bash
pixi run test
```

## 📝 Academic Citation

If you use this package in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Keywords

network analysis, graph theory, complex networks, computational neuroscience, graph-tool, reproducible research, Jupyter, MyST, Jupyter Book

---

*This repository is optimized for [Neurolibre](https://neurolibre.com/) publication with complete MyBinder compatibility.*