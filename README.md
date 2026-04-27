# Network Metrics Analysis Package

A comprehensive Python framework for analyzing complex networks using graph theory metrics with built-in visualization and reproducible research capabilities.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trernghwhuare/metrics-analysis-project/main?urlpath=lab)

---

## 📋 Table of Contents
- [📊 Features](#features)
- [🚀 Quick Start](#quick-start)
- [🎨 Network Visualization Capabilities](#network-visualization-capabilities)
- [🔬 Neuronal Network Activity Analysis](#neuronal-network-activity-analysis)
- [🔧 Core Analysis Scripts](#core-analysis-scripts)
- [📚 Documentation & Resources](#documentation-resources)
- [📝 Academic Citation](#academic-citation)
- [🤝 Contributing](#contributing)
- [📄 License](#license)
- [🏷️ Keywords](#keywords)

---

## 📊 Features {#features}

- **Advanced Network Metrics**: Comprehensive suite of graph theory metrics including centrality measures, clustering coefficients, path analysis, and community detection
- **Graph-tool Integration**: High-performance network analysis leveraging the powerful graph-tool library
- **Modular Network Visualization**: Static graph drawing with anatomical organization based on cortical layers, regions, and cell types
- **Hierarchical Community Detection**: Multi-level community structure visualization with nested blockmodel analysis
- **Neuronal Dynamics Modeling**: Compare node state models (SIRS epidemic) and edge rewiring patterns to simulate complex neuronal network activities
- **Reproducible Research**: Complete computational environment with MyBinder compatibility
- **Interactive Analysis**: Jupyter Notebook interface for exploratory data analysis
- **Professional Documentation**: MyST-powered documentation with Jupyter Book integration

## 🚀 Quick Start {#quick-start}

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
```

### Requirements

- **Python**: >= 3.8, < 3.12
- **Node.js**: >= 20 (required for documentation building)
- **Platforms**: Linux (x86_64), macOS (x86_64, arm64)

> **Note**: The `graph-tool` dependency is only available on Linux and macOS platforms through conda-forge.

## 🎨 Network Visualization Capabilities {#network-visualization-capabilities}

The framework provides three specialized scripts for **static graph modular and hierarchical visualization** of neuronal networks:

### **`gt_hierarchy.py` - Hierarchical Community Structure Visualization**
Creates anatomically-organized visualizations with multi-level community detection:
- **Nested Blockmodel Analysis**: Uses degree-corrected and non-degree-corrected hierarchical stochastic blockmodels
- **Biological Organization**: Arranges nodes based on cortical layers (L1-L6), brain regions (M1, M2, S1), and cell types (excitatory/inhibitory)
- **Hierarchical Tree Layouts**: Generates radial tree layouts showing community hierarchy with control points for edge routing
- **Animation Support**: Creates animated GIFs showing community structure evolution during MCMC sampling

### **`gt_graphdraw.py` - Modular Network Drawing**
Generates publication-ready static network drawings with probabilistic connection patterns:
- **Anatomical Positioning**: Uses SFDP layout with biological grouping constraints for realistic spatial organization
- **Edge Type Differentiation**: Visualizes different connection types (continuous projections, electrical projections, input connections) with distinct colors, widths, and dash patterns
- **Modular Structure**: Emphasizes intra-module vs inter-module connections with different probability parameters
- **Multi-view Output**: Creates separate visualizations for main network, input connections, and full integrated network

### **`gt_metrics.py` - Comprehensive Network Metrics Visualization**
Computes and visualizes multiple graph-theoretical centrality metrics with FacetGrid support:
- **Centrality Suite**: Calculates PageRank, betweenness, eigenvector, Katz, HITS, eigentrust, trust transitivity, and closeness
- **Individual Metric Plots**: Generates separate static plots for each centrality measure with color-coded nodes
- **FacetGrid Analysis**: Creates scatter plot matrices showing relationships between degree and various centrality measures
- **Animation Integration**: Produces animated GIFs showing metric evolution during network simulation

All three scripts process NeuroML network files and output high-quality static images (PNG) and animations (GIF) organized in dedicated output directories (`hierarchy/`, `graph_draw/`, `metrics/`).

## 🔬 Neuronal Network Activity Analysis {#neuronal-network-activity-analysis}

The `analysis/` directory provides specialized tools for comparing **node state dynamics** and **edge rewiring patterns** in neuronal network simulations:

### Model Types Compared
- **Node State Models**: SIRS (Susceptible-Infected-Recovered-Susceptible) epidemic dynamics that simulate how activation states propagate through neuronal networks
- **Edge Rewiring Patterns**: Dynamic network restructuring based on geometric or topological criteria, mimicking synaptic plasticity
- **Combined Models**: Simultaneous state propagation and network rewiring to capture complex neuronal activity patterns

### Analysis Workflow
1. **Lightweight Implementation**: Uses `networkx` and `numpy` instead of `graph-tool` for faster execution and broader compatibility
2. **Statistical Comparison**: Runs multiple trials with different random seeds to compute mean and standard deviation bands
3. **Visual Comparison**: Generates publication-ready plots comparing algorithm variants with statistical significance testing (t-tests and ANOVA)

### Key Scripts
- **`algorithms_extracted.py`**: Core implementations of SIRS dynamics, random rewiring, and combined models
- **`compare_algorithms.py`**: Automated comparison pipeline that generates:
  - Pairwise comparisons (basic vs advanced variants)
  - Triple comparisons (state-only vs rewiring-only vs combined models)
  - Statistical significance testing with p-values

**Usage**:
```bash
# Run the complete analysis pipeline
cd analysis/
python compare_algorithms.py

# Output: Plots saved to analysis/plots/ directory
# - pair_dynamic.png, pair_state.png, pair_combined.png
# - triple_nonadv.png, triple_adv.png
# - Artistic variants with mathematical formulas
```

## 🔧 Core Analysis Scripts {#core-analysis-scripts}

Your project includes several specialized scripts for different types of network analysis:

### Data Conversion & Preprocessing
- **`extract_gt_params.py`** - NeuroML Parameter Extraction  
  Extracts comprehensive network parameters from NeuroML (.net.nml) files and saves them as CSV and JSON files in the `gt/params` directory, including neuron types, layers, regions, and synaptic parameters.

  ```bash
  # Process all NeuroML files in net_files directory (default)
  python extract_gt_params.py
  
  # Process a specific file
  python extract_gt_params.py net_files/TC2PT.net.nml --output TC2PT_converted
  ```

### Network Metrics & Calibration
- **`robust_calibrate_centrality.py`** - Robust Centrality Calibration  
  Performs comprehensive centrality metric calibration across multiple network models with structural weighting and normalization support.

  ```bash
  # Calibrate all networks with structural weights
  python robust_calibrate_centrality.py --output-dir robust_calibrated
  
  # Calibrate without weights
  python robust_calibrate_centrality.py --input-dir . --output-dir robust_calibrated --no-weights
  ```

- **`plot_pairplots.py`** - Statistical Visualization of Calibrated Metrics  
  Generates comprehensive statistical visualizations from the calibrated centrality metrics produced by `robust_calibrate_centrality.py`. Creates univariate KDE distributions and scatterplot matrices comparing centralities versus degree for each network model.

  ```bash
  # Visualize all calibrated metrics in robust_calibrated/ directory
  python plot_pairplots.py robust_calibrated/
  
  # Specify custom output directory for plots
  python plot_pairplots.py robust_calibrated/ --output-dir robust_calibrated/plots/
  ```

### Dynamic State Analysis
- **Basic Analysis**: `gt_state.py`, `gt_dynamic.py`  
  Advanced state-based network analysis with real-time visualization
- **Enhanced Analysis**: `gt_state_adv.py`, `gt_dynamic_adv.py`  
  Enhanced versions with additional features and optimizations
- **Combined Analysis**: `gt_state_dynamic.py`, `gt_state_dynamic_adv.py`  
  Combined state-dynamic analysis for complex evolutionary models

## 📚 Documentation & Resources {#documentation-resources}

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
- Simulate and compare neuronal activity dynamics using state-based and structural plasticity models

The protocol documentation provides detailed methodology for neuroscience-specific applications.

## 📝 Academic Citation {#academic-citation}

If you use this package in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

## 🤝 Contributing {#contributing}

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License {#license}

This project is licensed under the MIT License - see the [LICENSE](#license) file for details.

## 🏷️ Keywords {#keywords}

network analysis, graph theory, complex networks, computational neuroscience, graph-tool, reproducible research, Jupyter, MyST, Jupyter Book, neuronal dynamics, SIRS model, edge rewiring, synaptic plasticity, hierarchical visualization, modular networks

---

*This repository is optimized for [Neurolibre](https://neurolibre.com/) publication with complete MyBinder compatibility.*