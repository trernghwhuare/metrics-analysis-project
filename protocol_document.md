# Analyzing Neural Network Files

This document provides specific instructions for analyzing neural network files with the network_metrics_package, with detailed neuroscience context and practical guidance for computational neuroscience applications.

## 🧠 Neuroscience Context and Applications

This framework is specifically designed for analyzing **brain connectivity networks** and **neural circuit data** in computational neuroscience research. The included `.gt` files represent biologically-inspired network models that capture key organizational principles of real neural systems:

### Network Model Categories
**Cortical or Thalamic Circuits:**
- **iT/iC_max_plus.net.nml**: Intra-Cortico or intra-thalamic loops
- **Loop models (L1-L6)**: intra cortical layers loops
-**C2T_max_plus.net.nml**: Cortico-thalamic loops
- **T2C_max_plus.net.nml**: Thalamo-cortical loops

**specific thalamo-cortical circuits loops:**
- **TC2CT.net.nml**: Thalamocortical → cortex pathways
- **TC2PT.net.nml**: Thalamocortical → pyramidal tract neuron pathways  
- **TC2IT2PTCT.net.nml**: Thalamocortical → intratelencephalic (IT) → pyramidal tract (PT) neurons → cortex pathways
- **TC2IT4_IT2CT.net.nml**: Thalamocortical → Layer 4 intratelencephalic → intratelencephalic (IT) → cortex pathways

**Cortico-thalamo-cortical (CTC) Microcircuit loops Models (Primary motion area, Secondary motion are, Primary sensory are):**
- **max_CTC_plus.net.nml**: Optimized cortico-thalamo-cortical loop architectures
-**M2a/M2a/M1a/M1b/S1a/S1b_max_plus.net.nml**: CTC microcircuit models in unilateral hemisphere for primary, secondary motion area or primary sensory area


**Multi-regional CTC Circuit Models:**
- **M1/M2_max_plus.net.nml**: CTC microcircuit in Motor area of  bilateral hemispheres
- **S1/S2_max_plus.net.nml**: CTC microcircuit in somatosensory cortex hierarchical organization of bilateral hemispheres
- **M2aM1aS1a/S1bM1bM2b_max_plus.net.nml**: CTC microcircuit across M2,M1,S1 areas of unilateral hemisphere
- **M2M1S1_max_plus.net.nml**: CTC microcircuit across M2,M1,S1 areas of unilateral hemisphere of bilateral hemispheres

These models are derived from empirical neuroanatomical data and theoretical principles of cortical organization, making them ideal for testing graph-theoretical hypotheses about brain network function.

## 📊 Graph Theory Metrics in Neuroscience

Each computed centrality metric provides unique insights into neural circuit organization and information processing:

### 1. **PageRank Centrality**

$$
PR(v) = \frac{1-d}{N} + d \sum_{u \in \Gamma^{-}(v)} \frac{PR(u)}{d^{+}(u)}
$$

### 2. **Betweenness Centrality** 

$$
C_B(v) = \sum_{\substack{s \neq v \neq t \in V \\ s \neq t}} \frac{\sigma_{st}(v)}{\sigma_{st}}
$$

### 3. **Closeness Centrality**

$$
c_i = \frac{1}{\sum_j d_{ij}}
$$

### 4. **Eigenvector Centrality**

$$
\mathbf{A}\mathbf{x} = \lambda\mathbf{x}
$$

### 5. **Katz Centrality**

$$
\mathbf{x} = \alpha\mathbf{A}\mathbf{x} + \boldsymbol{\beta}
$$

### 6. **HITS Hub and Authority Centrality**

$$
\begin{aligned}
\mathbf{x} &= \alpha\mathbf{A}\mathbf{y} \\
\mathbf{y} &= \beta\mathbf{A}^T\mathbf{x}
\end{aligned}
$$

### 7. **EigenTrust Centrality**

$$
\mathbf{t} = \lim_{n\to\infty} \left(C^T\right)^n \mathbf{c}
$$

### 8. **Trust Transitivity Centrality**

$$
t_{ij} = \frac{\sum_m A_{m,j} w^2_{G\setminus\{j\}}(i\to m)c_{m,j}}{\sum_m A_{m,j} w_{G\setminus\{j\}}(i\to m)}
$$

## 🎨 Network Visualization Capabilities

The framework provides three specialized scripts for **static graph modular and hierarchical visualization** of neuronal networks, each serving distinct analytical purposes:

### **`gt_hierarchy.py` - Hierarchical Community Structure Visualization**
Creates anatomically-organized visualizations with multi-level community detection:
- **Nested Blockmodel Analysis**: Uses degree-corrected and non-degree-corrected hierarchical stochastic blockmodels to reveal community structure
- **Biological Organization**: Arranges nodes based on cortical layers (L1-L6), brain regions (M1, M2, S1), and cell types (excitatory/inhibitory)
- **Hierarchical Tree Layouts**: Generates radial tree layouts showing community hierarchy with control points for edge routing
- **Animation Support**: Creates animated GIFs showing community structure evolution during MCMC sampling
- **Output Directory**: `hierarchy/{network_name}/`

### **`gt_graphdraw.py` - Modular Network Drawing**
Generates publication-ready static network drawings with probabilistic connection patterns:
- **Anatomical Positioning**: Uses SFDP layout with biological grouping constraints for realistic spatial organization
- **Edge Type Differentiation**: Visualizes different connection types (continuous projections, electrical projections, input connections) with distinct colors, widths, and dash patterns
- **Modular Structure**: Emphasizes intra-module vs inter-module connections with different probability parameters
- **Multi-view Output**: Creates separate visualizations for main network, input connections, and full integrated network
- **Output Directory**: `graph_draw/{network_name}/`

### **`gt_metrics.py` - Comprehensive Network Metrics Visualization**
Computes and visualizes multiple graph-theoretical centrality metrics with FacetGrid support:
- **Centrality Suite**: Calculates PageRank, betweenness, eigenvector, Katz, HITS, eigentrust, trust transitivity, and closeness
- **Individual Metric Plots**: Generates separate static plots for each centrality measure with color-coded nodes
- **FacetGrid Analysis**: Creates scatter plot matrices showing relationships between degree and various centrality measures
- **Animation Integration**: Produces animated GIFs showing metric evolution during network simulation
- **Output Directory**: `metrics/{network_name}/`

All three scripts process NeuroML network files and output high-quality static images (PNG) and animations (GIF) organized in dedicated output directories.

## 📁 Available Network Files

The following network files are available for analysis in this project's root directory:

**Core Thalamocortical Models:**
- `TC2CT.net.nml`, `TC2PT.net.nml`, `TC2IT2PTCT.net.nml`, `TC2IT4_IT2CT.net.nml`

**Cortical Circuit Models:**
- `M1_max_plus.net.nml`, `M2_max_plus.net.nml`, `S1_max_plus.net.nml`, `C2T_max_plus.net.nml`, `T2C_max_plus.net.nml`

**Integrated Multi-area Models:**
- `M2M1S1_max_plus.net.nml`, `M2aM1aS1a_max_plus.net.nml`, `S1bM1bM2b_max_plus.net.nml`

**Specialized Circuit Motifs:**
- `loop_L1.net.nml` through `loop_L6.net.nml` (canonical microcircuit loops)
- `iT_max_plus.net.nml`, `iC_max.net.nml` (inhibitory-dominated circuits)
- `max_CTC_plus.net.nml` (optimized cortico-thalamo-cortical loops)

## 🚀 How to Analyze Networks

### 1. Network Visualization Scripts

Based on your CLI preferences, you can use the specialized visualization scripts directly with position parameters:

```bash
# Hierarchical community structure visualization
python gt_hierarchy.py

# Modular network drawing with anatomical organization
python gt_graphdraw.py

# Comprehensive network metrics visualization
python gt_metrics.py
```

These scripts automatically process all available NeuroML files in the `net_files/` directory and generate comprehensive visualizations in their respective output directories.

### 2. Specialized Analysis and Data Processing Scripts

Your project includes several specialized scripts for different types of network analysis and data processing:

#### **extract_gt_params.py** - NeuroML to Graph-Tool Conversion
**Purpose**: Extracts network parameters from NeuroML (.net.nml) files and converts them to graph-tool (.csv, .json) format with comprehensive metadata.

**Usage**:
```bash
# Process a single NeuroML file
python extract_gt_params.py net_files/TC2PT.net.nml --output TC2PT_converted

# Process all NeuroML files in net_files directory
python extract_gt_params.py

# Specify custom input directory
python extract_gt_params.py --input-dir /path/to/custom/net_files
```

**Output**: Generates `.csv` and `.json` files with embedded vertex and edge properties including:
- Neuron type classifications (excitatory/inhibitory)
- Layer and region assignments  
- Synaptic connection parameters
- Network hierarchy metadata

This script serves as the primary data ingestion pipeline, converting biologically-detailed NeuroML network descriptions into analyzable graph-tool networks.

#### **robust_calibrate_centrality.py** - Robust Centrality Calibration
**Purpose**: Performs comprehensive centrality metric calibration across multiple network models with structural weighting and normalization support.

**Usage**:
```bash
# Calibrate all networks with structural weights using default pattern
python robust_calibrate_centrality.py --output-dir robust_calibrated

# Calibrate using a specific glob pattern
python robust_calibrate_centrality.py --input-dir "metrics/*/*.gt" --output-dir robust_calibrated

# Calibrate from a directory
python robust_calibrate_centrality.py --input-dir ./networks --output-dir robust_calibrated

# Calibrate without weights
python robust_calibrate_centrality.py --input-dir . --output-dir robust_calibrated --no-weights
```

**Features**:
- **Structural Weighting**: Automatically creates meaningful edge weights based on vertex degree products to enhance centrality calculations
- **Comprehensive Metrics**: Computes all major centrality measures (degree, betweenness, closeness, eigenvector, PageRank, Katz, HITS, EigenTrust)
- **Robust Error Handling**: Continues processing even if individual networks fail, providing detailed success/failure statistics
- **Flexible Input**: Supports both directory paths and glob patterns for input specification
- **Normalization Options**: Applies min-max normalization by default for consistent cross-network comparisons

This script is ideal for large-scale comparative analysis across multiple neuronal circuit models, ensuring consistent and biologically-meaningful centrality measurements.

### 3. Understanding the Output Structure

#### Metrics Files (stored in `robust_calibrated/` directory):
- `<network_name>_robust_calibrated_metrics.npz`: All metrics in NumPy format for programmatic access
- `<network_name>_robust_calibrated_metrics.csv`: All metrics in CSV format for easy viewing and statistical analysis

#### Visualization Outputs:
The analysis generates four key visualization types stored in appropriate directories:
- **Distribution Analysis**: Box plots and violin plots showing metric distributions
- **Correlation Analysis**: Heatmaps revealing relationships between different centrality measures  
- **Clustering Analysis**: Clustermaps showing grouped patterns in network properties
- **Network Topology**: Direct graph visualizations showing node/edge properties

*Note: Specific plot locations may vary depending on which analysis script is used. Check the `analysis/plots/` directory for comparative visualizations.*

### 4. Loading and Working with Results

```python
# Load from .npz file (recommended for programmatic analysis)
import numpy as np
data = np.load('robust_calibrated/TC2PT_robust_calibrated_metrics.npz')
pagerank = data['pagerank']
betweenness = data['betweenness']

# Or load from CSV (better for exploratory analysis)
import pandas as pd
df = pd.read_csv('robust_calibrated/TC2PT_robust_calibrated_metrics.csv')
print(f"Network has {len(df)} nodes")
print(f"PageRank range: {df['pagerank'].min():.4f} to {df['pagerank'].max():.4f}")
```

### 5. Neuroscience-Specific Analysis Guidelines

#### Interpreting Metric Distributions:
- **Bimodal distributions** often indicate distinct neuronal populations (e.g., excitatory vs inhibitory)
- **Heavy-tailed distributions** suggest hub-like organization common in biological networks
- **Uniform distributions** may indicate regular lattice-like connectivity

#### Comparing Across Network Models:
1. **Within-model comparison**: Compare different centrality metrics within the same network to identify multifunctional neurons
2. **Between-model comparison**: Compare the same metric across different network architectures to understand circuit-specific organization principles
3. **Hierarchical partitions**: Distribution over hierarchies by performing model averaging using the nested SBM.

#### Biological Validation Considerations:
- **Degree correlation**: Check if high-centrality nodes correspond to known anatomical hub regions
- **Functional relevance**: Correlate centrality measures with electrophysiological properties when available
- **Robustness analysis**: Test metric stability under edge weight perturbations to assess biological plausibility

### 6. Advanced Analysis Workflows

#### Dynamic Simulation:

- `gt_state.py` / `gt_state_adv.py`: SIRS model of neuronal activity
- `gt_dynamic.py` / `gt_dynamic_adv.py`: rewiring dyanmic layout
- `gt_state_dynamic.py` / `gt_state_dynamic_adv.py`: Combined SIRS state and rewiring dynamic

#### Weighted Network Analysis:
The framework supports edge-weighted networks for more biologically realistic modeling. When using weighted networks:
- Ensure edge weights reflect biological constraints (e.g., synaptic strength ranges)
- Use appropriate weighted versions of centrality algorithms
- Validate that weighting enhances rather than obscures biological signal

#### Comparative Analysis:
Comparison SIRS model and rewiring dynamic algorithms between edges catagories in pairs: pop_edges and input_edges *vs.* EE,EI,IE,II,input_EE,input_II
Comparison pop_edges/input_edges and EE/EI/IE/II/input_EE/input_II across SIRS model, rewiring dynamic and both combined algorithms.
```bash
# Generate comparative plots (outputs to analysis/plots/)
python analysis/compare_algorithms.py
```
## 🔬 Practical Neuroscience Applications

### Hypothesis Testing Examples:
1. **Hub Identification**: "Do layer 5 pyramidal neurons exhibit higher betweenness centrality than layer 2/3 neurons?"
2. **Circuit Comparison**: "Does the TC2IT2PTCT model show greater PageRank variation than the simpler TC2PT model?"
3. **Perturbation Analysis**: "How do centrality distributions change when inhibitory connections are selectively removed?"

### Publication-Ready Analysis:
The generated visualizations are suitable for direct inclusion in scientific publications, with proper statistical annotations and biological interpretations as demonstrated in the main paper (`paper.md`).

### Reproducible Research:
All analyses are fully reproducible through the pixi environment management system, ensuring consistent results across different computing platforms and over time.