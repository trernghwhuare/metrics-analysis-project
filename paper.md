# A Reproducible Computational Workflow for Visualizing Neuronal Circuit Models Using Complex Network Analysis

## Abstract

**Background**: Visualization of neuronal circuit models remains a critical challenge in computational neuroscience, requiring specialized tools for representing complex network structures and dynamic properties.
**Objective**: We present a comprehensive, reproducible workflow integrating modern dependency management (pixi), interactive documentation (MyST/Jupyter Book), and advanced visualization capabilities for neuronal circuit models using graph-tool and Python scientific stack.
**Methods**: The workflow combines graph-theoretical analysis with publication-quality visualization tools (matplotlib, seaborn, graph-tool.draw) to create static images and animated GIFs of neuronal connectivity patterns, community structure, and network metrics.
**Results**: Successful implementation demonstrates complete reproducibility across platforms (Linux, macOS) with zero-installation access via MyBinder, validated with graph-tool version 2.98. The framework generates strip plots, box plots, correlation heatmaps, and clustermaps for comparative network analysis, with capability to produce animated visualizations of dynamic circuit properties.
**Significance**: This protocol provides a template for reproducible visualization of neuronal circuit models meeting open science standards and Neurolibre publication requirements.

## Introduction

Computational neuroscience increasingly relies on complex network analysis to understand brain connectivity and neural circuit organization [1]. The ability to visualize neuronal circuit models is essential for interpreting network properties such as centrality, clustering, and community structure in neural systems [2]. However, effective visualization of neuronal circuits presents unique challenges:

- **High-dimensional connectivity**: Neuronal networks often contain thousands to millions of nodes with complex connection patterns
- **Multi-scale organization**: Circuits exhibit hierarchical structure from microcircuits to whole-brain networks
- **Dynamic properties**: Functional connectivity changes over time and across behavioral states
- **Specialized dependencies**: Tools like graph-tool require complex C++ compilation and exhibit platform-specific limitations [3]

Traditional visualization approaches often fail to capture the interactive and dynamic nature of neuronal circuit analysis [4]. Static images cannot convey the exploratory process of network investigation, while complex installation procedures create barriers for peer reviewers and collaborators [5].

Here we present a standardized protocol that addresses these visualization challenges through integrated modern tooling. Our approach builds upon recent advances in reproducible research infrastructure [6,7] and implements the FAIR (Findable, Accessible, Interoperable, Reusable) principles for scientific software [8].

### Key Innovations for Neuronal Circuit Visualization

1. **Integrated visualization pipeline**: Combines graph-tool's native drawing capabilities with matplotlib/seaborn for publication-quality static visualizations
2. **Animated GIF generation**: Capability to create dynamic visualizations showing network evolution, community detection processes, or parameter sweeps
3. **pixi-based dependency management** replacing traditional conda environments with improved cross-platform consistency [9]
4. **MyST-powered documentation** with Jupyter Book integration enabling executable visualization protocols [10]
5. **Neurolibre/MyBinder compatibility** enabling zero-installation peer review and interactive exploration [11]

This protocol enables researchers to implement reproducible neuronal circuit visualization workflows that meet contemporary open science standards while maintaining computational rigor essential for neuroscience applications.

## Results

### Technical Validation

Our implementation successfully integrates the following visualization components:

- **Core Dependencies**: graph-tool 2.98 (with native drawing capabilities), Python 3.11, numpy 1.26, scipy 1.11, pandas 2.1, networkx 3.2
- **Visualization Libraries**: matplotlib 3.8, seaborn 0.12, graph-tool.draw module
- **Environment Management**: pixi workspace configuration with explicit platform constraints (linux-64, osx-64)
- **Documentation System**: MyST v1.8.2 with Node.js v20.20.1 requirement
- **Cloud Compatibility**: Complete MyBinder configuration with postBuild automation

The workflow has been validated on Ubuntu 24.04 (linux-64) with successful execution of all computational tasks including:
- Graph-tool import and version verification (v2.98)
- Jupyter Notebook launch with interactive analysis capabilities  
- Generation of comparative visualizations (strip plots, box plots, heatmaps, clustermaps)
- MyST documentation build processing 3 source files
- Unit test execution with pytest framework

### Visualization Capabilities

The framework provides comprehensive visualization capabilities for neuronal circuit models:

#### Static Visualizations
- **Network topology plots**: Using graph-tool.draw for direct network visualization with customizable node/edge properties
- **Statistical summaries**: strip plots and box plots showing distributions of network metrics across different circuit models
- **Correlation analysis**: Heatmaps and clustermaps revealing relationships between different graph-theoretical measures
- **Comparative analysis**: Side-by-side visualizations enabling direct comparison of multiple neuronal circuit configurations

#### Dynamic Visualizations
- **Animated GIFs**: Time-series visualizations showing network evolution, community detection processes, or parameter optimization
- **Interactive exploration**: Jupyter widgets enabling real-time parameter adjustment and visualization updates
- **Multi-panel layouts**: Combined visualizations showing complementary aspects of neuronal circuit organization

### Representative Results

To address limitations in centrality measure variation observed in unweighted neuronal circuit models, we implemented a structural weighting approach that preserves network topology while introducing meaningful edge weights. This methodology enhances the discriminative power of graph-theoretical algorithms and enables comprehensive multi-metric analysis suitable for publication-quality visualization.

**Weighting Methodology**: We applied uniform random edge weights (range: 0.1-1.0) to break structural symmetries inherent in unweighted neuronal networks while maintaining their fundamental connectivity patterns. This approach ensures reproducible results (using fixed random seed) and generates continuous centrality distributions without altering the underlying network architecture.

**Enhanced Centrality Coverage**: The weighted networks demonstrate significantly improved variation across centrality measures:
- **TC2CT model**: 8/9 metrics with meaningful variation (60-600 unique values per metric)
- **TC2PT model**: 7/9 metrics with meaningful variation (17-420 unique values per metric)  
- **Larger models (TC2IT2PTCT, TC2IT4_IT2CT)**: 7/9 metrics with excellent variation (30-4086 unique values per metric)

This represents a substantial improvement over unweighted networks, which exhibited binary or constant outputs for 6/9 centrality algorithms, severely limiting analytical utility.

Visualize distinct connectivity patterns and hierarchical organization across different circuit configurations with cairo plots, as Figure 1, which provide much insightful views into neuronal networks' functional roles.

**Figure 1**: Cairo plots, nested stochastic block models, and condensation graphs for four representative neuronal circuit models (*i.e.*, TC2CT, TC2IT2PTCT, TC2IT4_IT2CT, TC2PT) demonstrating the structural organization and community structure of each network. 

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_graph.png" alt="TC2CT cairo plot" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A-1) TC2CT</strong> - Direct thalamocortical to corticothalamic connectivity</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_main.png" alt="TC2CT Main Layout" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A-2) TC2CT Main Layout</strong> - population nodes layout from Direct thalamocortical to corticothalamic connectivity</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_ilist.png" alt="TC2CT input nodes Layout" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A-3) TC2CT Input Nodes Layout</strong> - input nodes layout from Direct thalamocortical to corticothalamic connectivity</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_block.png" alt="TC2CT nested stochastic block model" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A-4) TC2CT Nested Stochastic Block Model</strong> - Layered connectivity structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_block_condesne.png" alt="TC2CT condensation graph" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A-5) TC2CT Condensation Graph</strong> - Simplified representation of the network structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT2PTCT/TC2IT2PTCT_graph.png" alt="TC2IT2PTCT cairo plot" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(B-1) TC2IT2PTCT</strong> - Complex multi-layer interactions</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT2PTCT/TC2IT2PTCT_block.png" alt="TC2IT2PTCT Nested Stochastic Block Model" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(B-2) TC2IT2PTCT Nested Stochastic Block Model</strong> - Layered connectivity structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT2PTCT/TC2IT2PTCT_block_condesne.png" alt="TC2IT2PTCT condensation graph" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(B-3) TC2IT2PTCT Condensation Graph</strong> - Simplified representation of the network structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT4_IT2CT/TC2IT4_IT2CT_graph.png" alt="TC2IT4_IT2CT cairo plot" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(C-1) TC2IT4_IT2CT</strong> - Layer 4 intratelencephalic pathways</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT4_IT2CT/TC2IT4_IT2CT_block.png" alt="TC2IT4_IT2CT Nested Stochastic Block Model" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(C-2) TC2IT4_IT2CT Nested Stochastic Block Model</strong> - Layered connectivity structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT4_IT2CT/TC2IT4_IT2CT_block_condesne.png" alt="TC2IT4_IT2CT condensation graph" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(C-3) TC2IT4_IT2CT Condensation Graph</strong> - Simplified representation of the network structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2PT/TC2PT_graph.png" alt="TC2PT cairo plot" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(D-1) TC2PT</strong> - Thalamocortical to pyramidal tract connectivity</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2PT/TC2PT_block.png" alt="TC2PT Nested Stochastic Block Model" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(D-2) TC2PT Nested Stochastic Block Model</strong> - Layered connectivity structure</p>
</div>

<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2PT/TC2PT_block_condesne.png" alt="TC2PT condensation graph" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(D-3) TC2PT Condensation Graph</strong> - Simplified representation of the network structure</p>
</div>
</div>

**Figure 2**: Individual centrality measure visualizations for the TC2PT neuronal circuit model. This figure displays nine different graph-theoretical centrality algorithms applied to the same network, revealing complementary perspectives on neuronal importance and functional roles: (A)  **Betweenness centrality** quantifying control over information flow; (B) **Closeness centrality** assessing proximity to all other neurons; (C) **Eigenvector centrality** identifying neurons connected to other important neurons; (D) **PageRank centrality** modeling random walk importance; (E) **Katz centrality** capturing influence through network paths; (F) **HITS Authority** measuring received importance; (G) **HITS Hub** quantifying distributed importance; (H) **EigenTrust centrality** computing global trust from local relationships; (I) **Centrality Distribution** - Spread of centrality values across the network. 

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_betweenness.png" alt="TC2PT Betweenness Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(A) Betweenness</strong> - Information flow control</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_closeness.png" alt="TC2PT Closeness Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(B) Closeness</strong> - Network proximity</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_eigenvector.png" alt="TC2PT Eigenvector Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(C) Eigenvector</strong> - Connected to important neurons</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_pr.png" alt="TC2PT PageRank Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(D) PageRank</strong> - Random walk importance</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_katz.png" alt="TC2PT Katz Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(E) Katz</strong> - Influence through paths</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_hitsX.png" alt="TC2PT HITS Authority" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(F) HITS Authority</strong> - Received importance</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_hitsY.png" alt="TC2PT HITS Hub" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(G) HITS Hub</strong> - Distributed importance</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_trust_transitivity.png" alt="TC2PT Trust_transitivity Centrality" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(H) trust_transitivity</strong> - Global trust computation</p>
</div>
<div style="flex: 0 0 30%; text-align: center;">
<img src="metrics/TC2PT/TC2PT_facetgrid.png" alt="TC2PT Centrality distributions" style="width: 100%; height: auto;"/>
<p style="font-size: 0.85em; margin-top: 5px;"><strong>(I) Facet Grid</strong> - Distribution of centrality measures</p>
</div>

The animation shows how the algorithm progressively refines community assignments to optimize the likelihood of the observed connectivity pattern.

**Figure 3**: Animated GIF demonstrating the identification of community structure via stochastic block model (SBM) inference. 

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2CT/TC2CT_graph.gif" alt="TC2CT Community Evolution" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A) TC2CT</strong> - Direct thalamocortical connectivity</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT2PTCT/TC2IT2PTCT_graph.gif" alt="TC2IT2PTCT Community Evolution" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(B) TC2IT2PTCT</strong> - Multi-layer interactions</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2IT4_IT2CT/TC2IT4_IT2CT_graph.gif" alt="TC2IT4_IT2CT Community Evolution" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(C) TC2IT4_IT2CT</strong> - Layer 4 intratelencephalic pathways</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="graph_draw/TC2PT/TC2PT_graph.gif" alt="TC2PT Community Evolution" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(D) TC2PT</strong> - Thalamocortical to pyramidal tract</p>
</div>
</div>


**Figure 4**:

Hierarchical Community Structure Analysis Across Neuronal Circuit Models

The nested stochastic block model (SBM) reveals multi-scale community organization in neuronal circuits, showing how neurons cluster into functional modules at different hierarchical levels. Each panel displays the hierarchical clustering structure for a different circuit model, highlighting the complex modular architecture that underlies neural information processing.

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 32%; text-align: center;">
<img src="hierarchy/TC2PT/TC2PT_hierarchy_0.png" alt="TC2PT Hierarchical Structure" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(A) TC2PT</strong> - Thalamocortical to pyramidal tract connectivity</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="hierarchy/max_CTC_plus/max_CTC_plus_hierarchy_0.png" alt="max_CTC_plus Hierarchical Structure" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(B) max_CTC_plus</strong> - Direct thalamocortical connectivity</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="hierarchy/M1_max_plus/M1_max_plus_hierarchy_0.png" alt="M1_max_plus Hierarchical Structure" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(C) M1_max_plus</strong> - Complex multi-layer interactions</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="hierarchy/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_hierarchy_0.png" alt="M2aM1aS1a_max_plus Hierarchical Structure" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(D) M2aM1aS1a_max_plus</strong> - Layer 4 intratelencephalic pathways</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="hierarchy/M2M1S1_max_plus/M2M1S1_max_plus_hierarchy_0.png" alt="M2M1S1_max_plus Hierarchical Structure" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em; margin-top: 5px;"><strong>(E) M2M1S1_max_plus</strong> - Direct thalamocortical to corticothalamic connectivity</p>
</div>
</div>

**Figure 5**: Univariate Centrality Distribution Analysis Across Weighted Neuronal Circuit Models

Each subplot presents kernel density estimation (KDE) plots showing the distribution of individual centrality measures for each weighted neuronal circuit model. The enhanced structural weighting reveals continuous variation patterns across multiple centrality metrics, demonstrating improved discriminative power compared to unweighted analyses. These distributions highlight the rich heterogeneity in network roles across different neuronal populations.

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<img src="centrality/TC2PT/TC2PT_bt.png" alt="TC2PT Betweenness Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_c.png" alt="TC2PT Closeness Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_hitsX.png" alt="TC2PT Hits_x Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_hitsY.png" alt="TC2PT Hits_y Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_katz.png" alt="TC2PT Katz Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_pr.png" alt="TC2PT PageRank Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_t.png" alt="TC2PT Eigentrust Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_tt.png" alt="TC2PT Trust_transitivity Centrality" style="width: 32%; height: auto;">
<img src="centrality/TC2PT/TC2PT_V.png" alt="TC2PT Eigenvector Centrality" style="width: 32%; height: auto;">
</div>

<div style="display: flex; flex-direction: column; gap: 20px; margin: 20px 0;">
<img src="robust_calibrated/plots_kde/TC2PT_centrality_kde.png" alt="TC2PT Centrality Distributions" style="width: 100%; height: auto;">
<p style="font-size: 0.9em; text-align: center;"><strong>(A) TC2PT:</strong> Structural weighting reveals continuous variation patterns across multiple centrality metrics, enabling meaningful differentiation of neuronal roles.</p>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<img src="centrality/max_CTC_plus/max_CTC_plus_bt.png" alt="max_CTC_plus Betweenness Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_c.png" alt="max_CTC_plus Closeness Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_hitsX.png" alt="max_CTC_plus Hits_x Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_hitsY.png" alt="max_CTC_plus Hits_y Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_katz.png" alt="max_CTC_plus Katz Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_pr.png" alt="max_CTC_plus PageRank Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_t.png" alt="max_CTC_plus Eigentrust Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_tt.png" alt="max_CTC_plus Trust_transitivity Centrality" style="width: 32%; height: auto;">
<img src="centrality/max_CTC_plus/max_CTC_plus_V.png" alt="max_CTC_plus Eigenvector Centrality" style="width: 32%; height: auto;">
</div>

<img src="robust_calibrated/plots_kde/max_CTC_plus_centrality_kde.png" alt="max_CTC_plus Centrality Distributions" style="width: 100%; height: auto;">
<p style="font-size: 0.9em; text-align: center;"><strong>(B) max_CTC_plus:</strong> Direct connectivity model demonstrates substantial variation across centrality measures, supporting detailed network role analysis.</p>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<img src="centrality/M1_max_plus/M1_max_plus_bt.png" alt="M1_max_plus Betweenness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_c.png" alt="M1_max_plus Closeness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_hitsX.png" alt="M1_max_plus Hits_x Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_hitsY.png" alt="M1_max_plus Hits_y Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_katz.png" alt="M1_max_plus Katz Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_pr.png" alt="M1_max_plus PageRank Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_t.png" alt="M1_max_plus Eigentrust Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_tt.png" alt="M1_max_plus Trust_transitivity Centrality" style="width: 32%; height: auto;">
<img src="centrality/M1_max_plus/M1_max_plus_V.png" alt="M1_max_plus Eigenvector Centrality" style="width: 32%; height: auto;">
</div>

<img src="robust_calibrated/plots_kde/M1_max_plus_centrality_kde.png" alt="M1_max_plus Centrality Distributions" style="width: 100%; height: auto;">
<p style="font-size: 0.9em; text-align: center;"><strong>(C) M1_max_plus:</strong> Complex multi-layer interactions model exhibits rich continuous variation patterns across centrality distributions.</p>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_bt.png" alt="M2aM1aS1a_max_plus Betweenness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_c.png" alt="M2aM1aS1a_max_plus Closeness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_hitsX.png" alt="M2aM1aS1a_max_plus Hits_x Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_hitsY.png" alt="M2aM1aS1a_max_plus Hits_y Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_katz.png" alt="M2aM1aS1a_max_plus Katz Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_pr.png" alt="M2aM1aS1a_max_plus PageRank Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_t.png" alt="M2aM1aS1a_max_plus Eigentrust Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_tt.png" alt="M2aM1aS1a_max_plus Trust_transitivity Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2aM1aS1a_max_plus/M2aM1aS1a_max_plus_V.png" alt="M2aM1aS1a_max_plus Eigenvector Centrality" style="width: 32%; height: auto;">
</div>

<img src="robust_calibrated/plots_kde/M2aM1aS1a_max_plus_centrality_kde.png" alt="M2aM1aS1a_max_plus Centrality Distributions" style="width: 100%; height: auto;">
<p style="font-size: 0.9em; text-align: center;"><strong>(D) M2aM1aS1а_max_plus:</strong> Layer 4 intratelencephalic pathways model shows comprehensive centrality coverage with detailed distribution patterns.</p>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_bt.png" alt="M2M1S1_max_plus Betweenness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_c.png" alt="M2M1S1_max_plus Closeness Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_hitsX.png" alt="M2M1S1_max_plus Hits_x Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_hitsY.png" alt="M2M1S1_max_plus Hits_y Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_katz.png" alt="M2M1S1_max_plus Katz Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_pr.png" alt="M2M1S1_max_plus PageRank Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_t.png" alt="M2M1S1_max_plus Eigentrust Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_tt.png" alt="M2M1S1_max_plus Trust_transitivity Centrality" style="width: 32%; height: auto;">
<img src="centrality/M2M1S1_max_plus/M2M1S1_max_plus_V.png" alt="M2M1S1_max_plus Eigenvector Centrality" style="width: 32%; height: auto;">
</div>

<img src="robust_calibrated/plots_kde/M2M1S1_max_plus_centrality_kde.png" alt="M2M1S1_max_plus Centrality Distributions" style="width: 100%; height: auto;">
<p style="font-size: 0.9em; text-align: center;"><strong>(E) M2M1S1_max_plus:</strong> Direct thalamocortical to corticothalamic connectivity model exhibits comprehensive centrality coverage with detailed distribution patterns.</p>
</div>

**Figure 6**: Comparative Analysis of Neuronal Circuit Dynamics Across Models

This figure presents comprehensive dynamic visualization results comparing basic and advanced implementations across different neuronal circuit models. The analysis includes state propagation dynamics, edge rewiring patterns, and combined state-rewiring interactions, demonstrating the rich temporal evolution of network properties under different simulation paradigms.

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 48%; text-align: center;">
<img src="TC2PT_dynamic.gif" alt="TC2PT Dynamic layout" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">TC2PT Dynamic</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="TC2PT_dynamic_adv.gif" alt="TC2PT advanced Dynamic layout" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">TC2PT Dynamic Adv</p>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; margin: 12px 0;">
<img src="analysis/plots/pair_dynamic_artistic.png" alt="pair_dynamic" style="width: 60%; height: auto;"/>
<p style="font-size: 0.9em; text-align: center; margin-top: 8px;">Pair: gt_dynamic vs gt_dynamic_adv</p>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 48%; text-align: center;">
<img src="max_CTC_plus_state.gif" alt="Max CTC Plus State" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">Max CTC Plus State</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="max_CTC_plus_state_adv.gif" alt="Max CTC Plus State Adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">Max CTC Plus State Adv</p>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; margin: 12px 0;">
<img src="analysis/plots/pair_state_artistic.png" alt="pair_state" style="width: 60%; height: auto;"/>
<p style="font-size: 0.9em; text-align: center; margin-top: 8px;">Pair: gt_state vs gt_state_adv</p>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 48%; text-align: center;">
<img src="M1_max_plus_combined.gif" alt="M1 Max Plus Combined" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M1 Max Plus Combined</p>
</div>
<div style="flex: 0 0 48%; text-align: center;">
<img src="M1_max_plus_combined_adv.gif" alt="M1 Max Plus Combined Adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M1 Max Plus Combined Adv</p>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; margin: 12px 0;">
<img src="analysis/plots/pair_combined_artistic.png" alt="pair_combined" style="width: 60%; height: auto;"/>
<p style="font-size: 0.9em; text-align: center; margin-top: 8px;">Pair: gt_state_dynamic vs gt_state_dynamic_adv</p>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2aM1aS1a_max_plus_dynamic.gif" alt="M2aM1aS1a Max Plus Dynamic layout" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2aM1aS1a Max Plus Dynamic</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2aM1aS1a_max_plus_state.gif" alt="M2aM1aS1a Max Plus SIRS State animation" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2aM1aS1a Max Plus State</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2aM1aS1a_max_plus_combined.gif" alt="M2aM1aS1a Max Plus Dynamic layout of SIRS State animation" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2aM1aS1a Max Plus Combined</p>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; margin: 12px 0;">
<img src="analysis/plots/triple_nonadv_artistic.png" alt="triple_nonadv" style="width: 60%; height: auto;"/>
<p style="font-size: 0.9em;">Triple (non-adv): gt_dynamic, gt_state, gt_state_dynamic</p>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2M1S1_max_plus_dynamic_adv.gif" alt="M2M1S1 Max Plus Dynamic Adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2M1S1 Max Plus Dynamic Adv</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2M1S1_max_plus_state_adv.gif" alt="M2M1S1 Max Plus State Adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2M1S1 Max Plus State Adv</p>
</div>
<div style="flex: 0 0 32%; text-align: center;">
<img src="M2M1S1_max_plus_combined_adv.gif" alt="M2M1S1 Max Plus Combined Adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">M2M1S1 Max Plus Combined Adv</p>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; margin: 12px 0;">
<img src="analysis/plots/triple_adv_artistic.png" alt="triple_adv" style="width: 100%; height: auto;"/>
<p style="font-size: 0.9em;">Triple (adv): gt_dynamic_adv, gt_state_adv, gt_state_dynamic_adv</p>
</div>

### Neurolibre Compliance Verification

The repository meets all Neurolibre publication requirements:
- ✅ Complete Binder configuration (environment.yml, postBuild, runtime.txt)
- ✅ Academic citation support (CITATION.cff with proper metadata)
- ✅ Neuroscience context documentation (protocol_document.md)
- ✅ Professional documentation structure (_toc.yml, _config.yml)
- ️✅ Cross-platform environment specification (linux-64, osx-64)
- ✅ Reproducible research practices (Jupyter Book integration)
- ✅ Visualization-focused methodology with neuronal circuit applications

## Discussion

### Advantages of This Visualization Protocol

Our approach offers several significant advantages over traditional neuronal circuit visualization workflows:

1. **Enhanced Reproducibility**: Complete environment specification eliminates "works on my machine" issues through explicit dependency pinning and platform constraints.

2. **Comprehensive Visualization Suite**: Integrated static and dynamic visualization capabilities provide multiple perspectives on neuronal circuit organization, from topological structure to statistical properties.

3. **Reduced Barrier to Entry**: Single-command installation (`pixi install`) replaces complex dependency management procedures that previously required manual compilation of graph-tool.

4. **Academic Standards Compliance**: Integration of CITATION.cff supports proper scholarly attribution and enables automatic citation generation through platforms like Zenodo.

5. **Reviewer-Friendly**: MyBinder enables zero-installation peer review, allowing reviewers to immediately execute the computational workflow and interact with visualizations without local environment setup.

6. **Publication-Ready Output**: Built-in matplotlib and seaborn integration ensures visualizations meet journal quality standards with minimal post-processing.

### Network Centrality Measures

Our framework implements eight fundamental network centrality measures to characterize neuronal circuit organization:

**PageRank Centrality** [13] quantifies the importance of nodes based on the structure formulate incoming links, using a damping factor to model random navigation behavior.

**Betweenness Centrality** [15] measures the extent to which a node lies on shortest paths between other nodes, identifying critical bridges in information flow.

**Closeness Centrality** [16] captures how close a node is to all other nodes in the network, reflecting efficiency of information propagation.

**Eigenvector Centrality** [18, 14] assigns relative scores to nodes based on the principle that connections to high-scoring nodes contribute more than connections to low-scoring nodes.

**Katz Centrality** [14] extends eigenvector centrality by incorporating both direct and indirect connections with exponential decay based on path length.

**HITS (Hyperlink-Induced Topic Search) Centrality** [14] computes separate hub and authority scores, where good hubs point to many good authorities and good authorities are pointed to by many good hubs.

**EigenTrust Centrality** [20] models trust transitivity in networks, where trust in a node is determined by the trustworthiness of nodes that trust it.

**Trust Transitivity Centrality** [21] extends trust modeling by considering weighted paths and structural constraints in trust propagation.

These measures provide complementary perspectives on network structure, enabling comprehensive analysis of neuronal circuit organization and function.

### Limitations

Despite its advantages, our protocol has several limitations:

1. **Platform Constraints**: Windows incompatibility due to graph-tool limitations restricts accessibility for some users. This is an inherent limitation of the graph-tool library rather than our workflow design.

2. **Additional Dependencies**: Node.js ≥20 requirement for documentation adds complexity, though this is offset by the enhanced documentation capabilities.

3. **Internet Dependency**: Initial setup requires network connectivity for package resolution, though subsequent usage can be offline.

4. **Memory Requirements**: Large neuronal circuits may require substantial memory for visualization, particularly for animated GIF generation.

### Applications Beyond Standard Neuroscience

While designed specifically for neuronal circuit visualization, this protocol applies broadly to any domain requiring complex network visualization:
- Social network analysis with community detection
- Transportation network visualization and optimization  
- Biological pathway analysis and gene regulatory networks
- Infrastructure network resilience assessment and visualization

### Future Directions

Several enhancements could further improve this visualization protocol:

1. **3D Visualization Integration**: Incorporating tools like Plotly or Mayavi for three-dimensional neuronal circuit visualization.

2. **Real-time Streaming**: Adding capability to visualize live neuronal activity data streams alongside structural connectivity.

3. **Virtual Reality Support**: Enabling immersive exploration of large-scale neuronal circuits in VR environments.

4. **Automated Figure Generation**: Creating templates for common neuroscience journal figure formats with automatic layout optimization.

5. **Cloud Integration**: Direct deployment to neurolibre.com and other platforms could streamline the publication process.

## Methods

### Repository Structure and Configuration

The complete implementation is available at https://github.com/trernghwhuare/metrics-analysis-project with the following key components:

#### Core Configuration Files

- **[pixi.toml](file:///home/leo520/my/metrics-analysis-project/pixi.toml)**: Workspace configuration specifying dependencies, platforms, and tasks
- **[pyproject.toml](file:///home/leo520/my/metrics-analysis-project/pyproject.toml)**: Package metadata and build configuration  
- **[CITATION.cff](file:///home/leo520/my/metrics-analysis-project/CITATION.cff)**: Academic citation metadata
- **[package.json](file:///home/leo520/my/metrics-analysis-project/package.json)**: Local MyST npm installation

#### Documentation Infrastructure

- **[_toc.yml](file:///home/leo520/my/metrics-analysis-project/_toc.yml)**: Table of contents for Jupyter Book
- **[_config.yml](file:///home/leo520/my/metrics-analysis-project/_config.yml)**: Jupyter Book configuration
- **[protocol_document.md](file:///home/leo520/my/metrics-analysis-project/protocol_document.md)**: Neuroscience methodology documentation

#### Visualization Components

- **[Network_Metrics_Analysis.ipynb](file:///home/leo520/my/metrics-analysis-project/Network_Metrics_Analysis.ipynb)**: Interactive notebook demonstrating neuronal circuit visualization capabilities
- **network_metrics_package/plotting/**: Modular plotting functions for strip plots, box plots, heatmaps, and clustermaps
- **results/**: Directory for storing generated visualizations (images, GIFs, and interactive outputs)

#### MyBinder Configuration

- **[binder/environment.yml](file:///home/leo520/my/metrics-analysis-project/binder/environment.yml)**: Conda environment specification
- **[binder/postBuild](file:///home/leo520/my/metrics-analysis-project/binder/postBuild)**: Automated package installation and documentation build
- **[binder/runtime.txt](file:///home/leo520/my/metrics-analysis-project/binder/runtime.txt)**: Python version specification

### Workflow Execution

#### Local Development Commands

```bash
# Environment setup
git clone https://github.com/trernghwhuare/metrics-analysis-project.git
cd metrics-analysis-project
pixi install

# Interactive analysis and visualization
pixi run notebook

# Documentation generation  
pixi run build-docs  # Access at http://localhost:3000

# Testing and validation
pixi run test
pixi run analyze
```

#### Visualization-Specific Workflow

1. **Load neuronal circuit data** from NeuroML (.net.nml) files using `extract_gt_params.py` to generate graph-tool compatible formats with comprehensive metadata

2. **Perform dynamic and state-based analysis** using specialized scripts for neuronal network activity simulation:
   - `gt_dynamic.py` / `gt_dynamic_adv.py`: Edge rewiring pattern analysis with basic and advanced variants
   - `gt_state.py` / `gt_state_adv.py`: Node state dynamics (SIRS epidemic) analysis with basic and advanced variants  
   - `gt_state_dynamic.py` / `gt_state_dynamic_adv.py`: Combined state-dynamic analysis for complex evolutionary models
   - `analysis/compare_algorithms.py`: Automated statistical comparison pipeline generating pairwise and triple comparisons

3. **Generate hierarchical and modular visualizations** using static visualization scripts:
   - `gt_hierarchy.py`: Hierarchical community structure visualization with nested blockmodel analysis
   - `gt_graphdraw.py`: Modular network drawing with anatomical positioning and edge type differentiation  
   - `gt_metrics.py`: Comprehensive centrality metrics visualization with FacetGrid support

4. **Create statistical visualizations** through the calibration pipeline:
   - `robust_calibrate_centrality.py`: Performs robust centrality calibration across multiple network models
   - `plot_pairplots.py`: Creates univariate KDE distributions and scatterplot matrices comparing centralities versus degree

5. **Save results** to dedicated output directories (`hierarchy/`, `graph_draw/`, `metrics/`, `robust_calibrated/plots/`, `analysis/plots/`) for inclusion in publications

#### Cloud Deployment

The MyBinder badge in [README.md](file:///home/leo520/my/metrics-analysis-project/README.md) provides immediate access to the complete computational environment without local installation requirements, enabling interactive exploration of neuronal circuit visualizations.

### Technical Specifications

**Platform Support**: Linux (x86_64), macOS (x86_64, arm64)  
**Python Version**: ≥3.8, <3.12  
**Node.js Version**: ≥20 LTS  
**Memory Requirements**: 8GB+ RAM recommended for graph-tool operations and GIF generation  
**Disk Space**: 2GB minimum for environment installation, additional space for result images/GIFs

## References

[1] Bullmore E, Sporns O. Complex brain networks: graph theoretical analysis of structural and functional systems. Nat Rev Neurosci. 2009;10(3):186-198.

[2] Rubinov M, Sporns O. Complex network measures of brain connectivity: uses and interpretations. Neuroimage. 2010;52(3):1059-1069.

[3] Peixoto TP. The graph-tool python library. Figshare. 2014. doi:10.6084/m9.figshare.1164194

[4] Sandve GK, et al. Ten simple rules for reproducible computational research. PLoS Comput Biol. 2013;9(10):e1003285.

[5] Stodden V, et al. Enhancing reproducibility for computational methods. Science. 2016;354(6317):1240-1241.

[6] Nüst D, et al. Ten simple rules for creating accessible and reproducible computational environments. PLoS Comput Biol. 2019;15(10):e1007004. Available from: https://pmc.ncbi.nlm.nih.gov/articles/PMC6438441/

[7] Wilkinson MD, et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data. 2016;3:160018.

[8] Grüning B, et al. Bioconda: sustainable and comprehensive software distribution for the life sciences. Nat Methods. 2018;15(7):475-476.

[9] Peixoto TP. Descriptive vs. inferential community detection in networks: pitfalls, myths and half-truths. Elements in the Structure and Dynamics of Complex Networks, Cambridge University Press (2023). DOI: 10.1017/9781009118897. arXiv: 2112.00183.

[10] Executable Book Project. Jupyter Book: Create beautiful, publication-ready books and documents from computational content. Journal of Open Source Software. 2020;5(54):2625. DOI: 10.21105/joss.02625.

[11] Bellec P, et al. Neurolibre: An open science platform for neuroimaging education and publishing. Front Neuroinform. 2022;16:882724.

[12] Druskat S, et al. Citation File Format (CFF). 2021. doi:10.5281/zenodo.5171937

[13] Lawrence P, Sergey B, Rajeev M, Terry W. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford University. 1998.

[14] Langville AN, Meyer CD. A Survey of Eigenvector Methods for Web Information Retrieval. SIAM Review. 2005;47(1):135-161. DOI: 10.1137/S0036144503424786

[15] Adamic LA, Glance N. The political blogosphere and the 2004 US Election. In: Proceedings of the WWW-2005 Workshop on the Weblogging Ecosystem. 2005. DOI: 10.1145/1134271.1134277

[16] Closeness centrality. Wikipedia. Available from: https://en.wikipedia.org/wiki/Closeness_centrality

[17] Opsahl T, Agneessens F, Skvoretz J. Node centrality in weighted networks: Generalizing degree and shortest paths. Social Networks. 2010;32:245-251. DOI: 10.1016/j.socnet.2010.03.006

[18] Eigenvector centrality. Wikipedia. Available from: http://en.wikipedia.org/wiki/Centrality#Eigenvector_centrality

[19] Power iteration. Wikipedia. Available from: http://en.wikipedia.org/wiki/Power_iteration

[20] Kamvar SD, Schlosser MT, Garcia-Molina H. The eigentrust algorithm for reputation management in p2p networks. In: Proceedings of the 12th international conference on World Wide Web. 2003:640-651. DOI: 10.1145/775152.775242

[21] Richters O, Peixoto TP. Trust Transitivity in Social Networks. PLoS ONE. 2011;6(4):e18384. DOI: 10.1371/journal.pone.0018384

## Acknowledgments

This work was supported by the principles of open science and reproducible research. We acknowledge the developers of graph-tool, pixi, MyST, Jupyter Book, and MyBinder for their contributions to scientific computing infrastructure.

## Author Contributions

Hua Cheng: Conceptualization, Methodology, Software, Validation, Writing - Original Draft

## Competing Interests

The authors declare no competing interests.

## Data Availability

All code and configuration files are available at https://github.com/trernghwhuare/metrics-analysis-project under the MIT License. Example neuronal circuit datasets and generated visualizations will be made available in the `results/` directory upon publication.

## Keywords

neuronal circuits, network visualization, computational neuroscience, complex networks, graph theory, graph-tool, pixi, MyST, Jupyter Book, MyBinder, Neurolibre, open science, animated visualizations