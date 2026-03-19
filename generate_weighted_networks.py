#!/usr/bin/env python3
"""
Generate weighted versions of neuronal circuit networks to produce continuous centrality measures.

This script addresses the issue where unweighted networks produce binary/discrete 
centrality outputs for certain algorithms (PageRank, Eigenvector, Katz, etc.).

By adding meaningful edge weights based on network topology, we can generate
continuous centrality distributions suitable for KDE visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import graph_tool.all as gt
from network_metrics_package.metrics.generator import compute_and_save_metrics

def add_structural_weights(G, method='degree_product'):
    """
    Add meaningful edge weights to unweighted graph based on structural properties.
    
    Args:
        G: graph_tool.Graph (unweighted)
        method: str, weighting method
            - 'degree_product': weight = degree(source) * degree(target)
            - 'random_uniform': weight = random uniform [0.1, 1.0]
            - 'inverse_distance': weight = 1 / (shortest_path_length + 1)
            - 'jaccard_similarity': weight = Jaccard similarity of neighbors
    
    Returns:
        graph_tool.Graph with edge weights added
    """
    G_weighted = G.copy()
    weight_prop = G_weighted.new_edge_property("double")
    
    if method == 'degree_product':
        # Weight edges by product of endpoint degrees
        deg = G_weighted.degree_property_map("out")
        for e in G_weighted.edges():
            weight_prop[e] = float(deg[e.source()] * deg[e.target()])
            
    elif method == 'random_uniform':
        # Add small random weights to break symmetry while maintaining structure
        np.random.seed(42)  # For reproducibility
        for e in G_weighted.edges():
            weight_prop[e] = np.random.uniform(0.1, 1.0)
            
    elif method == 'jaccard_similarity':
        # Weight by Jaccard similarity of neighbor sets
        for e in G_weighted.edges():
            source_neighbors = set(G_weighted.get_out_neighbors(e.source()))
            target_neighbors = set(G_weighted.get_out_neighbors(e.target()))
            intersection = len(source_neighbors & target_neighbors)
            union = len(source_neighbors | target_neighbors)
            if union > 0:
                weight_prop[e] = intersection / union
            else:
                weight_prop[e] = 0.0
                
    elif method == 'betweenness':
        # Use betweenness as edge weights (requires computation)
        edge_betweenness = gt.edge_betweenness(G_weighted)
        max_bet = max(edge_betweenness.a) if len(edge_betweenness.a) > 0 else 1.0
        for i, e in enumerate(G_weighted.edges()):
            weight_prop[e] = float(edge_betweenness.a[i] / max_bet) if max_bet > 0 else 1.0
            
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to [0.1, 1.0] range to avoid extreme values
    min_weight = min(weight_prop.a) if len(weight_prop.a) > 0 else 0.0
    max_weight = max(weight_prop.a) if len(weight_prop.a) > 0 else 1.0
    
    if max_weight > min_weight:
        for e in G_weighted.edges():
            normalized = 0.1 + 0.9 * (weight_prop[e] - min_weight) / (max_weight - min_weight)
            weight_prop[e] = normalized
    else:
        # All weights are the same, set to 1.0
        for e in G_weighted.edges():
            weight_prop[e] = 1.0
    
    G_weighted.ep.weight = weight_prop
    return G_weighted

def compute_weighted_metrics(G_weighted, output_dir, prefix, normalize=True):
    """
    Compute centrality metrics on weighted graph.
    """
    # Modify the compute_and_save_metrics function to use edge weights
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    logger = logging.getLogger(__name__)
    logger.info("Computing weighted metrics for graph with %d vertices, %d edges", 
                G_weighted.num_vertices(), G_weighted.num_edges())
    
    # Get edge weight property
    weight_prop = G_weighted.ep.weight if "weight" in G_weighted.ep else None
    
    # Set number of threads for OpenMP
    original_threads = gt.openmp_get_num_threads()
    gt.openmp_set_num_threads(8)
    
    try:
        # PageRank with weights
        try:
            if weight_prop is not None:
                metrics['pagerank'] = gt.pagerank(G_weighted, weight=weight_prop).get_array()
            else:
                metrics['pagerank'] = gt.pagerank(G_weighted).get_array()
            metrics['pagerank'] = np.array(metrics['pagerank'], dtype=float)
        except Exception as e:
            logger.warning("Weighted pagerank failed: %s", e)
            metrics['pagerank'] = np.full(G_weighted.num_vertices(), np.nan)

        # Betweenness with weights
        try:
            if weight_prop is not None:
                btw_vp, _ = gt.betweenness(G_weighted, weight=weight_prop)
            else:
                btw_vp, _ = gt.betweenness(G_weighted)
            metrics['betweenness'] = np.array(btw_vp.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted betweenness failed: %s", e)
            metrics['betweenness'] = np.full(G_weighted.num_vertices(), np.nan)

        # Closeness with weights (harmonic)
        try:
            if weight_prop is not None:
                harmonic_closeness = gt.closeness(G_weighted, weight=weight_prop, harmonic=True)
            else:
                harmonic_closeness = gt.closeness(G_weighted, harmonic=True)
            metrics['closeness'] = np.array(harmonic_closeness.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted harmonic closeness failed: %s", e)
            try:
                if weight_prop is not None:
                    standard_closeness = gt.closeness(G_weighted, weight=weight_prop, harmonic=False)
                else:
                    standard_closeness = gt.closeness(G_weighted, harmonic=False)
                metrics['closeness'] = np.array(standard_closeness.get_array(), dtype=float)
            except Exception as e2:
                logger.warning("Standard closeness also failed: %s", e2)
                metrics['closeness'] = np.full(G_weighted.num_vertices(), np.nan)

        # Eigenvector with weights
        try:
            if weight_prop is not None:
                eigenvector_result = gt.eigenvector(G_weighted, weight=weight_prop)
            else:
                eigenvector_result = gt.eigenvector(G_weighted)
            
            if isinstance(eigenvector_result, tuple):
                ev_prop = eigenvector_result[1]  # Second element is the eigenvector
            else:
                ev_prop = eigenvector_result
            
            metrics['eigenvector'] = np.array(ev_prop.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted eigenvector failed: %s", e)
            metrics['eigenvector'] = np.full(G_weighted.num_vertices(), np.nan)

        # Katz with weights
        try:
            if weight_prop is not None:
                katz_result = gt.katz(G_weighted, weight=weight_prop)
            else:
                katz_result = gt.katz(G_weighted)
            metrics['katz'] = np.array(katz_result.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted katz failed: %s", e)
            metrics['katz'] = np.full(G_weighted.num_vertices(), np.nan)

        # HITS (doesn't support weights in graph-tool)
        try:
            h_res = gt.hits(G_weighted)
            if isinstance(h_res, tuple):
                auth = next((x for x in h_res if hasattr(x, "a")), None)
                hub = next((x for x in reversed(h_res) if hasattr(x, "a")), None)
                metrics['hits_authority'] = np.array(auth.get_array() if auth is not None else np.full(G_weighted.num_vertices(), np.nan), dtype=float)
                metrics['hits_hub'] = np.array(hub.get_array() if hub is not None else np.full(G_weighted.num_vertices(), np.nan), dtype=float)
            else:
                tmp = np.array(h_res, dtype=float)
                metrics['hits_authority'] = tmp
                metrics['hits_hub'] = tmp
        except Exception as e:
            logger.warning("HITS failed: %s", e)
            metrics['hits_authority'] = np.full(G_weighted.num_vertices(), np.nan)
            metrics['hits_hub'] = np.full(G_weighted.num_vertices(), np.nan)

        # EigenTrust with weights
        try:
            if weight_prop is not None:
                eigentrust_result = gt.eigentrust(G_weighted, weight=weight_prop)
            else:
                eigentrust_result = gt.eigentrust(G_weighted)
            metrics['eigentrust'] = np.array(eigentrust_result.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted eigentrust failed: %s", e)
            metrics['eigentrust'] = np.full(G_weighted.num_vertices(), np.nan)

        # Trust transitivity with weights
        try:
            vs = list(G_weighted.vertices())
            if vs and weight_prop is not None:
                trust_result = gt.trust_transitivity(G_weighted, weight=weight_prop, source=vs[0])
            elif vs:
                trust_result = gt.trust_transitivity(G_weighted, source=vs[0])
            else:
                trust_result = G_weighted.new_vertex_property("double")
            metrics['trust_transitivity'] = np.array(trust_result.get_array(), dtype=float)
        except Exception as e:
            logger.warning("Weighted trust_transitivity failed: %s", e)
            metrics['trust_transitivity'] = np.full(G_weighted.num_vertices(), np.nan)

    finally:
        gt.openmp_set_num_threads(original_threads)

    # Save metrics
    for k, v in metrics.items():
        # Handle NaN values
        v_clean = np.nan_to_num(v, nan=np.nan)
        metrics[k] = v_clean

    if normalize:
        from network_metrics_package.metrics.utils import minmax_normalize
        for k in list(metrics.keys()):
            metrics[k] = minmax_normalize(metrics[k])

    # Save to files
    npz_path = os.path.join(output_dir, f"{prefix}_metrics.npz")
    np.savez_compressed(npz_path, **metrics)
    df = pd.DataFrame(metrics)
    df.index.name = "vertex_id"
    csv_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    df.to_csv(csv_path, index=True)
    logger.info("Saved weighted metrics to %s and %s", npz_path, csv_path)

    return metrics, npz_path, csv_path

def main():
    """Main function to process all networks."""
    logging.basicConfig(level=logging.INFO)
    
    networks = [
        ("TC2CT.gt", "TC2CT"),
        ("TC2IT2PTCT.gt", "TC2IT2PTCT"), 
        ("TC2IT4_IT2CT.gt", "TC2IT4_IT2CT"),
        ("TC2PT.gt", "TC2PT")
    ]
    
    weighting_methods = ['degree_product', 'random_uniform']
    
    for gt_file, name in networks:
        print(f"\nProcessing {name}...")
        
        # Load original unweighted network
        G = gt.load_graph(gt_file)
        print(f"Original network: {G.num_vertices()} vertices, {G.num_edges()} edges")
        
        for method in weighting_methods:
            print(f"  Applying {method} weighting...")
            
            # Create weighted network
            G_weighted = add_structural_weights(G, method=method)
            
            # Compute weighted metrics
            output_dir = f"metrics_out_weighted_{method}"
            prefix = f"{name}_weighted_{method}"
            
            try:
                metrics, npz_path, csv_path = compute_weighted_metrics(
                    G_weighted, output_dir, prefix, normalize=True
                )
                
                # Print summary statistics
                print(f"    Generated weighted metrics for {name} ({method})")
                for metric_name, values in metrics.items():
                    unique_vals = len(np.unique(values[~np.isnan(values)]))
                    print(f"      {metric_name}: {unique_vals} unique values")
                    
            except Exception as e:
                print(f"    Failed to compute weighted metrics for {name} ({method}): {e}")

if __name__ == "__main__":
    main()