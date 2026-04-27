#!/usr/bin/env python3
"""
Robust Centrality Calibration with Version Compatibility
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import graph_tool.all as gt
    HAS_GRAPH_TOOL = True
except ImportError as e:
    print(f"Missing graph-tool dependency: {e}")
    HAS_GRAPH_TOOL = False


def setup_logging(verbose=False):
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_structural_weights(G):
    """Create meaningful edge weights based on structural properties."""
    # Get vertex degrees
    v_degrees = G.degree_property_map("total")
    
    # Create edge weight property
    w = G.new_edge_property("double")
    num_edges = G.num_edges()
    if num_edges > 0:
        w.a = np.random.random(num_edges) * 42
    else:
        w.a = np.array([], dtype=float)
    
    for e in G.edges():
        source_deg = v_degrees[e.source()]
        target_deg = v_degrees[e.target()]
        # Use degree product + 1 to avoid zero weights
        w[e] = (source_deg + 1) * (target_deg + 1)
    
    return w


def compute_pagerank_safe(G, w=None):
    """Compute PageRank with safe parameters."""
    try:
        # Always call without ret_iter to get consistent return type
        if w is not None:
            pr_result = gt.pagerank(G, weight=w, damping=0.85, max_iter=1000)
        else:
            pr_result = gt.pagerank(G)
        
        # Try to convert to numpy array in multiple ways
        try:
            # Try direct conversion first
            result_array = np.asarray(pr_result, dtype=float)
        except (TypeError, ValueError):
            # If that fails, try to access .a attribute (for graph-tool property maps)
            array_attr = getattr(pr_result, 'a', None)
            if array_attr is not None:
                try:
                    result_array = np.asarray(array_attr, dtype=float)
                except (TypeError, ValueError):
                    result_array = np.zeros(G.num_vertices())
            else:
                result_array = np.zeros(G.num_vertices())
        
        # Ensure correct length
        if len(result_array) != G.num_vertices():
            logging.warning(f"PageRank returned array of length {len(result_array)}, expected {G.num_vertices()}")
            if len(result_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(result_array)] = result_array
                result_array = padded
            else:
                result_array = result_array[:G.num_vertices()]
        
        return result_array
    except Exception as e:
        logging.warning(f"PageRank failed: {e}")
        return np.zeros(G.num_vertices())


def compute_betweenness_safe(G, w=None):
    """Compute betweenness with safe parameters on the full graph."""
    try:
        # Compute on full graph, not just largest component
        if w is not None:
            vb, eb = gt.betweenness(G, weight=w)
        else:
            vb, eb = gt.betweenness(G)
        
        result_array = np.asarray(vb.get_array() if hasattr(vb, 'get_array') else vb, dtype=float)
        
        # Ensure correct length
        if len(result_array) != G.num_vertices():
            logging.warning(f"Betweenness returned array of length {len(result_array)}, expected {G.num_vertices()}")
            if len(result_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(result_array)] = result_array
                result_array = padded
            else:
                result_array = result_array[:G.num_vertices()]
        
        return result_array
    except Exception as e:
        logging.warning(f"Betweenness failed: {e}")
        return np.zeros(G.num_vertices())


def compute_closeness_safe(G, w=None):
    """Compute closeness with safe parameters on the full graph."""
    try:
        if w is not None:
            # Create weight property for the full graph
            weight_map = G.new_edge_property("double")
            for e in G.edges():
                weight_map[e] = w[e]
            cl = gt.closeness(G, weight=weight_map, norm=True, harmonic=True)
        else:
            cl = gt.closeness(G, norm=True, harmonic=True)
            
        # Convert to numpy array
        result_array = np.asarray(cl.get_array() if hasattr(cl, 'get_array') else cl, dtype=float)
        
        # Ensure correct length
        if len(result_array) != G.num_vertices():
            logging.warning(f"Closeness returned array of length {len(result_array)}, expected {G.num_vertices()}")
            if len(result_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(result_array)] = result_array
                result_array = padded
            else:
                result_array = result_array[:G.num_vertices()]
        
        return result_array
        
    except Exception as e:
        logging.warning(f"Closeness failed: {e}")
        return np.zeros(G.num_vertices())


def compute_eigenvector_safe(G, w=None):
    """
    Compute eigenvector centrality with fallback strategies on the full graph.
    """
    try:
        if w is not None:
            # Create weight property for the full graph
            weight_map = G.new_edge_property("double")
            for e in G.edges():
                weight_map[e] = w[e]
            ee, ev = gt.eigenvector(G, weight=weight_map, epsilon=1e-6)
        else:
            ee, ev = gt.eigenvector(G, epsilon=1e-6)
        
        # Safely get the array from the eigenvector property map
        ev_array = getattr(ev, 'a', None) if ev is not None else None
        
        # Handle case where ev_array might be None or invalid
        if ev_array is not None:
            try:
                result_array = np.asarray(ev_array, dtype=float)
            except (TypeError, ValueError):
                result_array = np.zeros(G.num_vertices())
        else:
            result_array = np.zeros(G.num_vertices())
        
        # Ensure correct length
        if len(result_array) != G.num_vertices():
            logging.warning(f"Eigenvector returned array of length {len(result_array)}, expected {G.num_vertices()}")
            if len(result_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(result_array)] = result_array
                result_array = padded
            else:
                result_array = result_array[:G.num_vertices()]
        
        # Check if eigenvector is all zeros or has very small values
        if np.allclose(result_array, 0, atol=1e-10):
            logging.warning("Eigenvector centrality returned all zeros - using PageRank as fallback")
            return compute_pagerank_safe(G, w)
        else:
            return result_array
            
    except Exception as e:
        logging.warning(f"Eigenvector failed: {e} - using PageRank as fallback")
        return compute_pagerank_safe(G, w)


def compute_katz_safe(G, w=None):
    """
    Compute Katz centrality with proper weight handling to avoid 'key_type' errors.
    The issue occurs when w is passed incorrectly.
    """
    try:
        # For Katz centrality, we need to choose alpha carefully
        max_degree = np.max(G.degree_property_map("total").get_array())
        alpha = 0.9 / (max_degree + 1)  # Conservative choice
        if w is not None:
            # Properly attach weight property to graph as edge property
            weight_map = G.new_edge_property("double")
            for e in G.edges():
                weight_map[e] = w[e]
            
            # Create beta as vertex property map (all ones)
            beta_map = G.new_vertex_property("double")
            beta_map.a = np.ones(G.num_vertices())
            
            katz = gt.katz(G, weight=weight_map, alpha=alpha, beta=beta_map, epsilon=1e-6)
        else:
            # Without weights, beta can be None (defaults to 1 for all vertices)
            katz = gt.katz(G, alpha=alpha, beta=None, epsilon=1e-6)
            
        katz_array = getattr(katz, 'a', None)
        if katz_array is not None:
            try:
                katz_array = np.asarray(katz_array, dtype=float)
            except (TypeError, ValueError):
                katz_array = None
        
        # Ensure correct length
        if katz_array is not None and len(katz_array) != G.num_vertices():
            logging.warning(f"Katz returned array of length {len(katz_array)}, expected {G.num_vertices()}")
            if len(katz_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(katz_array)] = katz_array
                katz_array = padded
            else:
                katz_array = katz_array[:G.num_vertices()]
        elif katz_array is None:
            katz_array = np.zeros(G.num_vertices())
        
        # Check if Katz is all zeros
        if np.allclose(katz_array, 0, atol=1e-10):
            logging.warning("Katz centrality returned all zeros - using PageRank as fallback")
            return compute_pagerank_safe(G, w)
        else:
            return katz_array
            
    except Exception as e:
        logging.warning(f"Katz failed: {e} - using PageRank as fallback")
        return compute_pagerank_safe(G, w)


def compute_hits_safe(G):
    """Compute HITS with version compatibility handling."""
    try:
        hits_result = gt.hits(G, epsilon=1e-6)
        
        # Handle different return signatures across graph-tool versions
        if isinstance(hits_result, tuple):
            if len(hits_result) == 3:
                ee, hits_auth, hits_hub = hits_result
            else:
                raise ValueError(f"Unexpected HITS return length: {len(hits_result)}")
        else:
            # Single return value (unlikely but possible)
            raise ValueError("HITS returned unexpected single value")
            
        auth_array = getattr(hits_auth, 'a', None)
        hub_array = getattr(hits_hub, 'a', None)
        
        if auth_array is not None:
            auth_array = np.asarray(auth_array, dtype=float)
        else:
            auth_array = np.zeros(G.num_vertices())
            
        if hub_array is not None:
            hub_array = np.asarray(hub_array, dtype=float)
        else:
            hub_array = np.zeros(G.num_vertices())
        
        # Ensure correct lengths
        if len(auth_array) != G.num_vertices():
            logging.warning(f"HITS authority returned array of length {len(auth_array)}, expected {G.num_vertices()}")
            if len(auth_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(auth_array)] = auth_array
                auth_array = padded
            else:
                auth_array = auth_array[:G.num_vertices()]
                
        if len(hub_array) != G.num_vertices():
            logging.warning(f"HITS hub returned array of length {len(hub_array)}, expected {G.num_vertices()}")
            if len(hub_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(hub_array)] = hub_array
                hub_array = padded
            else:
                hub_array = hub_array[:G.num_vertices()]
        
        return auth_array, hub_array
        
    except Exception as e:
        logging.warning(f"HITS failed: {e}")
        zeros = np.zeros(G.num_vertices())
        return zeros, zeros


def compute_eigentrust_safe(G, w=None):
    """
    Compute eigentrust centrality with fallback strategies.
    Eigentrust requires edge weights, so we handle cases where weights are not available.
    """
    try:
        if w is not None:
            # Properly attach weight property to graph as edge property
            trust_map = G.new_edge_property("double")
            for e in G.edges():
                trust_map[e] = w[e]
            eigentrust_result = gt.eigentrust(G, trust_map, epsilon=1e-6)
        else:
            # Create default uniform weights if none provided
            trust_map = G.new_edge_property("double")
            trust_map.a = np.ones(G.num_edges())
            eigentrust_result = gt.eigentrust(G, trust_map, epsilon=1e-6)
            
        # Handle potential tuple returns (some versions may return additional info)
        if isinstance(eigentrust_result, tuple):
            # Find the element with 'a' attribute (vertex property map)
            eigentrust = None
            for item in reversed(eigentrust_result):
                if hasattr(item, "a"):
                    eigentrust = item
                    break
            if eigentrust is None:
                raise ValueError("No valid vertex property map found in eigentrust result")
        else:
            eigentrust = eigentrust_result
            
        # Convert to numpy array safely
        eigentrust_array = getattr(eigentrust, 'a', None)
        if eigentrust_array is not None:
            try:
                eigentrust_array = np.asarray(eigentrust_array, dtype=float)
            except (TypeError, ValueError):
                eigentrust_array = None
        
        # Ensure correct length
        if eigentrust_array is not None and len(eigentrust_array) != G.num_vertices():
            logging.warning(f"Eigentrust returned array of length {len(eigentrust_array)}, expected {G.num_vertices()}")
            if len(eigentrust_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(eigentrust_array)] = eigentrust_array
                eigentrust_array = padded
            else:
                eigentrust_array = eigentrust_array[:G.num_vertices()]
        elif eigentrust_array is None:
            eigentrust_array = np.zeros(G.num_vertices())
        
        # Check if eigentrust is all zeros or invalid
        if (np.allclose(eigentrust_array, 0, atol=1e-10) or 
            np.any(np.isnan(eigentrust_array))):
            logging.warning("Eigentrust centrality returned invalid values - using PageRank as fallback")
            return compute_pagerank_safe(G, w)
        else:
            return eigentrust_array
            
    except Exception as e:
        logging.warning(f"Eigentrust failed: {e} - using PageRank as fallback")
        return compute_pagerank_safe(G, w)


def compute_trust_transitivity_safe(G, w=None):
    """
    Compute trust transitivity with fallback strategies on the full graph.
    Trust transitivity requires a source vertex and may have different parameter requirements.
    """
    try:
        # Get all vertices for source selection
        vertices = list(G.vertices())
        if not vertices:
            logging.warning("Empty graph for trust transitivity - returning zeros")
            return np.zeros(G.num_vertices())
            
        # Select first vertex as source
        source_vertex = vertices[0]
        
        # Handle weights properly for full graph
        if w is not None:
            trust_map = G.new_edge_property("double")
            for e in G.edges():
                trust_map[e] = w[e]
            trust_trans_result = gt.trust_transitivity(G, trust_map, source=source_vertex)
        else:
            trust_map = G.new_edge_property("double")
            trust_map.a = np.ones(G.num_edges())
            trust_trans_result = gt.trust_transitivity(G, trust_map, source=source_vertex)
            
        # Handle potential tuple returns
        if isinstance(trust_trans_result, tuple):
            trust_trans = None
            for item in reversed(trust_trans_result):
                if hasattr(item, "a"):
                    trust_trans = item
                    break
            if trust_trans is None:
                raise ValueError("No valid vertex property map found in trust_transitivity result")
        else:
            trust_trans = trust_trans_result
            
        # Convert to numpy array
        trust_trans_array = getattr(trust_trans, 'a', None)
        if trust_trans_array is not None:
            try:
                trust_trans_array = np.asarray(trust_trans_array, dtype=float)
            except (TypeError, ValueError):
                trust_trans_array = None
        
        # Ensure correct length
        if trust_trans_array is not None and len(trust_trans_array) != G.num_vertices():
            logging.warning(f"Trust transitivity returned array of length {len(trust_trans_array)}, expected {G.num_vertices()}")
            if len(trust_trans_array) < G.num_vertices():
                padded = np.zeros(G.num_vertices())
                padded[:len(trust_trans_array)] = trust_trans_array
                trust_trans_array = padded
            else:
                trust_trans_array = trust_trans_array[:G.num_vertices()]
        elif trust_trans_array is None:
            trust_trans_array = np.zeros(G.num_vertices())
        
        # Check if trust transitivity is all zeros or invalid
        if np.allclose(trust_trans_array, 0, atol=1e-10) or np.any(np.isnan(trust_trans_array)):
            logging.warning("Trust transitivity returned invalid values - using PageRank as fallback")
            return compute_pagerank_safe(G, w)
        else:
            return trust_trans_array
            
    except TypeError:
        # Handle case where source parameter is not accepted (older versions)
        try:
            if w is not None:
                trust_map = G.new_edge_property("double")
                for e in G.edges():
                    trust_map[e] = w[e]
                trust_trans_result = gt.trust_transitivity(G, trust_map)
            else:
                trust_map = G.new_edge_property("double")
                trust_map.a = np.ones(G.num_edges())
                trust_trans_result = gt.trust_transitivity(G, trust_map)
                
            # Handle potential tuple returns
            if isinstance(trust_trans_result, tuple):
                trust_trans = None
                for item in reversed(trust_trans_result):
                    if hasattr(item, "a"):
                        trust_trans = item
                        break
                if trust_trans is None:
                    raise ValueError("No valid vertex property map found in trust_transitivity result")
            else:
                trust_trans = trust_trans_result
                
            # Convert to numpy array
            trust_trans_array = getattr(trust_trans, 'a', None)
            if trust_trans_array is not None:
                try:
                    trust_trans_array = np.asarray(trust_trans_array, dtype=float)
                except (TypeError, ValueError):
                    trust_trans_array = None
            
            # Ensure correct length
            if trust_trans_array is not None and len(trust_trans_array) != G.num_vertices():
                logging.warning(f"Trust transitivity (no source) returned array of length {len(trust_trans_array)}, expected {G.num_vertices()}")
                if len(trust_trans_array) < G.num_vertices():
                    padded = np.zeros(G.num_vertices())
                    padded[:len(trust_trans_array)] = trust_trans_array
                    trust_trans_array = padded
                else:
                    trust_trans_array = trust_trans_array[:G.num_vertices()]
            elif trust_trans_array is None:
                trust_trans_array = np.zeros(G.num_vertices())
            
            if np.allclose(trust_trans_array, 0, atol=1e-10) or np.any(np.isnan(trust_trans_array)):
                logging.warning("Trust transitivity (no source) returned invalid values - using PageRank as fallback")
                return compute_pagerank_safe(G, w)
            else:
                return trust_trans_array
                
        except Exception as e2:
            logging.warning(f"Trust transitivity failed (both with and without source): {e2} - using PageRank as fallback")
            return compute_pagerank_safe(G, w)
            
    except Exception as e:
        logging.warning(f"Trust transitivity failed: {e} - using PageRank as fallback")
        return compute_pagerank_safe(G, w)


def calibrate_single_network_robust(gt_file, output_dir, use_weights=True, normalize=True):
    """Robustly calibrate a single network handling all compatibility issues."""
    logger = logging.getLogger(__name__)
    gt_path = Path(gt_file)
    network_name = gt_path.stem
    
    logger.info(f"Robustly calibrating network: {network_name}")
    
    # Load graph
    G = gt.load_graph(str(gt_path))
    
    # Create weights if requested
    w = None
    if use_weights:
        w = create_structural_weights(G)
    
    # Compute all centrality metrics with proper handling
    metrics = {}
    
    # Degree (always works)
    metrics['degree'] = G.degree_property_map("total").get_array()
    logger.info("Computed degree")
    
    # PageRank
    metrics['pagerank'] = compute_pagerank_safe(G, w)
    logger.info("Computed PageRank")
    
    # Betweenness
    metrics['betweenness'] = compute_betweenness_safe(G, w)
    logger.info("Computed betweenness")
    
    # Closeness
    metrics['closeness'] = compute_closeness_safe(G, w)
    logger.info("Computed closeness")
    
    # Eigenvector (with fallback)
    metrics['eigenvector'] = compute_eigenvector_safe(G, w)
    logger.info("Computed eigenvector")
    
    # Katz (with proper weight handling)
    metrics['katz'] = compute_katz_safe(G, w)
    logger.info("Computed Katz")
    
    # HITS (with version compatibility)
    auth, hub = compute_hits_safe(G)
    metrics['hits_authority'] = auth
    metrics['hits_hub'] = hub
    logger.info("Computed HITS")
    
    # Eigentrust (with fallback)
    metrics['eigentrust'] = compute_eigentrust_safe(G, w)
    logger.info("Computed eigentrust")
    
    # Trust transitivity (with fallback)
    metrics['trust_transitivity'] = compute_trust_transitivity_safe(G, w)
    logger.info("Computed trust transitivity")
    
    # Normalize if requested
    if normalize:
        for key, values in metrics.items():
            if len(values) > 0:
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    metrics[key] = (values - min_val) / (max_val - min_val)
                else:
                    metrics[key] = np.zeros_like(values)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate that all metric arrays have the correct length
    total_vertices = G.num_vertices()
    validated_metrics = {}
    for key, values in metrics.items():
        values_array = np.asarray(values, dtype=float)
        if len(values_array) != total_vertices:
            logging.warning(f"Metric '{key}' has incorrect length {len(values_array)}, expected {total_vertices}. Padding with zeros.")
            # Pad or truncate to correct length
            if len(values_array) < total_vertices:
                padded = np.zeros(total_vertices)
                padded[:len(values_array)] = values_array
                validated_metrics[key] = padded
            else:
                validated_metrics[key] = values_array[:total_vertices]
        else:
            validated_metrics[key] = values_array
    
    npz_path = output_path / f"{network_name}_robust_calibrated_metrics.npz"
    csv_path = output_path / f"{network_name}_robust_calibrated_metrics.csv"
    
    # Save NPZ
    np.savez_compressed(npz_path, **validated_metrics)
    
    # Save CSV
    df_data = {'vertex_id': np.arange(total_vertices)}
    for key, values in validated_metrics.items():
        df_data[key] = values
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved robust calibrated metrics to {npz_path} and {csv_path}")
    
    # Analyze results
    total_vertices = G.num_vertices()
    for metric, values in metrics.items():
        zeros = np.sum(values == 0)
        zero_pct = (zeros / total_vertices) * 100
        unique_vals = len(np.unique(values[values != 0])) if np.any(values != 0) else 0
        logger.info(f"  {metric}: {zero_pct:.1f}% zeros, {unique_vals} unique non-zero values")
    
    return npz_path


def calibrate_all_networks_robust(input_pattern, output_dir, use_weights=True, normalize=True):
    """Calibrate all networks robustly."""
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    
    # Handle both directory paths and glob patterns
    if '*' in input_pattern or '?' in input_pattern:
        # Treat as glob pattern
        gt_files = list(Path('.').glob(input_pattern))
    else:
        # Treat as directory (search recursively for .gt files in subfolders)
        input_path = Path(input_pattern)
        if input_path.is_dir():
            gt_files = list(input_path.rglob("*.gt"))
        else:
            # If a single .gt file path was provided, use it if it exists
            if input_path.suffix == '.gt' and input_path.exists():
                gt_files = [input_path]
            else:
                gt_files = []
    
    if not gt_files:
        logger.warning(f"No *.gt files found with pattern/dir: {input_pattern}")
        return
    
    logger.info(f"Found {len(gt_files)} network models to calibrate robustly")
    
    successful = 0
    for gt_file in sorted(gt_files):
        try:
            result = calibrate_single_network_robust(
                gt_file, output_dir, use_weights, normalize
            )
            if result:
                successful += 1
        except Exception as e:
            logger.error(f"Failed to process {gt_file.name}: {e}")
            continue
    
    logger.info(f"Robust calibration completed! {successful}/{len(gt_files)} networks processed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robust Centrality Calibration with Version Compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Calibrate all networks with structural weights using default pattern
        python robust_calibrate_centrality.py --output-dir robust_calibrated
        
        # Calibrate using a specific glob pattern
        python robust_calibrate_centrality.py --input-dir "data/*.gt" --output-dir robust_calibrated
        
        # Calibrate from a directory
        python robust_calibrate_centrality.py --input-dir ./networks --output-dir robust_calibrated
        
        # Calibrate without weights
        python robust_calibrate_centrality.py --input-dir . --output-dir robust_calibrated --no-weights
        """
    )
    
    parser.add_argument("--input-dir", default="metrics/*/*.gt",
                       help="Input directory containing *.gt files OR glob pattern like 'metrics/*/*.gt' (default: metrics/*/*.gt)")
    parser.add_argument("--output-dir", default="robust_calibrated", 
                       help="Output directory for calibrated results (default: robust_calibrated)")
    parser.add_argument("--no-weights", dest="use_weights", action="store_false",
                       help="Disable structural weighting")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                       help="Disable min-max normalization")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if not HAS_GRAPH_TOOL:
        print("Required dependencies not available.")
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        calibrate_all_networks_robust(
            args.input_dir, args.output_dir, 
            args.use_weights, args.normalize
        )
        print(f"Robust calibration complete! Results in {args.output_dir}/")
    except Exception as e:
        logging.error(f"Robust calibration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()