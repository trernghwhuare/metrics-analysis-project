import os
import logging
import numpy as np
import pandas as pd
import graph_tool.all as gt
import gc
import traceback
from .utils import sanitize_array, minmax_normalize

logger = logging.getLogger(__name__)

def detect_chain_like_structure(G):
    """
    Detect if the graph has chain-like or feedforward structure that makes
    certain centrality metrics unsuitable.
    
    Returns:
        dict: Detection results with boolean flags and statistics
    """
    try:
        # Basic statistics
        n_vertices = G.num_vertices()
        n_edges = G.num_edges()
        
        if n_vertices == 0:
            return {'is_chain_like': False, 'reason': 'Empty graph'}
            
        # Check if graph is directed
        is_directed = G.is_directed()
        
        # Calculate average degree
        avg_degree = (2 * n_edges) / n_vertices if not is_directed else n_edges / n_vertices
        
        # Check connectivity
        largest_component = gt.label_largest_component(G)
        largest_component_size = np.sum(largest_component.a)
        connectivity_ratio = largest_component_size / n_vertices
        
        # For directed graphs, check if there are many sources/sinks
        if is_directed:
            in_degrees = np.array([v.in_degree() for v in G.vertices()])
            out_degrees = np.array([v.out_degree() for v in G.vertices()])
            
            sources = np.sum(in_degrees == 0)
            sinks = np.sum(out_degrees == 0)
            
            source_ratio = sources / n_vertices
            sink_ratio = sinks / n_vertices
            
            # Chain-like indicators for directed graphs:
            # - High source/sink ratio
            # - Low average degree
            # - Poor connectivity
            is_chain_like = (
                (source_ratio > 0.1 or sink_ratio > 0.1) and
                avg_degree < 3.0 and
                connectivity_ratio < 0.8
            )
            
            reason = f"Directed graph: sources={sources}({source_ratio:.2%}), sinks={sinks}({sink_ratio:.2%}), avg_degree={avg_degree:.2f}, connectivity={connectivity_ratio:.2%}"
            
        else:
            # For undirected graphs, check if it's a path-like structure
            degrees = np.array([v.out_degree() + v.in_degree() for v in G.vertices()])
            endpoints = np.sum(degrees <= 1)
            endpoint_ratio = endpoints / n_vertices
            
            is_chain_like = (
                endpoint_ratio > 0.1 and
                avg_degree < 2.5 and
                connectivity_ratio < 0.9
            )
            
            reason = f"Undirected graph: endpoints={endpoints}({endpoint_ratio:.2%}), avg_degree={avg_degree:.2f}, connectivity={connectivity_ratio:.2%}"
        
        return {
            'is_chain_like': is_chain_like,
            'reason': reason,
            'n_vertices': n_vertices,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'connectivity_ratio': connectivity_ratio
        }
        
    except Exception as e:
        logger.warning(f"Failed to detect chain-like structure: {e}")
        return {'is_chain_like': False, 'reason': f'Detection failed: {e}'}

def warn_about_unsuitable_metrics(G, metrics_to_compute):
    """
    Warn about metrics that may be unsuitable for the given graph topology.
    """
    detection_result = detect_chain_like_structure(G)
    
    if detection_result['is_chain_like']:
        logger.warning("⚠️  DETECTED CHAIN-LIKE OR FEEDFORWARD NETWORK STRUCTURE")
        logger.warning(f"   Reason: {detection_result['reason']}")
        logger.warning("   This topology may cause certain centrality metrics to be:")
        logger.warning("   - Zero or near-zero (closeness, eigenvector, katz)")
        logger.warning("   - Mathematically undefined (HITS in some cases)")
        logger.warning("   - Not meaningful for analysis purposes")
        
        unsuitable_metrics = []
        if 'closeness' in metrics_to_compute:
            unsuitable_metrics.append('closeness')
        if 'eigenvector' in metrics_to_compute:
            unsuitable_metrics.append('eigenvector')
        if 'katz' in metrics_to_compute:
            unsuitable_metrics.append('katz')
        if 'hits_authority' in metrics_to_compute or 'hits_hub' in metrics_to_compute:
            unsuitable_metrics.append('HITS (authority/hub)')
            
        if unsuitable_metrics:
            logger.warning(f"   Potentially problematic metrics: {', '.join(unsuitable_metrics)}")
            logger.warning("   Consider focusing on more robust metrics like PageRank and Betweenness")
    else:
        logger.info("✅ Network topology appears suitable for all centrality metrics")

def safe_compute_metric(computation_func, metric_name, G, fallback_value=None, max_retries=2, **kwargs):
    """
    Safely compute a metric with retry mechanism
    """
    if fallback_value is None:
        fallback_value = np.full(G.num_vertices(), np.nan)
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Computing {metric_name} (attempt {attempt + 1}/{max_retries + 1})...")
            result = computation_func(G, **kwargs)
            gc.collect()  # Force garbage collection
            logger.info(f"{metric_name} computed successfully")
            return result
        except Exception as e:
            logger.warning(f"{metric_name} failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                logger.info("Retrying after short delay...")
                import time
                time.sleep(0.1)  # Brief pause before retry
                gc.collect()
            else:
                logger.error(f"All attempts for {metric_name} failed. Using fallback value.")
                return fallback_value

def safe_convert_to_numpy(obj, size, metric_name="unknown"):
    """
    Safely convert various object types to numpy array
    """
    try:
        # Handle graph-tool property maps
        if hasattr(obj, 'get_array'):
            arr = np.array(obj.get_array(), dtype=float)
            logger.debug(f"Converted property map {metric_name} to numpy array with shape {arr.shape}")
            return arr
            
        # Handle regular arrays/lists
        if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
            arr = np.array(obj, dtype=float)
            logger.debug(f"Converted array {metric_name} to numpy array with shape {arr.shape}")
            return arr
            
        # Handle scalar values
        arr = np.full(size, float(obj), dtype=float)
        logger.debug(f"Converted scalar {metric_name} to numpy array with shape {arr.shape}")
        return arr
        
    except Exception as e:
        logger.warning(f"Failed to convert {metric_name} to numpy array: {e}")
        return np.full(size, np.nan, dtype=float)

def safe_sanitize_and_normalize(metrics, G, normalize):
    """
    Safely sanitize and normalize metrics with error handling
    """
    try:
        logger.info("Starting sanitization process...")
        sanitized_metrics = {}
        
        # Process each metric individually to isolate issues
        for k, v in metrics.items():
            try:
                logger.debug(f"Sanitizing metric: {k}")
                # First convert to numpy array
                numpy_array = safe_convert_to_numpy(v, G.num_vertices(), k)
                
                # Then sanitize
                sanitized_v = sanitize_array(numpy_array)
                
                # Check size and adjust if needed
                if len(sanitized_v) != G.num_vertices():
                    arr = np.full(G.num_vertices(), np.nan, dtype=float)
                    copy_size = min(len(sanitized_v), G.num_vertices())
                    arr[:copy_size] = sanitized_v[:copy_size]
                    sanitized_metrics[k] = arr
                else:
                    sanitized_metrics[k] = sanitized_v
                    
                # Force garbage collection after each metric
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Sanitization failed for {k}: {e}")
                logger.debug(traceback.format_exc())
                sanitized_metrics[k] = np.full(G.num_vertices(), np.nan, dtype=float)
        
        # Normalize if requested
        if normalize:
            logger.info("Starting normalization process...")
            for k in list(sanitized_metrics.keys()):
                try:
                    logger.debug(f"Normalizing metric: {k}")
                    sanitized_metrics[k] = minmax_normalize(sanitized_metrics[k])
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Normalization failed for {k}: {e}")
                    logger.debug(traceback.format_exc())
                    # Keep original values if normalization fails
        
        logger.info("Sanitization and normalization completed successfully")
        return sanitized_metrics
        
    except Exception as e:
        logger.error(f"Critical error in sanitize and normalize: {e}")
        logger.debug(traceback.format_exc())
        # Return original metrics if sanitization fails completely
        return metrics

def safe_save_files(metrics, out_dir, prefix):
    """
    Safely save metrics to files with error handling
    """
    npz_path = csv_path = None
    
    try:
        logger.info("Starting file saving process...")
        npz_path = os.path.join(out_dir, f"{prefix}_metrics.npz")
        
        try:
            # Save to NPZ with careful handling
            logger.info("Preparing data for NPZ file...")
            safe_metrics = {}
            for k, v in metrics.items():
                try:
                    # Convert to standard numpy array with explicit type checking
                    safe_metrics[k] = safe_convert_to_numpy(v, len(next(iter(metrics.values()))), k)
                    logger.debug(f"Prepared {k} for NPZ with shape {safe_metrics[k].shape}")
                except Exception as e:
                    logger.warning(f"Could not prepare {k} for NPZ: {e}")
                    safe_metrics[k] = np.full(len(next(iter(metrics.values()))), np.nan, dtype=float)
            
            logger.info("Saving to NPZ file...")
            np.savez_compressed(npz_path, **safe_metrics)
            logger.info(f"Successfully saved NPZ file: {npz_path}")
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save NPZ file: {e}")
            logger.debug(traceback.format_exc())
            npz_path = None
            
        try:
            logger.info("Preparing data for CSV file...")
            # Prepare DataFrame with explicit conversion
            df_data = {}
            for k, v in metrics.items():
                try:
                    df_data[k] = safe_convert_to_numpy(v, len(next(iter(metrics.values()))), k)
                except Exception as e:
                    logger.warning(f"Could not prepare {k} for CSV: {e}")
                    df_data[k] = np.full(len(next(iter(metrics.values()))), np.nan, dtype=float)
            
            logger.info("Creating DataFrame...")
            df = pd.DataFrame(df_data)
            df.index.name = "vertex_id"
            
            csv_path = os.path.join(out_dir, f"{prefix}_metrics.csv")
            logger.info(f"Saving to CSV file: {csv_path}")
            # Use 'NaN' as placeholder for missing values to ensure consistent output
            df.to_csv(csv_path, index=True, na_rep='NaN')
            logger.info(f"Successfully saved CSV file: {csv_path}")
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save CSV file: {e}")
            logger.debug(traceback.format_exc())
            csv_path = None
            
    except Exception as e:
        logger.error(f"Unexpected error during file saving: {e}")
        logger.debug(traceback.format_exc())
    
    logger.info("File saving process completed")
    return npz_path, csv_path

def compute_and_save_metrics(G, out_dir=".", prefix="network", normalize=True, nthreads=1, save_files=True):
    """
    Return (metrics_dict, npz_path_or_None, csv_path_or_None)
    metrics_dict: name -> numpy array (len == G.num_vertices())
    """
    try:
        logger.info(f"Starting metrics computation for {prefix}")
        os.makedirs(out_dir, exist_ok=True)
        metrics = {}
        logger.info("Computing metrics for graph with %d vertices, %d edges", G.num_vertices(), G.num_edges())

        # Use only 1 thread to prevent memory issues
        with gt.openmp_context(nthreads=nthreads):
            # Warn about potentially unsuitable metrics based on topology
            metrics_to_compute = ['pagerank', 'betweenness', 'closeness', 'eigenvector', 'katz', 'hits_authority', 'hits_hub', 'eigentrust', 'trust_transitivity']
            warn_about_unsuitable_metrics(G, metrics_to_compute)
            # PageRank - Most stable metric
            try:
                d = G.degree_property_map("total")
                periphery = d.a <= 2
                p = G.new_vertex_property("double")
                p.a[periphery] = 100
                metrics['pagerank'] = safe_compute_metric(
                    lambda G, **kw: gt.pagerank(G, max_iter=1000),
                    'pagerank', gt.GraphView(G, vfilt=gt.label_largest_component(G))
                )
            except Exception as e:
                logger.warning(f"Pagerank computation failed: {e}")
                metrics['pagerank'] = np.full(G.num_vertices(), np.nan)

            # # Degree - Very stable
            # try:
            #     metrics['degree'] = safe_compute_metric(
            #         lambda G, **kw: np.array([v.out_degree() + v.in_degree() for v in G.vertices()]),
            #         'degree', G,
            #         fallback_value=np.full(G.num_vertices(), np.nan)
            #     )
            # except Exception as e:
            #     logger.warning(f"Degree computation failed: {e}")
            #     metrics['degree'] = np.full(G.num_vertices(), np.nan)

            # Betweenness - Can be memory intensive
            try:
                metrics['betweenness'] = safe_compute_metric(
                    lambda G, **kw: sanitize_array(gt.betweenness(G, norm=True)[0]),
                    'betweenness', gt.GraphView(G, vfilt=gt.label_largest_component(G))
                )
            except Exception as e:
                logger.warning(f"Betweenness computation failed: {e}")
                metrics['betweenness'] = np.full(G.num_vertices(), np.nan)

            # Closeness - Generally stable
            try:
                metrics['closeness'] = safe_compute_metric(
                    lambda G, **kw: gt.closeness(G),
                    'closeness', gt.GraphView(G, vfilt=gt.label_largest_component(G))
                )
            except Exception as e:
                logger.warning(f"Closeness computation failed: {e}")
                metrics['closeness'] = np.full(G.num_vertices(), np.nan)

            # Eigenvector - Sometimes unstable
            try:
                w = G.new_edge_property("double")
                w.a = np.random.random(len(w.a)) * 42
                metrics['eigenvector'] = safe_compute_metric(
                    lambda G, **kw: sanitize_array(gt.eigenvector(G)[1]),
                    'eigenvector', gt.GraphView(G, vfilt=gt.label_largest_component(G)), w
                )
            except Exception as e:
                logger.warning(f"Eigenvector computation failed: {e}")
                metrics['eigenvector'] = np.full(G.num_vertices(), np.nan)

            # Katz - May have convergence issues
            try:
                w = G.new_edge_property("double")
                w.a = np.random.random(len(w.a))
                metrics['katz'] = safe_compute_metric(
                    lambda G, **kw: gt.katz(G),
                    'katz', gt.GraphView(G, vfilt=gt.label_largest_component(G)), w
                )
            except Exception as e:
                logger.warning(f"Katz computation failed: {e}")
                metrics['katz'] = np.full(G.num_vertices(), np.nan)

            # HITS - Enhanced error handling with better fallback
            try:
                # Try HITS computation with multiple fallback strategies
                largest_component_view = gt.GraphView(G, vfilt=gt.label_largest_component(G))
                if largest_component_view.num_vertices() > 0:
                    hits_result = gt.hits(largest_component_view)
                    metrics['hits_authority'] = sanitize_array(hits_result[1])
                    metrics['hits_hub'] = sanitize_array(hits_result[2])
                    logger.info("HITS computation successful")
                else:
                    logger.warning("HITS: Largest component is empty, using NaN fallback")
                    metrics['hits_authority'] = np.full(G.num_vertices(), np.nan)
                    metrics['hits_hub'] = np.full(G.num_vertices(), np.nan)
            except Exception as e:
                logger.warning(f"HITS computation failed: {e}")
                # Try alternative approach for directed graphs
                try:
                    if G.is_directed() and G.num_vertices() > 1:
                        # Create a small subgraph to test HITS
                        test_vertices = min(100, G.num_vertices())
                        vertex_subset = G.new_vertex_property("bool")
                        count = 0
                        for v in G.vertices():
                            if count < test_vertices:
                                vertex_subset[v] = True
                                count += 1
                            else:
                                break
                        test_graph = gt.GraphView(G, vfilt=vertex_subset)
                        if test_graph.num_vertices() > 1:
                            hits_test = gt.hits(test_graph)
                            logger.info("HITS works on subset, but failed on full graph")
                    
                    metrics['hits_authority'] = np.full(G.num_vertices(), np.nan)
                    metrics['hits_hub'] = np.full(G.num_vertices(), np.nan)
                except Exception as e2:
                    logger.warning(f"HITS fallback also failed: {e2}")
                    metrics['hits_authority'] = np.full(G.num_vertices(), np.nan)
                    metrics['hits_hub'] = np.full(G.num_vertices(), np.nan)

            # Eigentrust - Added with improved error handling
            try:
                w = G.new_edge_property("double")
                w.a = np.random.random(len(w.a)) * 42
                metrics['eigentrust'] = safe_compute_metric(
                    lambda G, **kw: sanitize_array(gt.eigentrust(G, G.edge_index)),
                    'eigentrust', gt.GraphView(G, vfilt=gt.label_largest_component(G)), w
                )
            except Exception as e:
                logger.warning(f"Eigentrust computation failed: {e}")
                metrics['eigentrust'] = np.full(G.num_vertices(), np.nan)
            
            # Trust transitivity - Fixed source vertex issue
            try:
                w = G.new_edge_property("double")
                w.a = np.random.random(len(w.a)) * 42
                largest_component_view = gt.GraphView(G, vfilt=gt.label_largest_component(G))
                if largest_component_view.num_vertices() > 0:
                    # Use a valid source vertex from the largest component
                    source_vertex = None
                    for v in largest_component_view.vertices():
                        source_vertex = v
                        break
                    
                    if source_vertex is not None:
                        metrics['trust_transitivity'] = safe_compute_metric(
                            lambda G, **kw: sanitize_array(gt.trust_transitivity(G, G.edge_index, source=source_vertex)),
                            'trust_transitivity', largest_component_view, w
                        )
                    else:
                        metrics['trust_transitivity'] = np.full(G.num_vertices(), np.nan)
                else:
                    metrics['trust_transitivity'] = np.full(G.num_vertices(), np.nan)
            except Exception as e:
                logger.warning(f"Trust transitivity computation failed: {e}")
                metrics['trust_transitivity'] = np.full(G.num_vertices(), np.nan)


        # Sanitize lengths and normalize with error handling
        logger.info("Sanitizing and normalizing metrics...")
        try:
            metrics = safe_sanitize_and_normalize(metrics, G, normalize)
        except Exception as e:
            logger.error(f"Failed during sanitization/normalization: {e}")
            logger.debug(traceback.format_exc())

        npz_path = csv_path = None
        if save_files:
            logger.info("Saving metrics to files...")
            try:
                npz_path, csv_path = safe_save_files(metrics, out_dir, prefix)
            except Exception as e:
                logger.error(f"Failed during file saving: {e}")
                logger.debug(traceback.format_exc())

        # Explicit cleanup
        logger.info("Performing cleanup...")
        try:
            # Don't delete the metrics dict before returning it
            pass
        except Exception as e:
            logger.warning(f"Error during metrics deletion: {e}")
        gc.collect()
        
        logger.info("Metrics computation completed successfully")
        return metrics, npz_path, csv_path  # Return the actual metrics dict instead of {}
    except Exception as e:
        logger.error(f"Critical error in compute_and_save_metrics: {e}")
        logger.debug(traceback.format_exc())
        gc.collect()
        return {}, None, None
