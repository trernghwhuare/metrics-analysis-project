"""
Network Metrics Generator
Author: Hua Cheng <trernghwhuare@aliyun.com>
"""

import os
import logging
import numpy as np
import pandas as pd
import graph_tool.all as gt
from .utils import sanitize_array, minmax_normalize

logger = logging.getLogger(__name__)

def _metric_per_component_mapped(G, metric_callable):
    """
    Compute metric_callable on every connected component (subgraph) and map results
    back to a full-graph numpy array (NaN for vertices where metric fails).
    """
    # component labels (integer per vertex)
    comp_map = gt.label_components(G)[0].get_array()
    uniq, counts = np.unique(comp_map, return_counts=True)
    arr = np.full(G.num_vertices(), np.nan, dtype=float)

    # pre-create a reusable boolean vertex property
    vfilt_prop = G.new_vertex_property("bool")

    for comp_label in uniq:
        # select vertices in this component
        vids = np.where(comp_map == comp_label)[0]
        if vids.size == 0:
            continue
        # mark vfilt_prop True for vertices in this component
        vfilt_prop.a[:] = False
        vfilt_prop.a[vids] = True
        G_sub = gt.GraphView(G, vfilt=vfilt_prop)

        # prepare an edge-weight property for subgraph (size matches edges in subview)
        try:
            ew = G_sub.new_edge_property("double")
            ew.a = np.random.random(ew.a.shape)
        except Exception:
            ew = G.new_edge_property("double")
            ew.a = np.random.random(ew.a.shape)

        # call metric on subgraph; handle call signatures
        try:
            res = metric_callable(G_sub, ew)
        except TypeError:
            try:
                res = metric_callable(G_sub)
            except Exception:
                res = None
        except Exception:
            res = None

        if res is None:
            # leave NaNs for this component
            continue

        # pick a vertex_property if returned in a tuple
        if isinstance(res, tuple):
            for item in reversed(res):
                if hasattr(item, "a"):
                    res = item
                    break

        # Convert res to numpy array safely
        res_array = None
        if hasattr(res, "a"):
            res_array_attr = getattr(res, 'a', None)
            if res_array_attr is not None:
                try:
                    res_array = np.asarray(res_array_attr, dtype=float)
                except (TypeError, ValueError):
                    res_array = None
        elif hasattr(res, "get_array"):
            try:
                res_array = sanitize_array(res)
            except (TypeError, ValueError):
                res_array = None
        
        if res_array is not None and len(res_array) > 0:
            for i, v in enumerate(G_sub.vertices()):
                if i < len(res_array):
                    arr[int(v)] = float(res_array[i])
        else:
            # Fallback: try direct conversion
            try:
                tmp = np.asarray(res, dtype=float)
                idxs = [int(v) for v in G_sub.vertices()]
                for i, vid in enumerate(idxs):
                    if i < tmp.size:
                        arr[vid] = float(tmp[i])
            except (TypeError, ValueError, IndexError):
                # If conversion fails, leave as NaN (already initialized)
                pass

    return arr

def compute_and_save_metrics(G, out_dir=".", prefix="network", normalize=True, nthreads=8, save_files=True, use_undirected_for_eigenvector=False):
    """
    Return (metrics_dict, npz_path_or_None, csv_path_or_None)
    metrics_dict: name -> numpy array (len == G.num_vertices())
    
    Parameters:
    - use_undirected_for_eigenvector: If True, creates undirected version of graph 
      for eigenvector-based metrics (eigenvector, katz, eigentrust, trust_transitivity)
      to handle sparse directed networks with no strongly connected components.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = {}
    logger.info("Computing metrics for graph with %d vertices, %d edges", G.num_vertices(), G.num_edges())

    # Set number of threads for OpenMP (compatible with graph-tool 2.59+)
    original_threads = gt.openmp_get_num_threads()
    gt.openmp_set_num_threads(nthreads)
    
    try:
        try:
            metrics['pagerank'] = sanitize_array(gt.pagerank(G))
        except Exception as e:
            logger.warning("pagerank failed: %s", e)
            metrics['pagerank'] = np.full(G.num_vertices(), np.nan)

        try:
            btw_vp, _ = gt.betweenness(G)
            metrics['betweenness'] = sanitize_array(btw_vp)
        except Exception as e:
            logger.warning("betweenness failed: %s", e)
            metrics['betweenness'] = np.full(G.num_vertices(), np.nan)

        try:
            cl = gt.closeness(G, harmonic=True)
            metrics['closeness'] = sanitize_array(cl)
        except Exception as e:
            logger.warning("closeness failed: %s", e)
            metrics['closeness'] = np.full(G.num_vertices(), np.nan)

        # Validate all metrics have correct length
        expected_length = G.num_vertices()
        for key, arr in list(metrics.items()):
            if len(arr) != expected_length:
                logger.warning(f"Metric '{key}' has length {len(arr)}, expected {expected_length}. Fixing...")
                if len(arr) < expected_length:
                    fixed_arr = np.full(expected_length, np.nan)
                    fixed_arr[:len(arr)] = arr
                    metrics[key] = fixed_arr
                else:
                    metrics[key] = arr[:expected_length]

        # Create undirected graph if needed for eigenvector-based metrics
        G_undirected = None
        if use_undirected_for_eigenvector:
            G_undirected = G.copy()
            G_undirected.set_directed(False)
            logger.info("Created undirected version for eigenvector-based metrics")

        try:
            if use_undirected_for_eigenvector and G_undirected is not None:
                metrics['eigenvector'] = _metric_per_component_mapped(G_undirected, lambda g, w=None: gt.eigenvector(g, w if w is not None else None))
            else:
                metrics['eigenvector'] = _metric_per_component_mapped(G, lambda g, w=None: gt.eigenvector(g, w if w is not None else None))
        except Exception as e:
            logger.warning("eigenvector failed: %s", e)
            metrics['eigenvector'] = np.full(G.num_vertices(), np.nan)

        try:
            if use_undirected_for_eigenvector and G_undirected is not None:
                metrics['katz'] = _metric_per_component_mapped(G_undirected, lambda g, w=None: gt.katz(g, weight=w if w is not None else None))
            else:
                metrics['katz'] = _metric_per_component_mapped(G, lambda g, w=None: gt.katz(g, weight=w if w is not None else None))
        except Exception as e:
            logger.warning("katz failed: %s", e)
            metrics['katz'] = np.full(G.num_vertices(), np.nan)

        try:
            h_res = gt.hits(G)
            if isinstance(h_res, tuple):
                auth = next((x for x in h_res if hasattr(x, "a")), None)
                hub = next((x for x in reversed(h_res) if hasattr(x, "a")), None)
                metrics['hits_authority'] = sanitize_array(auth.get_array()) if auth is not None else np.full(G.num_vertices(), np.nan)
                metrics['hits_hub'] = sanitize_array(hub.get_array()) if hub is not None else np.full(G.num_vertices(), np.nan)
            else:
                tmp = sanitize_array(np.asarray(h_res))
                metrics['hits_authority'] = tmp
                metrics['hits_hub'] = tmp
        except Exception as e:
            logger.warning("HITS failed: %s", e)
            metrics['hits_authority'] = np.full(G.num_vertices(), np.nan)
            metrics['hits_hub'] = np.full(G.num_vertices(), np.nan)

        try:
            if use_undirected_for_eigenvector and G_undirected is not None:
                metrics['eigentrust'] = _metric_per_component_mapped(G_undirected, lambda g, w=None: gt.eigentrust(g, w if w is not None else None))
            else:
                metrics['eigentrust'] = _metric_per_component_mapped(G, lambda g, w=None: gt.eigentrust(g, w if w is not None else None))
        except Exception as e:
            logger.warning("eigentrust failed: %s", e)
            metrics['eigentrust'] = np.full(G.num_vertices(), np.nan)

        def trust_call(g, w=None):
            vs = list(g.vertices())
            if not vs:
                return g.new_vertex_property("double")
            try:
                return gt.trust_transitivity(g, w if w is not None else None, source=vs[0])
            except TypeError:
                return gt.trust_transitivity(g, w if w is not None else None)

        try:
            if use_undirected_for_eigenvector and G_undirected is not None:
                metrics['trust_transitivity'] = _metric_per_component_mapped(G_undirected, trust_call)
            else:
                metrics['trust_transitivity'] = _metric_per_component_mapped(G, trust_call)
        except Exception as e:
            logger.warning("trust_transitivity failed: %s", e)
            metrics['trust_transitivity'] = np.full(G.num_vertices(), np.nan)

    finally:
        gt.openmp_set_num_threads(original_threads)

    # sanitize lengths and normalize
    for k, v in list(metrics.items()):
        metrics[k] = sanitize_array(v)
        if len(metrics[k]) != G.num_vertices():
            arr = np.full(G.num_vertices(), np.nan)
            arr[:min(len(metrics[k]), G.num_vertices())] = metrics[k][:min(len(metrics[k]), G.num_vertices())]
            metrics[k] = arr

    if normalize:
        for k in list(metrics.keys()):
            metrics[k] = minmax_normalize(metrics[k])

    npz_path = csv_path = None
    if save_files:
        npz_path = os.path.join(out_dir, f"{prefix}_metrics.npz")
        np.savez_compressed(npz_path, **metrics)
        df = pd.DataFrame(metrics)
        df.index.name = "vertex_id"
        csv_path = os.path.join(out_dir, f"{prefix}_metrics.csv")
        df.to_csv(csv_path, index=True)
        logger.info("Saved metrics to %s and %s", npz_path, csv_path)

    return metrics, npz_path, csv_path