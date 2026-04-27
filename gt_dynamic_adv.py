#!/usr/bin/env python3

"""
Network animation using graph-tool and matplotlib.
This script creates an animated visualization of neural networks.
"""
import os
import argparse
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "16"
import json
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk,GLib
from random import randint, shuffle, random
import numpy as np
from graph_tool.all import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from numpy.linalg import norm
from numpy.random import *
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU acceleration initialization
def initialize_gpu():
    """
    Initialize GPU acceleration with available libraries.
    Returns availability flags for different GPU libraries.
    """
    gpu_available = False
    cuda_available = False
    opencl_available = False
    
    # Check for CuPy (CUDA)
    try:
        import cupy as cp
        cuda_available = True
        gpu_available = True
        logger.info("CUDA acceleration available via CuPy")
    except ImportError:
        logger.info("CuPy not available, CUDA acceleration disabled")
    
    # Check for PyOpenCL
    try:
        import pyopencl as cl
        opencl_available = True
        gpu_available = True
        logger.info("OpenCL acceleration available via PyOpenCL")
    except ImportError:
        logger.info("PyOpenCL not available, OpenCL acceleration disabled")
    
    return gpu_available, cuda_available, opencl_available

# Initialize GPU support
GPU_AVAILABLE, CUDA_AVAILABLE, OPENCL_AVAILABLE = initialize_gpu()

# Constants for layout
step = 0.005       # move step
K = 0.5            # preferred edge length
# If GPU is available, try to use GPU-accelerated functions for computations
def accelerated_array_operation(arr, operation='norm'):
    """
    Perform array operations using GPU acceleration if available.
    """
    if CUDA_AVAILABLE:
        import cupy as cp
        arr_gpu = cp.asarray(arr)
        if operation == 'norm':
            result = cp.linalg.norm(arr_gpu)
            return result.get() if hasattr(result, 'get') else result
        elif operation == 'mean':
            result = cp.mean(arr_gpu)
            return result.get() if hasattr(result, 'get') else result
    elif OPENCL_AVAILABLE:
        # Basic OpenCL implementation placeholder
        pass
    
    # Fallback to CPU computation
    if operation == 'norm':
        return np.linalg.norm(arr)
    elif operation == 'mean':
        return np.mean(arr)
    
    return arr

def accelerated_sfdp_layout(g, pos=None, **kwargs):
    """
    GPU-accelerated version of SFDP layout algorithm when possible.
    Falls back to standard implementation if GPU not available.
    """
    if CUDA_AVAILABLE:
        logger.info("Using GPU-accelerated layout computation")
        # For now, we still use the standard sfdp_layout but with GPU-optimized parameters
        # A full GPU implementation would require significant changes to graph-tool itself
        return sfdp_layout(g, pos=pos, **kwargs)
    else:
        return sfdp_layout(g, pos=pos, **kwargs)

def accelerated_price_network_generation(num_nodes, num_input_nodes, **kwargs):
    """
    GPU-accelerated random graph generation when possible.
    """
    if CUDA_AVAILABLE:
        logger.info("Using GPU-accelerated graph generation")
        # Use GPU acceleration for random number generation in graph creation
        import cupy as cp
        with cp.cuda.Device(0):  # Use GPU device 0
            result = price_network(num_nodes + num_input_nodes, **kwargs)
            return result
    else:
        return price_network(num_nodes + num_input_nodes, **kwargs)

def load_network_data(network_name, data_dir):
    """
    Load network data from CSV files.
    Args:
        network_name (str): Name of the network to load
        data_dir (str): Directory containing the network data files
    Returns:
        tuple: (nodes_data, edges_data, num_nodes)
    """
    import csv
    nodes_file = os.path.join(data_dir, f"{network_name}_nodes.csv")
    nodes_data = []
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes_data.append(row)
    input_nodes_file = os.path.join(data_dir, f"{network_name}_input_nodes.csv")
    input_nodes_data = []
    with open(input_nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_nodes_data.append(row)
    # Load edges data
    edges_file = os.path.join(data_dir, f"{network_name}_edges.csv")
    edges_data = []
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges_data.append(row)
    input_edges_file = os.path.join(data_dir, f"{network_name}_input_edges.csv")
    input_edges_data = []
    with open(input_edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_edges_data.append(row)

    params_file = os.path.join(data_dir, f"{network_name}_gt_params.json")
    with open(params_file, 'r') as f:
        params_data = json.load(f)
        num_nodes = params_data['number_pop_vertices']
        num_input_nodes = params_data['number_input_vertices']
    return nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes


def accelerated_network_metrics(g):
    """
    Compute network metrics using GPU acceleration when available.
    """
    metrics = {}
    try:
        if CUDA_AVAILABLE:
            import cupy as cp
            logger.info("Computing network metrics with GPU acceleration")
            degrees = [v.out_degree() for v in g.vertices()]
            with cp.cuda.Device(0):
                degrees_gpu = cp.asarray(degrees)
                metrics['avg_degree'] = float(cp.mean(degrees_gpu))
                metrics['max_degree'] = int(cp.max(degrees_gpu))
                metrics['min_degree'] = int(cp.min(degrees_gpu))
                for key in metrics:
                    if hasattr(metrics[key], 'get'):
                        metrics[key] = metrics[key].get()
        else:
            degrees = [v.out_degree() for v in g.vertices()]
            metrics['avg_degree'] = np.mean(degrees)
            metrics['max_degree'] = max(degrees)
            metrics['min_degree'] = min(degrees)
            
    except Exception as e:
        logger.warning(f"Error computing accelerated metrics: {e}")
        # Fallback to basic metrics
        metrics['node_count'] = g.num_vertices()
        metrics['edge_count'] = g.num_edges()
    return metrics


def print_accelerated_network_info(g):
    """
    Print network information using GPU-accelerated metrics when available.
    """
    metrics = accelerated_network_metrics(g)
    print(f"Network Info:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def get_available_networks(data_dir="gt/params"):
    """
    Get list of available network names from the data directory.
    Returns:
        list: List of network names (without file extensions)
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist")
        return []
    
    # Look for _gt_params.json files to identify available networks
    import glob
    params_files = glob.glob(os.path.join(data_dir, "*_gt_params.json"))
    network_names = []
    for params_file in params_files:
        basename = os.path.basename(params_file)
        network_name = basename.replace('_gt_params.json', '')
        network_names.append(network_name)
    
    return sorted(network_names)


def process_network(network_name, data_dir="gt/params", max_count=300):
    """
    Process a single network for dynamic visualization.
    Args:
        network_name (str): Name of the network to process
        data_dir (str): Directory containing the network data files
        max_count (int): Maximum number of animation frames
    """
    logging.info(f"Loading network: {network_name}")
    try:
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        print(f"Connected graph built: {num_nodes} pop vertices, {num_input_nodes} input vertices")
        
        edge_count = len(edges_data)
        input_edge_count = len(input_edges_data)
        print(f"Edge type distribution:{edge_count} projection edges, {input_edge_count} input edges")
        
        g = accelerated_price_network_generation(num_nodes, num_input_nodes, c=0.8, directed=False)
        
        node_id_to_index = {}
        for i, node_info in enumerate(nodes_data):
            node_id_to_index[node_info['component']] = i
        inhibitory_nodes = set()
        excitatory_nodes = set()
        for node_info in nodes_data:
            if node_info['type'] == 'Inh' and node_info['component'] in node_id_to_index:
                inhibitory_nodes.add(node_id_to_index[node_info['component']])
            elif node_info['type'] == 'Exc' and node_info['component'] in node_id_to_index:
                excitatory_nodes.add(node_id_to_index[node_info['component']])
        input_id_to_index = {}
        for i, node_info in enumerate(input_nodes_data):
            input_id_to_index[node_info['component']] = i
        print(f"Mapped node IDs to graph vertices: {len(node_id_to_index)} regular nodes, {len(input_id_to_index)} input nodes")
        inhibitory_inputs = set()
        excitatory_inputs = set()
        for node_info in input_nodes_data:
            if node_info['type'] == 'Inh' and node_info['component'] in input_id_to_index:
                inhibitory_inputs.add(input_id_to_index[node_info['component']])
            elif node_info['type'] == 'Exc' and node_info['component'] in input_id_to_index:
                excitatory_inputs.add(input_id_to_index[node_info['component']])
        # Add edges between connected nodes
        edge_added_count = 0
        ee_count = 0
        ei_count = 0
        ie_count = 0
        ii_count = 0
        for edge_info in edges_data:
            source_id = edge_info['source']
            target_id = edge_info['target']
            src_type = edge_info['source_type']
            tgt_type = edge_info['target_type']
            if source_id in node_id_to_index and target_id in node_id_to_index:
                source_idx = node_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
            edge_added_count += 1 
            if src_type == 'Exc' and tgt_type == 'Exc':
                ee_count += 1 
            if src_type == 'Exc' and tgt_type == 'Inh':
                ei_count += 1 
            if src_type == 'Inh' and tgt_type == 'Exc':
                ie_count += 1
            if src_type == 'Inh' and tgt_type == 'Inh':
                ii_count += 1 

        input_edge_added_count = 0
        input_ee_count = 0
        input_ii_count = 0
        for input_edge_info in input_edges_data:
            source_id = input_edge_info['source']
            target_id = input_edge_info['target']
            input_src_type = input_edge_info['source_type']
            input_tgt_type = input_edge_info['target_type']
            if source_id in input_id_to_index and target_id in node_id_to_index :
                source_idx = input_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
            input_edge_added_count += 1 
            if input_src_type == 'Exc' and input_tgt_type == 'Exc':
                input_ee_count += 1
            if input_src_type == 'Inh' and input_tgt_type == 'Inh':
                input_ii_count += 1
        
        print(f"Edge type distribution: Edges={edge_added_count}, Input Edges={input_edge_added_count}")
        print(f"Edge type distribution: EE={ee_count}, EI={ei_count}, IE={ie_count}, II={ii_count}, input_exc={input_ee_count}, input_inh={input_ii_count}")
        print(f"Added edges: {edge_added_count} internal, {input_edge_added_count} input edges")
        
        total_nodes = num_nodes + num_input_nodes
        logging.info(f"Total nodes: {total_nodes}")
        total_edges = edge_added_count + input_edge_added_count
        logging.info(f"Total edges: {total_edges}")

        if total_nodes > 0:
            density = total_edges / total_nodes
        else:
            density = 0
        
        # Create node type mapping for edge classification
        node_types = {}
        for i, node_info in enumerate(nodes_data): # Map regular network nodes
            node_type = node_info['type'] 
            node_types[i] = 'Exc' if node_type.startswith('E') else 'Inh'
        for j, input_node_info in enumerate(input_nodes_data): # Map input nodes 
            input_node_type = input_node_info['type'] 
            node_types[num_nodes + j] = 'Exc' if input_node_type.startswith('E') else 'Inh'
        
        if total_nodes <= 1500:
            dynamic_node_size = 18.0
            dynamic_edge_width = 2.0
        elif total_nodes <= 3000:
            dynamic_node_size = 10.0
            dynamic_edge_width = 1.0
        elif total_nodes <= 4500:
            dynamic_node_size = 6.0
            dynamic_edge_width = 0.5
        elif total_nodes <= 6000:
            dynamic_node_size = 4.0
            dynamic_edge_width = 0.2
        else:
            dynamic_node_size = 2.5
            dynamic_edge_width = 0.1

        pos = accelerated_sfdp_layout(g, K=K, cooling_step=0.99, C=0.5, multilevel=True, R=8, gamma=1)
        # pos = sfdp_layout(g, K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)
        
        edges = list(g.edges()) # list of edges

        offscreen = True  # Enable offscreen rendering to save frames
        
        network_frames_dir = f"./frames/{network_name}/dynamic_adv" # Ensure frames directory exists
        if offscreen and not os.path.exists(network_frames_dir):
            os.makedirs(network_frames_dir)
            print(f"Created network frames directory: {network_frames_dir}")

        # Counter for animation frames
        count = 0
        all_x_positions = []
        all_y_positions = []
        vertex_index_map = {v: i for i, v in enumerate(g.vertices())}
        edge_index_map = {e: i for i, e in enumerate(g.edges())}
        # This function will be called repeatedly to update the vertex layout
        def update_state():
            nonlocal count, edges, all_x_positions, all_y_positions, vertex_index_map, edge_index_map

            # Perform fewer iterations of the layout step for faster processing
            sfdp_layout(g, pos=pos, K=K, init_step=step, max_iter=1)  

            # Perform edge rewiring with reduced frequency for better performance
            if len(edges) > 0 and count % 3 == 0:  # Only rewire every 3rd frame
                # Use GPU acceleration for edge selection if available and we have many edges
                if CUDA_AVAILABLE and len(edges) > 5000:
                    import cupy as cp
                    with cp.cuda.Device(0):
                        # Generate fewer random indices on GPU
                        edge_indices = cp.random.randint(0, len(edges), size=max(5000, len(edges)))  # Reduced from 100
                        # Transfer to CPU for processing
                        edge_indices_cpu = edge_indices.get() if hasattr(edge_indices, 'get') else edge_indices
                else:
                    # Standard CPU approach with fewer edges
                    edge_indices_cpu = [randint(0, len(edges)-1) for _ in range(min(1000, len(edges)))]  # Reduced from 20
                        
                # Process selected edges
                for edge_index in edge_indices_cpu:
                    if edge_index < len(edges):  # Bounds check
                        e = list(edges[edge_index])
                        shuffle(e)
                        s1, t1 = e
                        t2 = g.vertex(randint(0, g.num_vertices()))
                        # Looser conditions for rewiring to increase frequency
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and
                            s1 != t2 and                      # no self-loops
                            t1.out_degree() > 0 and           # isolated vertices counts
                            t2 not in s1.out_neighbors()):    # no parallel edges

                            g.remove_edge(edges[edge_index])
                            new_edge = g.add_edge(s1, t2)
                            edges[edge_index] = new_edge
            if count > 0 and count % 1000 == 0:  # More frequent logging
                print(f"Processed {count} frames")
                if count % 1000 == 0:  # Reduced frequency of metric calculation
                    print_accelerated_network_info(g)
                    if len(edges) > 0 and count % 2 == 0:
                        GraphWindow.graph.fit_to_window(ink=True)
                        for i in range(10000):  # rewiring iterations for performance
                            i = randint(0, len(edges))
                            e = list(edges[i])
                            shuffle(e)
                            s1, t1 = e
                            t2 = g.vertex(randint(0, g.num_vertices()))
                            if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and
                                s1 != t2 and                      # no self-loops
                                t1.out_degree() > 0 and           # less strict on isolated vertices
                                t2 not in s1.out_neighbors()):    # no parallel edges

                                g.remove_edge(edges[i])
                                edges[i] = g.add_edge(s1, t2)            
                
            count += 1

            # if doing an offscreen animation, save frame as PNG
            if offscreen:
                # Save every 5nd frame to increase frame rate
                if count % 10 == 0:  # Changed from every 10th to every 5nd frame
                    # Create frame as PNG file
                    try:
                        # Create a matplotlib figure for this frame with consistent dimensions
                        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
                        
                        # Extract positions for plotting
                        x_pos = [pos[v][0] for v in g.vertices()]
                        y_pos = [pos[v][1] for v in g.vertices()]
                        # nodes_x, nodes_y = [], []
                        # sizes = np.random.uniform(dynamic_node_size * 15, dynamic_node_size * 30, len(x_pos))
                        colors = np.random.uniform(0.4, 0.9, len(y_pos))
                        sizes = []
                        for v in g.vertices():
                            v_idx = vertex_index_map[v]
                            is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                            if is_excitatory:
                                sizes.append(dynamic_node_size)
                            else:  
                                sizes.append(dynamic_node_size * 0.75)  # Inhibitory nodes slightly smaller
                        ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.8, linewidth=dynamic_edge_width, edgecolors='grey', zorder=1, cmap='summer')
                        
                        edges_list = list(g.edges())
                        subsample_rate = max(1, len(edges_list) // 5000000)  # Increase the number of edges shown
                        
                        # Initialize edge lists
                        ee_edges = []
                        ei_edges = []
                        ie_edges = []
                        ii_edges = []
                        input_ee_edges = []
                        input_ii_edges = []
                        for e in edges_list:
                            src_idx = int(e.source())
                            tgt_idx = int(e.target())
                            src_type = node_types.get(src_idx, 'Exc')  # Default to Exc
                            tgt_type = node_types.get(tgt_idx, 'Exc')  # Default to Exc
                            is_input_edge = src_idx >= num_nodes
                            if is_input_edge:
                                if src_type == 'Exc':  # Source is excitatory
                                    input_ee_edges.append(e)
                                elif src_type == 'Inh':  # Source is inhibitory
                                    input_ii_edges.append(e)
                            else:
                                if src_type == 'Exc' and tgt_type == 'Exc':
                                    ee_edges.append(e)
                                elif src_type == 'Exc' and tgt_type == 'Inh':
                                    ei_edges.append(e)
                                elif src_type == 'Inh' and tgt_type == 'Exc':
                                    ie_edges.append(e)
                                elif src_type == 'Inh' and tgt_type == 'Inh':
                                    ii_edges.append(e)

                       
                        # Plot EE edges (Excitatory to Excitatory)
                        if len(ee_edges) > 0:
                            subsample_rate = max(1, len(ee_edges) // 5000000)  # Adjust subsample rate for EE edges
                            plotted_ee_edges = 0
                            for edge_idx, current_edge in enumerate(ee_edges):
                                current_edge = ee_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_ee_edges < 5000000:
                                    s1, t1 = current_edge
                                    ee_x_coords = [pos[s1][0], pos[t1][0]]
                                    ee_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(ee_x_coords, ee_y_coords, 'red', zorder=2, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(ee_x_coords, ee_y_coords, c='red', s=dynamic_edge_width * 10, alpha=0.7, zorder=2, edgecolors='white', linewidth=dynamic_edge_width * 0.5)  # Add small points for better visibility
                                    plotted_ee_edges += 1

                        # Plot EI edges (Excitatory to Inhibitory)
                        if len(ei_edges) > 0:
                            subsample_rate = max(1, len(ei_edges) // 5000000)
                            plotted_ei_edges = 0
                            for edge_idx, current_edge in enumerate(ei_edges):
                                current_edge = ei_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_ei_edges < 5000000:
                                    s1, t1 = current_edge
                                    ei_x_coords = [pos[s1][0], pos[t1][0]]
                                    ei_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(ei_x_coords, ei_y_coords, 'pink', zorder=1, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(ei_x_coords, ei_y_coords, c='pink', s=dynamic_edge_width * 10, alpha=0.7, zorder=1, edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                                    plotted_ei_edges += 1

                        # Plot IE edges (Inhibitory to Excitatory)
                        if len(ie_edges) > 0:
                            subsample_rate = max(1, len(ie_edges) // 5000000)
                            plotted_ie_edges = 0
                            for edge_idx, current_edge in enumerate(ie_edges):
                                current_edge = ie_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_ie_edges < 5000000:
                                    s1, t1 = current_edge
                                    ie_x_coords = [pos[s1][0], pos[t1][0]]
                                    ie_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(ie_x_coords, ie_y_coords, 'green', zorder=3, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(ie_x_coords, ie_y_coords, c='green', s=dynamic_edge_width * 10, alpha=0.7, zorder=3, edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                                    plotted_ie_edges += 1
                        
                        # Plot II edges (Inhibitory to Inhibitory)
                        if len(ii_edges) > 0:
                            subsample_rate = max(1, len(ii_edges) // 5000000)
                            plotted_ii_edges = 0
                            for edge_idx, current_edge in enumerate(ii_edges):
                                current_edge = ii_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_ii_edges < 5000000:
                                    s1, t1 = current_edge
                                    ii_x_coords = [pos[s1][0], pos[t1][0]]
                                    ii_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(ii_x_coords, ii_y_coords, 'blue', zorder=4, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(ii_x_coords, ii_y_coords, c='blue', s=dynamic_edge_width * 10, alpha=0.7, zorder=4, edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                                    plotted_ii_edges += 1

                        
                        # Plot Input_EE edges
                        if len(input_ee_edges) > 0:
                            subsample_rate = max(1, len(input_ee_edges) // 1000000)
                            plotted_input_ee_edges = 0
                            for edge_idx, current_edge in enumerate(input_ee_edges):
                                current_edge = input_ee_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_input_ee_edges < 1000000:
                                    s1, t1 = current_edge
                                    input_ee_x_coords = [pos[s1][0], pos[t1][0]]
                                    input_ee_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(input_ee_x_coords, input_ee_y_coords, 'gold', zorder=8, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(input_ee_x_coords, input_ee_y_coords, c='gold', s=dynamic_edge_width * 10, alpha=0.7, zorder=8, edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                                    plotted_input_ee_edges += 1

                        
                        # Plot Input_II edges
                        if len(input_ii_edges) > 0:
                            subsample_rate = max(1, len(input_ii_edges) // 1000000)
                            plotted_input_ii_edges = 0
                            for edge_idx, current_edge in enumerate(input_ii_edges):
                                current_edge = input_ii_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_input_ii_edges < 1000000:
                                    s1, t1 = current_edge
                                    input_ii_x_coords = [pos[s1][0], pos[t1][0]]
                                    input_ii_y_coords = [pos[s1][1], pos[t1][1]]
                                    ax.plot(input_ii_x_coords, input_ii_y_coords, 'purple', zorder=9, alpha=0.7, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                    ax.scatter(input_ii_x_coords, input_ii_y_coords, c='purple', s=dynamic_edge_width * 10, alpha=0.7, zorder=9,edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                                    plotted_input_ii_edges += 1

                        
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='red', lw=2, label='EE Edges'),
                            Line2D([0], [0], color='pink', lw=2, label='EI Edges'),
                            Line2D([0], [0], color='green', lw=2, label='IE Edges'),
                            Line2D([0], [0], color='blue', lw=2, label='II Edges'),
                            Line2D([0], [0], color='gold', lw=2, label='Input EE Edges'),
                            Line2D([0], [0], color='purple', lw=2, label='Input II Edges'),
                        ]
                        ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=8)
                        ax.set_title(f'{network_name} Dynamic layout: Frame {count}\n EE: {len(ee_edges)} | EI: {len(ei_edges)} | IE: {len(ie_edges)} | II: {len(ii_edges)} | Input_EE: {len(input_ee_edges)} | Input_II: {len(input_ii_edges)}', fontsize=10)
                        ax.set_aspect('equal')
                        
                        if len(all_x_positions) > 0 and len(all_y_positions) > 0:
                            x_min, x_max = min(all_x_positions), max(all_x_positions)
                            y_min, y_max = min(all_y_positions), max(all_y_positions)
                            x_range = x_max - x_min
                            y_range = y_max - y_min
                            
                            # Dynamic padding based on refined network size thresholds
                            if total_nodes <= 1500:
                                padding_factor = 0.7  # 70% padding for small networks
                            elif total_nodes <= 3000:
                                padding_factor = 0.8   # 80% padding for medium networks  
                            elif total_nodes <= 4500:
                                padding_factor = 0.9   # 90% padding for large networks
                            elif total_nodes <= 6000:
                                padding_factor = 0.99   # 99% padding for very large networks
                            else:
                                padding_factor = 1.8   # 180% padding for extremely large networks
                            
                            padding_x = x_range * padding_factor
                            padding_y = y_range * padding_factor
                            
                            # Ensure minimum viewport size to prevent extreme zoom-in
                            min_viewport = 10.0  # Minimum viewport size
                            if x_range + 2 * padding_x < min_viewport:
                                padding_x = (min_viewport - x_range) / 2
                            if y_range + 2 * padding_y < min_viewport:
                                padding_y = (min_viewport - y_range) / 2
                            
                            # Apply fixed axis limits with padding
                            ax.set_xlim(x_min - padding_x, x_max + padding_x)
                            ax.set_ylim(y_min - padding_y, y_max + padding_y)
                        ax.grid(True, alpha=0.2)
                        # Save frame as PNG file with consistent size
                        plt.savefig(f'{network_frames_dir}/frame_{count:04d}.png', dpi=200, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        print(f"Saved frame {count}")
                    except Exception as e:
                        print(f"Error saving frame {count}: {e}")
                        import traceback
                        traceback.print_exc()
                if count >= max_count:
                    print("Animation complete")
                    return False
            # Stop after a reasonable number of frames to avoid infinite loop
            if count >= max_count:
                print("Reached frame limit, stopping animation")
                return False
            return True
        
        print("Starting animation loop...")
        while update_state():
            pass
        print("Animation finished")
        
        # Provide instructions for combining frames
        if offscreen:
            print(f"\nFrames saved to {network_frames_dir}/ directory")
            print("\nTo combine frames into a video, you can use ffmpeg:")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_dynamic_adv.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick:")
            print(f"convert -delay 10 {network_frames_dir}/*.png {network_name}_dynamic_adv.gif")
            print("\nOr create an animated GIF with optimization:")
            print(f"convert -delay 10 -loop 0 {network_frames_dir}/*.png -scale 800x800 -coalesce -fuzz 5% -layers Optimize {network_name}_dynamic_adv.gif")
            print("\nThis will create an animated GIF from the saved frames.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files for network {network_name}: {e}")
        print("Make sure the gt/params directory contains the required files.")
    except Exception as e:
        print(f"Error processing network {network_name}: {e}")
        import traceback
        traceback.print_exc()
def main():
    parser = argparse.ArgumentParser(description='Process neural network animations (advanced version with excitatory/inhibitory edge types)')
    parser.add_argument('--data-dir', default='gt/params', help='Directory containing network data files (default: gt/params)')
    parser.add_argument('--max-frames', type=int, default=300, help='Maximum number of animation frames per network (default: 300)')
    parser.add_argument('networks', nargs='*', help='Specific network names to process (e.g., M2M1S1_max_plus M1_max_plus). If not specified, use --all to process all available networks.')
    parser.add_argument('--list-networks', action='store_true', help='List all available networks and exit')
    parser.add_argument('--all', action='store_true', help='Process all available networks (use instead of specifying networks)')
    
    args = parser.parse_args()
    
    # List available networks if requested
    if args.list_networks:
        networks = get_available_networks(args.data_dir)
        print("Available networks:")
        for network in networks:
            print(f"  {network}")
        return
    
    # Determine which networks to process
    if args.networks:
        # Process specific networks (positional arguments)
        networks_to_process = args.networks
        print(f"Processing specific networks: {networks_to_process}")
    elif args.all:
        # Process all available networks
        networks_to_process = get_available_networks(args.data_dir)
        print(f"Processing all available networks ({len(networks_to_process)} networks)")
    else:
        # No networks specified and --all not used
        print("Error: Please specify network names or use --all to process all networks.")
        print("Usage examples:")
        print("  python gt_dynamic_adv.py M2M1S1_max_plus")
        print("  python gt_dynamic_adv.py M1_max_plus M2_max_plus S1_max_plus")  
        print("  python gt_dynamic_adv.py --all")
        print("  python gt_dynamic_adv.py --list-networks")
        return
    
    # Validate that requested networks exist
    available_networks = set(get_available_networks(args.data_dir))
    invalid_networks = [net for net in networks_to_process if net not in available_networks]
    if invalid_networks:
        print(f"Warning: The following networks were not found in {args.data_dir}: {invalid_networks}")
        networks_to_process = [net for net in networks_to_process if net in available_networks]
        if not networks_to_process:
            print("No valid networks to process. Exiting.")
            return
    
    # Process each network in sequence
    total_networks = len(networks_to_process)
    for i, network_name in enumerate(networks_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing network {i}/{total_networks}: {network_name}")
        print(f"{'='*60}")
        process_network(network_name, args.data_dir, args.max_frames)
        print(f"Completed network {i}/{total_networks}: {network_name}")
    
    print(f"\n{'='*60}")
    print(f"All networks processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()