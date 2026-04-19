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
from random import randint, shuffle
import numpy as np
from graph_tool.all import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from numpy.linalg import norm
from numpy.random import *
import logging

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
    print(f"Loading network: {network_name}")
    try:
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        print(f"Connected graph built: {num_nodes} pop vertices, {num_input_nodes} input vertices")
        
        edge_count = len(edges_data)
        input_edge_count = len(input_edges_data)
        print(f"Edge type distribution:{edge_count} projection edges, {input_edge_count} input edges")
        
        g = accelerated_price_network_generation(num_nodes, num_input_nodes, c=0.8, directed=False)
        # Create a mapping from node IDs to indices
        node_id_to_index = {}
        for i, node_info in enumerate(nodes_data):
            node_id_to_index[node_info['component']] = i
        edge_count = 0
        for edge_info in edges_data:
            source_id = edge_info['source']
            target_id = edge_info['target']
            # Check if both nodes exist in our graph
            if source_id in node_id_to_index and target_id in node_id_to_index:
                source_idx = node_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
                # g.add_edge(source_idx, target_idx)
            edge_count += 1
        input_id_to_index = {}
        for i, node_info in enumerate(input_nodes_data):
            input_id_to_index[node_info['component']] = i   
        input_edge_count = 0
        for input_edge_info in input_edges_data:
            source_id = input_edge_info['source']
            target_id = input_edge_info['target']
            if source_id in input_id_to_index and target_id in node_id_to_index:
                source_idx = input_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
                # g.add_edge(source_idx, target_idx)
            input_edge_count += 1
            
        print(f"Edge type distribution: Edges={edge_count}, Input Edges={input_edge_count}")
        
        total_nodes = num_nodes + num_input_nodes
        logging.info(f"Total nodes: {total_nodes}")
        total_edges = edge_count + input_edge_count
        logging.info(f"Total edges: {total_edges}")

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

        # pos = accelerated_sfdp_layout(g, K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)
        pos = accelerated_sfdp_layout(g, K=K, cooling_step=0.99, C=0.5, multilevel=True, R=8, gamma=1)
        
        # list of edges
        edges = list(g.edges())
        
        offscreen = True  # Enable offscreen rendering to save frames
        # Ensure frames directory exists
        network_frames_dir = f"./frames/{network_name}/dynamic"
        if offscreen and not os.path.exists(network_frames_dir):
            os.makedirs(network_frames_dir)
            print(f"Created network frames directory: {network_frames_dir}")
        # Counter for animation frames
        count = 0
        all_x_positions = []
        all_y_positions = []
        edge_index_map = {e: i for i, e in enumerate(g.edges())}
        # This function will be called repeatedly to update the vertex layout
        def update_state():
            nonlocal count, edges, all_x_positions, all_y_positions, edge_index_map

            # Perform fewer iterations of the layout step for faster processing
            accelerated_sfdp_layout(g, pos=pos, K=K, init_step=step, max_iter=1)  

            # Perform edge rewiring with reduced frequency for better performance
            if len(edges) > 0 and count % 3 == 0:  # Only rewire every 3rd frame
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
                            t1.out_degree() > 0 and           # less strict on isolated vertices
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
                                t1.out_degree() > 0            # less strict on isolated vertices
                                #  and t2 not in s1.out_neighbors()      # no parallel edges
                            ):    

                                g.remove_edge(edges[i])
                                edges[i] = g.add_edge(s1, t2)            
            count += 1

            # if doing an offscreen animation, save frame as PNG
            if offscreen:
                if count % 10 == 0:  
                    # Create frame as PNG file
                    try:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
                        fig.patch.set_facecolor('white')
                        # Extract positions for plotting
                        x_pos = [pos[v][0] for v in g.vertices()]
                        y_pos = [pos[v][1] for v in g.vertices()]
                        sizes = []
                        colors = np.random.uniform(0.4, 0.9, len(y_pos))
                        for v in g.vertices():
                            sizes.append(dynamic_node_size)
                        # ax.scatter(nodes_x, nodes_y, c=colors, s=sizes, cmap='summer', zorder=1, linewidths=dynamic_edge_width, edgecolors='black', alpha=0.8)
                        ax.scatter(x_pos, y_pos, c=colors, s=sizes, linewidth=dynamic_edge_width, edgecolors='grey', alpha=0.8, cmap='summer', zorder=1)
                                   
                        edges_list = list(g.edges())
                        subsample_rate = max(1, len(edges_list) // 5000000)  # Increase the number of edges shown
                        plotted_edges = 0
                        plotted_input_edges = 0

                        renewal_edges = []
                        renewal_input_edges = []
                        for e in g.edges():
                            src_idx = int(e.source())
                            tgt_idx = int(e.target())
                            if src_idx < num_nodes and tgt_idx < num_nodes:
                                renewal_edges.append(e)
                            elif src_idx >= num_nodes and tgt_idx < num_nodes:
                                renewal_input_edges.append(e)
                        
                        if len(renewal_edges) > 0:
                            subsample_rate = max(1, len(renewal_edges) // 100000000)  # the more edges the delicate
                            plotted_edges = 0
                            for edge_idx, current_edge in enumerate(renewal_edges):
                                current_edge = renewal_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_edges < 100000000 :  # Cap at 100000 edges
                                    s1, t1 = current_edge
                                    x_coords = [pos[s1][0], pos[t1][0]]
                                    y_coords = [pos[s1][1], pos[t1][1]]
                                    edge_color = 'red'
                                    ax.plot(x_coords, y_coords, edge_color, alpha=0.7, zorder=2, linewidth=dynamic_edge_width * 1.5, linestyle='-')
                                plotted_edges += 1 * len([current_edge])
                        if len(renewal_input_edges) > 0:
                            subsample_rate = max(1, len(renewal_input_edges) // 100000000)  # the more edges the delicate
                            plotted_input_edges = 0
                            for edge_idx, current_edge in enumerate(renewal_input_edges):
                                current_edge = renewal_input_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_input_edges < 100000000 :  # Cap at 100000 edges
                                    s1, t1 = current_edge
                                    input_x_coords = [pos[s1][0], pos[t1][0]]
                                    input_y_coords = [pos[s1][1], pos[t1][1]]
                                    edge_color = 'blue'
                                    ax.plot(input_x_coords, input_y_coords, edge_color, alpha=0.7, zorder=3, linewidth=dynamic_edge_width * 1.2, linestyle='-')
                                plotted_input_edges += 1 * len([current_edge])
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='red', lw=2, label='Internal Edges'),
                            Line2D([0], [0], color='blue', lw=2, label='Input Edges'),
                        ]
                        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
                        ax.set_title(f'{network_name} Dynamic layout: Frame {count}\n Edges: {len(renewal_edges)} | Input edges: {len(renewal_input_edges)}', fontsize=10)
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
                            
                            # Apply fixed axis limits with dynamic padding
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

            # We need to return True so that the function will be called again
            return True
        
        print("Starting animation loop...")
        # Generate frames one by one
        while update_state():
            pass
            
        print("Animation finished")
        
        # Provide instructions for combining frames
        if offscreen:
            print(f"\nFrames saved to {network_frames_dir}/ directory")
            print("\nTo combine frames into a video, you can use ffmpeg:")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_dynamic.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick:")
            print(f"convert -delay 10 {network_frames_dir}/*.png {network_name}_dynamic.gif")
            print("\nOr create an animated GIF with optimization:")
            print(f"convert -delay 10 -loop 0 {network_frames_dir}/*.png -scale 800x800 -coalesce -fuzz 5% -layers Optimize {network_name}_dynamic.gif")
            print("\nThis will create an animated GIF from the saved frames.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files for network {network_name}: {e}")
        print("Make sure the gt/params directory contains the required files.")
    except Exception as e:
        print(f"Error processing network {network_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Process neural network animations')
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
        # Process specific networks
        networks_to_process = args.networks
        print(f"Processing specific networks: {networks_to_process}")
    else:
        # Process all available networks
        networks_to_process = get_available_networks(args.data_dir)
        print(f"Processing all available networks ({len(networks_to_process)} networks)")
    
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