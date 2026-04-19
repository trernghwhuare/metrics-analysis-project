#!/usr/bin/env python3

"""
Network animation using graph-tool and matplotlib.
This script creates an animated visualization of neural networks.
"""

import os
import sys
import argparse
from matplotlib import lines

os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "16"
import json
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk, GLib
from random import randint, shuffle, random
import numpy as np
from graph_tool.all import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
import matplotlib.animation as animation
from numpy.linalg import norm
from numpy.random import *
import logging
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# %%
seed(42)
seed_rng(42)

# Constants for layout
step = 0.005       # move step
K = 0.5            # preferred edge length

# SIRS Model parameters
x = 0.005   # spontaneous outbreak probability
r = 0.5     # I->R probability
s = 0.05    # R->S probability

# Create colormaps for the three states
cmap_S = plt.get_cmap('tab20c')    # inactive state colormap
cmap_I = plt.get_cmap('autumn')    # active state colormap  
cmap_R = plt.get_cmap('viridis')   # refractory state colormap
S = list(cmap_S(0.5))[:4]          # inactive state 
I = list(cmap_I(0.5))[:4]          # active state 
R = list(cmap_R(0.5))[:4]          # refractory state 

def get_available_networks(data_dir="gt/params"):
    """
    Get list of available network names from the data directory.
    Returns:
        list: List of network names (without file extensions)
    """
    if not os.path.exists(data_dir):
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
    # Load nodes data
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
    # Load params to get number of nodes
    params_file = os.path.join(data_dir, f"{network_name}_gt_params.json")
    with open(params_file, 'r') as f:
        params_data = json.load(f)
        num_nodes = params_data['number_pop_vertices']
        num_input_nodes = params_data['number_input_vertices']
    
    return nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors"""
    return [
        color1[0] + (color2[0] - color1[0]) * factor,
        color1[1] + (color2[1] - color1[1]) * factor,
        color1[2] + (color2[2] - color1[2]) * factor,
        color1[3] + (color2[3] - color1[3]) * factor
    ]


def state_to_rgba(state_value):
    """Convert state vector to RGBA values for smooth transitions"""
    # Compare with tolerance since we're now using float values from colormaps
    def approx_equal(a, b, tol=1e-6):
        if len(a) != len(b):
            return False
        return all(abs(a[i] - b[i]) < tol for i in range(len(a)))
    
    if approx_equal(state_value, S):    # S - inactive (from tab20c colormap)
        return S
    elif approx_equal(state_value, I):  # I - active (from collwarm colormap)
        return I
    # else:
    elif approx_equal(state_value, R):  
        return R                        # R - refractory (from tab20c_r colormap)


def process_network(network_name, data_dir="gt/params", max_count=300, no_offscreen=False):
    """Process a single network and generate state animation."""
    offscreen = not no_offscreen
    print(f"Loading network: {network_name}")
    try:
        # Load data
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        
        # Create graph-tool graph with the correct number of nodes from JSON
        g = price_network(num_nodes + num_input_nodes, c=0.2, directed=False)
        ini_state = AxelrodState(g, f=10, q=30, r=int(0.005))
        
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
        
        logging.info(f"Edge type distribution: Edges={edge_count}, Input Edges={input_edge_count}")
        logging.info(f"Connected graph built: {num_nodes + num_input_nodes} vertices, {edge_count + input_edge_count} edges")
        logging.info(f"Added edges: {edge_count} internal, {input_edge_count} input edges")
        
        total_nodes = num_nodes + num_input_nodes
        logging.info(f"Total nodes: {total_nodes}")
        total_edges = edge_count + input_edge_count
        logging.info(f"Total edges: {total_edges}")
        
        # Calculate network density (edges per node) for better scaling
        if total_nodes > 0:
            density = total_edges / total_nodes
        else:
            density = 0
        
        # Set dynamic_node_size and dynamic_edge_width based on network density
        if density <= 2.0:
            # Very sparse networks: large nodes and thick edges
            dynamic_node_size = max(60.0, 80.0 - (density * 5.0))
            dynamic_edge_width = max(2.5, 4.0 - (density * 0.5))
        elif density <= 10.0:
            # Sparse networks: medium-large nodes and medium-thick edges  
            dynamic_node_size = max(30.0, 50.0 - (density * 1.5))
            dynamic_edge_width = max(1.5, 2.5 - (density * 0.08))
        elif density <= 50.0:
            # Medium density networks: medium nodes and medium edges
            dynamic_node_size = max(15.0, 25.0 - (np.log10(density) * 3.0))
            dynamic_edge_width = max(0.8, 1.5 - (np.log10(density) * 0.15))
        elif density <= 200.0:
            # Dense networks: small nodes and thin edges
            dynamic_node_size = max(8.0, 12.0 - (np.log10(density) * 1.2))
            dynamic_edge_width = max(0.3, 0.8 - (np.log10(density) * 0.1))
        else:
            # Very dense networks: very small nodes and very thin edges
            dynamic_node_size = max(2.0, 5.0 / np.log10(density))
            dynamic_edge_width = max(0.1, 0.3 / np.log10(density))

        # Generate layout using graph-tool
        pos = sfdp_layout(g,  K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)
        # Adjusted parameters for smoother animation
        x = 0.005   # spontaneous outbreak probability
        r = 0.5     # I->R probability
        s = 0.05    # R->S probability

        ini_state = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            ini_state[v] = S

        curr_state = g.new_vertex_property("vector<double>")
        prev_state = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            curr_state[v] = S  # Start all nodes in inactive state
            prev_state[v] = ini_state[v] = S
        
        newly_transmited = g.new_vertex_property("bool")
        refractory = g.new_vertex_property("bool")

        edges = list(g.edges())    
        network_frames_dir = f"./frames/{network_name}/state"
        if offscreen and not os.path.exists(network_frames_dir):
            os.makedirs(network_frames_dir)
            print(f"Created network frames directory: {network_frames_dir}")
        
        count = 0
        frame_limits = None  # To store axis limits for consistent frame sizes
        all_x_positions = []
        all_y_positions = []
        vertex_index_map = {v: i for i, v in enumerate(g.vertices())}

        # This function will be called repeatedly to update the vertex layout
        def update_state():
            nonlocal count, frame_limits, all_x_positions, all_y_positions, vertex_index_map
            newly_transmited.a = False
            refractory.a = False
            
            # Count states for debugging
            active_count = 0
            inactive_count = 0
            refractory_count = 0
            # Count current states before updating
            for v in g.vertices():
                if curr_state[v] == I:  # active
                    active_count += 1
                elif curr_state[v] == S:  # inactive
                    inactive_count += 1
                elif curr_state[v] == R:  # refractory
                    refractory_count += 1
            print(f"Frame {count}: Active={active_count} | Inactive={inactive_count} | Refractory={refractory_count}")
            
            # Randomly make a few nodes initially infected to start the simulation
            if count == 0 and active_count == 0:
                vs = list(g.vertices())
                shuffle(vs)
                nodes = [v for v in vs if vertex_index_map[v] in node_id_to_index]
                input_nodes = [v for v in vs if vertex_index_map[v] in input_id_to_index]
                num_initial_infections = min(25, len(nodes), len(input_nodes))  # Infect at most 25 nodes
                for i in range(num_initial_infections):
                    if i < len(nodes):
                        curr_state[nodes[i]] = I
                        newly_transmited[nodes[i]] = True
                    if i < len(input_nodes):
                        curr_state[input_nodes[i]] = I
                        newly_transmited[input_nodes[i]] = True

            vs = list(g.vertices())
            shuffle(vs)
            for v in vs:
                v_idx = vertex_index_map[v]
                if curr_state[v] == I:  # active
                    if random() < r:
                        curr_state[v] = R
                elif curr_state[v] == S:  # inactive
                    if random() < x:
                        curr_state[v] = I
                        newly_transmited[v] = True
                    else:
                        ns = list(v.out_neighbors())
                        if len(ns) > 0:
                            w = ns[randint(0, len(ns))]  # choose a random neighbor
                            if curr_state[w] == I:
                                curr_state[v] = I
                                newly_transmited[v] = True
                elif random() < s:
                    curr_state[v] = S
                if curr_state[v] == R:
                    refractory[v] = True

            # Save frame as PNG
            if offscreen:
                # Save every 5th frame for better performance
                if count % 5 != 0 and count != max_count - 1:
                    count += 1
                    for v in g.vertices():
                        prev_state[v] = curr_state[v]
                    return True
                
                # Save frame as PNG file
                try:
                    # Create a matplotlib figure for this frame with dynamic dimensions
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
                    fig.patch.set_facecolor('white')  # Set figure background to white
                    
                    # Extract positions for plotting
                    x_pos = [pos[v][0] for v in g.vertices()]
                    y_pos = [pos[v][1] for v in g.vertices()]
                    
                    colors = []
                    sizes = []
                    for v in g.vertices():
                        colors.append(state_to_rgba(S))
                        sizes.append(dynamic_node_size)
                    ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=dynamic_edge_width * 0.5)

                    # Draw halos around active vertices for better visibility
                    active_x_pos = []
                    active_y_pos = []
                    active_colors = []
                    acitve_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == I:  # If vertex is active (infected)
                            active_x_pos.append(pos[v][0])
                            active_y_pos.append(pos[v][1])
                            active_colors.append(state_to_rgba(I))
                            acitve_sizes.append(dynamic_node_size * 2.0)  # Active nodes larger
                    if active_x_pos:
                        ax.scatter(active_x_pos, active_y_pos, c=active_colors, # cmap='autumn', 
                                   s=acitve_sizes, alpha=0.5, edgecolors='white', linewidth=dynamic_edge_width * 0.5)

                    inactive_x_pos = []
                    inactive_y_pos = []
                    inactive_colors = []
                    inactive_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == S:  # If vertex is inactive
                            inactive_x_pos.append(pos[v][0])
                            inactive_y_pos.append(pos[v][1])
                            inactive_colors.append(state_to_rgba(S))
                            inactive_sizes.append(dynamic_node_size * 0.2)  # Inactive nodes smaller
                    if inactive_x_pos:
                        ax.scatter(inactive_x_pos, inactive_y_pos, c=inactive_colors, # cmap='tab20c', 
                                   s=inactive_sizes, alpha=0.5, edgecolors='white', linewidth=dynamic_edge_width * 0.5)

                    refractory_x_pos = []
                    refractory_y_pos = []
                    refrac_colors = []
                    refrac_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == R:
                            refractory_x_pos.append(pos[v][0])
                            refractory_y_pos.append(pos[v][1])
                            refrac_colors.append(state_to_rgba(R))
                            refrac_sizes.append(dynamic_node_size * 1.2)  # Refractory nodes medium size
                    if refractory_x_pos:
                        ax.scatter(refractory_x_pos, refractory_y_pos, c=refrac_colors, # cmap='viridis', 
                                   s=refrac_sizes, alpha=0.7, edgecolors='white', linewidth=dynamic_edge_width * 0.5)
                    ax.set_facecolor('white')  # Set axes background to white
                    
                    # Plot edges
                    edges_list = list(g.edges())
                    # Subsample edges for faster plotting - only plot every Nth edge
                    subsample_rate = max(1, len(edges_list) // 50000000)  # the more edges the delicate
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
                                scr_state = curr_state[s1]
                                dst_state = curr_state[t1]
                            if scr_state == I or dst_state == I:
                                ax.plot(x_coords, y_coords, 'red', alpha=0.7, linewidth=dynamic_edge_width, linestyle='--')
                            elif scr_state == R or dst_state == R:
                                ax.plot(x_coords, y_coords, 'gold', alpha=0.5, linewidth=dynamic_edge_width * 0.7, linestyle='-')
                            elif scr_state == S or dst_state == S:
                                ax.plot(x_coords, y_coords, 'blue', alpha=0.3, linewidth=dynamic_edge_width * 0.5, linestyle=':')
                            plotted_edges += 1
                    if len(renewal_input_edges) > 0:
                            subsample_rate = max(1, len(renewal_input_edges) // 100000000)  # the more edges the delicate
                            plotted_input_edges = 0
                            for edge_idx, current_edge in enumerate(renewal_input_edges):
                                current_edge = renewal_input_edges[edge_idx]
                                if edge_idx % subsample_rate == 0 and plotted_input_edges < 100000000 :  # Cap at 100000 edges
                                    s1, t1 = current_edge
                                    input_x_coords = [pos[s1][0], pos[t1][0]]
                                    input_y_coords = [pos[s1][1], pos[t1][1]]
                                    scr_state = curr_state[s1]
                                    dst_state = curr_state[t1]
                                if scr_state == I or dst_state == I:
                                    ax.plot(input_x_coords, input_y_coords, 'purple', alpha=0.7, linewidth=dynamic_edge_width, linestyle='--')
                                elif scr_state == R or dst_state == R:
                                    ax.plot(input_x_coords, input_y_coords, 'lightcoral', alpha=0.5, linewidth=dynamic_edge_width * 0.7, linestyle='-')
                                elif scr_state == S or dst_state == S:
                                    ax.plot(input_x_coords, input_y_coords, 'green', alpha=0.3, linewidth=dynamic_edge_width * 0.5, linestyle=':')
                                plotted_input_edges += 1
                    
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', lw=2, label='I (Edges)', linestyle='--', alpha=0.7),
                        Line2D([0], [0], color='gold', lw=2, label='R (Edges)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='blue', lw=2, label='S (Edges)', linestyle=':', alpha=0.3),
                        Line2D([0], [0], color='purple', lw=2, label='I (Input Edges)', linestyle='--', alpha=0.7),
                        Line2D([0], [0], color='lightcoral', lw=2, label='R (Input Edges)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='green', lw=2, label='S (Input Edges)', linestyle=':', alpha=0.3),
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=12)
                    ax.set_title(f'{network_name} S->I->R->S epidemic model: Frame {count}\n Active={active_count} | Inactive={inactive_count} | Refractory={refractory_count}')
                    ax.set_aspect('equal')
                    
                    # Set fixed axis limits to ensure consistent frame sizes
                    if len(all_x_positions) > 0 and len(all_y_positions) > 0:
                        x_min, x_max = min(all_x_positions), max(all_x_positions)
                        y_min, y_max = min(all_y_positions), max(all_y_positions)
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        
                        # Add fixed padding around the network
                        padding = 0.1
                        padding_x = x_range * padding
                        padding_y = y_range * padding
                        # Store limits for future frames
                        frame_limits = (x_min - padding_x, x_max + padding_x, y_min - padding_y, y_max + padding_y)
                    
                    # Apply consistent axis limits for all frames
                    if frame_limits is not None:
                        ax.set_xlim(frame_limits[0], frame_limits[1])
                        ax.set_ylim(frame_limits[2], frame_limits[3])
                    
                    # Save frame as PNG file with even dimensions
                    plt.savefig(f'{network_frames_dir}/frame_{count:04d}.png', dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"Saved frame {count}")
                except Exception as e:
                    print(f"Error saving frame {count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Store current states as previous states
            for v in g.vertices():
                prev_state[v] = curr_state[v]
            
            # Increment the counter
            count += 1
            
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
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_state.mp4")
            print("\nFor large networks (>1000 nodes), use lower resolution to avoid memory issues:")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=1920:1080' -c:v libx264 -pix_fmt yuv420p {network_name}_state.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick (may fail for very large networks):")
            print(f"convert -delay 5 {network_frames_dir}/*.png {network_name}_state.gif")
            print("\nFor large networks, create optimized GIF with reduced resolution:")
            print(f"convert -delay 5 -loop 0 {network_frames_dir}/*.png -scale 1200x1200 -coalesce -fuzz 5% -layers Optimize {network_name}_state.gif")
            print("\nThis will create an animated GIF from the saved frames.")
            print("\nTo run without saving frames, use: python gt_state.py <NetworkName> no_offscreen")
        else:
            print("\nRunning without saving frames.")
            print("To save frames, run: python gt_state.py <NetworkName>")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files: {e}")
        print("Make sure the gt/params directory contains the required files.")
        available_networks = get_available_networks(data_dir)
        if available_networks:
            print(f"Available networks: {', '.join(available_networks[:10])}{'...' if len(available_networks) > 10 else ''}")
            print("Use --list-networks to see all available networks")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate network state animation')
    parser.add_argument('--data-dir', default='gt/params', help='Directory containing network data files')
    parser.add_argument('--max-frames', type=int, default=300, help='Maximum number of frames to generate')
    parser.add_argument('networks', nargs='*', help='Specific network names to process (e.g., M2M1S1_max_plus M1_max_plus). If not specified, processes the default network.')
    parser.add_argument('--all', action='store_true', help='Process all available networks (use instead of specifying networks)')
    parser.add_argument('--list-networks', action='store_true', help='List all available networks and exit')
    
    args = parser.parse_args()
    
    # Handle --list-networks option
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
    elif args.all:
        # Process all available networks
        networks_to_process = get_available_networks(args.data_dir)
        print(f"Processing all available networks ({len(networks_to_process)} networks)")
    else:
        # Process default network
        networks_to_process = ["M2M1S1_max_plus"]
        print(f"Processing default network: {networks_to_process[0]}")
    
    # Validate that requested networks exist
    available_networks = set(get_available_networks(args.data_dir))
    invalid_networks = [net for net in networks_to_process if net not in available_networks]
    if invalid_networks:
        print(f"Warning: The following networks were not found in {args.data_dir}: {invalid_networks}")
        networks_to_process = [net for net in networks_to_process if net in available_networks]
        if not networks_to_process:
            print("No valid networks to process. Exiting.")
            return
    
    # Check if "no_offscreen" is in the networks list (old usage pattern)
    no_offscreen = False
    if "no_offscreen" in networks_to_process:
        no_offscreen = True
        networks_to_process = [net for net in networks_to_process if net != "no_offscreen"]
        if not networks_to_process:
            networks_to_process = ["M2M1S1_max_plus"]
    
    # Process each network in sequence
    total_networks = len(networks_to_process)
    for i, network_name in enumerate(networks_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing network {i}/{total_networks}: {network_name}")
        print(f"{'='*60}")
        process_network(network_name, args.data_dir, args.max_frames, no_offscreen)
        print(f"Completed network {i}/{total_networks}: {network_name}")
    
    if total_networks > 1:
        print(f"\n{'='*60}")
        print(f"All networks processed successfully!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()