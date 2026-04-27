#!/usr/bin/env python3

"""
Network animation using graph-tool and matplotlib.
This script creates an animated visualization of neural networks.
"""
import os
import sys
from matplotlib import lines
import logging
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
import scipy.stats
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# %%
seed(42)
seed_rng(42)

# Constants for layout
step = 0.2       # move step
K = 0.8          # preferred edge length

# SIRS Model parameters
x = 0.02   # spontaneous outbreak probability
r = 0.8     # I->R probability
s = 0.2    # R->S probability

# Create colormaps for the three states
cmap_S = plt.get_cmap('tab20c')    # inactive state colormap
cmap_I = plt.get_cmap('autumn')    # active state colormap  
cmap_R = plt.get_cmap('viridis')   # refractory state colormap
exc_S = list(cmap_S(0.5))[:4]          # inactive state 
exc_I = list(cmap_I(0.5))[:4]          # Active state 
exc_R = list(cmap_R(0.5))[:4]          # Refractory state 
inh_S = list(cmap_S(0.5))[:6]          # inactive state 
inh_I = list(cmap_I(0.5))[:6]          # Active state 
inh_R = list(cmap_R(0.5))[:6] 

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
    
    if approx_equal(state_value, exc_S):    # S - inactive (from tab20c colormap)
        return exc_S
    elif approx_equal(state_value, exc_I):  # I - active (from collwarm colormap)
        return exc_I
    elif approx_equal(state_value, exc_R):  # R - refractory (from tab20c_r colormap)
        return exc_R
    elif approx_equal(state_value, inh_S):  # S - inactive (from tab20c colormap)
        return inh_S
    elif approx_equal(state_value, inh_I):  # I - active (from collwarm colormap)
        return inh_I
    elif approx_equal(state_value, inh_R):  # R - refractory (from tab20c_r colormap)
        return inh_R                        
def process_network(network_name, data_dir="gt/params", max_count=300, no_offscreen=False):
    """Process a single network and generate state animation."""
    offscreen = not no_offscreen
    print(f"Loading network: {network_name}")
    
    try:
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        g = price_network(num_nodes + num_input_nodes, c=0.2, directed=False)
        ini_exc_state = AxelrodState(g, f=10, q=30, r=int(0.005))
        ini_inh_state = PottsGlauberState(g, np.eye(4) * 0.1)

        # Create a mapping from node IDs to indices
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
        edge_count = 0
        for edge_info in edges_data:
            source_id = edge_info['source']
            target_id = edge_info['target']
            if source_id in node_id_to_index and target_id in node_id_to_index:
                source_idx = node_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]
                # g.add_edge(source_idx, target_idx)
            edge_count += 1
        input_id_to_index = {}
        for i, node_info in enumerate(input_nodes_data):
            input_id_to_index[node_info['component']] = num_nodes + i  # Offset by num_nodes
        
        inhibitory_inputs = set()
        excitatory_inputs = set()
        for node_info in input_nodes_data:
            if node_info['type'] == 'Inh' and node_info['component'] in input_id_to_index:
                inhibitory_inputs.add(input_id_to_index[node_info['component']])
            elif node_info['type'] == 'Exc' and node_info['component'] in input_id_to_index:
                excitatory_inputs.add(input_id_to_index[node_info['component']])
        input_edge_count = 0
        for input_edge_info in input_edges_data:
            source_id = input_edge_info['source']
            target_id = input_edge_info['target']
            if source_id in input_id_to_index and target_id in node_id_to_index:  # target_id should map to node_id_to_index, not input_id_to_index
                source_idx = input_id_to_index[source_id]
                target_idx = node_id_to_index[target_id]  # Use node_id_to_index for target (regular nodes)
                # g.add_edge(source_idx, target_idx)
            input_edge_count += 1  # Increment by 1 for each edge, not len(target_idx)
        
        print(f"Edge type distribution: Edges={edge_count}, Input Edges={input_edge_count}")
        print(f"Connected graph built: {num_nodes + num_input_nodes} vertices, {edge_count + input_edge_count} edges")
        print(f"Added edges: {edge_count} internal, {input_edge_count} input edges")
        total_nodes = num_nodes + num_input_nodes
        logging.info(f"Total nodes: {total_nodes}")
        total_edges = edge_count + input_edge_count
        logging.info(f"Total edges: {total_edges}")
        # Calculate network density (edges per node) for better scaling
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
        
        # Count edges by type to calculate type-specific densities
        ee_count = ei_count = ie_count = ii_count = input_ee_count = input_ii_count = 0
        edges_list = list(g.edges())
        for e in edges_list:
            src_idx = int(e.source())
            tgt_idx = int(e.target())
            src_type = node_types.get(src_idx, 'Exc')
            tgt_type = node_types.get(tgt_idx, 'Exc')
            is_input_edge = src_idx >= num_nodes
            
            if is_input_edge:
                if src_type == 'Exc':
                    input_ee_count += 1
                elif src_type == 'Inh':
                    input_ii_count += 1
            else:
                if src_type == 'Exc' and tgt_type == 'Exc':
                    ee_count += 1
                elif src_type == 'Exc' and tgt_type == 'Inh':
                    ei_count += 1
                elif src_type == 'Inh' and tgt_type == 'Exc':
                    ie_count += 1
                elif src_type == 'Inh' and tgt_type == 'Inh':
                    ii_count += 1
        
        # Calculate densities for each edge type
        ee_density = ee_count / total_edges if total_edges > 0 else 0
        ei_density = ei_count / total_edges if total_edges > 0 else 0
        ie_density = ie_count / total_edges if total_edges > 0 else 0
        ii_density = ii_count / total_edges if total_edges > 0 else 0
        input_ee_density = input_ee_count / total_edges if total_edges > 0 else 0
        input_ii_density = input_ii_count / total_edges if total_edges > 0 else 0
        
        # Set dynamic_node_size based on overall network density
        if density <= 2.0:
            dynamic_node_size = max(60.0, 80.0 - (density * 5.0))
        elif density <= 10.0:
            dynamic_node_size = max(30.0, 50.0 - (density * 1.5))
        elif density <= 50.0:
            dynamic_node_size = max(15.0, 25.0 - (np.log10(density) * 3.0))
        elif density <= 200.0:
            dynamic_node_size = max(8.0, 12.0 - (np.log10(density) * 1.2))
        else:
            dynamic_node_size = max(2.0, 5.0 / np.log10(density))
        
        # Set edge widths based on type-specific densities for better animation effects
        def calculate_edge_width(edge_density):
            if edge_density <= 0.5:
                # Very sparse edges: thicker lines for visibility
                return max(2.0, 3.0 - (edge_density * 1.0))
            elif edge_density <= 2.0:
                # Sparse edges: medium-thick lines
                return max(1.2, 2.0 - (edge_density * 0.3))
            elif edge_density <= 5.0:
                # Medium density edges: medium lines
                return max(0.5, 1.0 - (np.log10(edge_density) * 0.2))
            elif edge_density <= 20.0:
                # Dense edges: thin lines
                return max(0.1, 0.3 - (np.log10(edge_density) * 0.1))
            else:
                # Very dense edges: very thin lines
                return max(0.05, 0.1 / np.log10(edge_density))
        
        # Calculate specific edge widths for each type
        ee_edge_width = calculate_edge_width(ee_density)
        ei_edge_width = calculate_edge_width(ei_density) 
        ie_edge_width = calculate_edge_width(ie_density)
        ii_edge_width = calculate_edge_width(ii_density)
        input_ee_edge_width = calculate_edge_width(input_ee_density)
        input_ii_edge_width = calculate_edge_width(input_ii_density)

        # Generate layout using graph-tool
        pos = sfdp_layout(g,  K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)
        
        # Adjusted parameters for smoother animation
        x = 0.02   # spontaneous outbreak probability
        r = 0.8    # I->R probability
        s = 0.2   # R->S probability

        ini_exc_state = g.new_vertex_property("vector<double>")
        ini_inh_state = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            ini_exc_state[v] = exc_S
            ini_inh_state[v] = inh_S
        
        curr_exc_state = g.new_vertex_property("vector<double>")
        prev_exc_state = g.new_vertex_property("vector<double>")
        curr_inh_state = g.new_vertex_property("vector<double>")
        prev_inh_state = g.new_vertex_property("vector<double>")
        
        # Create vertex index map for proper initialization
        vertex_index_map = {v: i for i, v in enumerate(g.vertices())}
        
        # Initialize states based on node type
        for v in g.vertices():
            v_idx = vertex_index_map[v]
            is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
            is_inhibitory = (v_idx in inhibitory_nodes) or (v_idx in inhibitory_inputs)
            
            if is_excitatory:
                curr_exc_state[v] = exc_S
                prev_exc_state[v] = exc_S
            if is_inhibitory:
                curr_inh_state[v] = inh_S  
                prev_inh_state[v] = inh_S

        newly_exc_transmited = g.new_vertex_property("bool")
        newly_inh_transmited = g.new_vertex_property("bool")
        exc_refractory = g.new_vertex_property("bool")
        inh_refractory = g.new_vertex_property("bool")

        edges = list(g.edges()) 
           
        max_count = 300
        # Check if we should disable offscreen rendering
        no_offscreen = sys.argv[1] == "no_offscreen" if len(sys.argv) > 1 else False
        offscreen = not no_offscreen
        network_frames_dir = f"./frames/{network_name}/state_adv"
        if offscreen and not os.path.exists(network_frames_dir):
            os.makedirs(network_frames_dir)
            print(f"Created network frames directory: {network_frames_dir}")
        
        count = 0
        frame_limits = None  # To store axis limits for consistent frame sizes
        all_x_positions = []
        all_y_positions = []
        
        # Create vertex index map once to use in both the main loop and visualization
        vertex_index_map = {v: i for i, v in enumerate(g.vertices())}
        
        # This function will be called repeatedly to update the vertex layout
        def update_state():
            nonlocal count, frame_limits, all_x_positions, all_y_positions, vertex_index_map
            newly_exc_transmited.a = False
            newly_inh_transmited.a = False
            exc_refractory.a = False
            inh_refractory.a = False
            
            # Count states for debugging
            exc_active_count = 0
            exc_inactive_count = 0
            exc_refractory_count = 0
            inh_active_count = 0
            inh_inactive_count = 0
            inh_refractory_count = 0
            
            # Count current states before updating
            for v in g.vertices():
                v_idx = vertex_index_map[v]
                is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                is_inhibitory = (v_idx in inhibitory_nodes) or (v_idx in inhibitory_inputs)
                
                if is_excitatory:
                    if curr_exc_state[v] == exc_I:  # active
                        exc_active_count += 1
                    elif curr_exc_state[v] == exc_S:  # inactive
                        exc_inactive_count += 1
                    elif curr_exc_state[v] == exc_R:  # refractory
                        exc_refractory_count += 1

                if is_inhibitory:
                    if curr_inh_state[v] == inh_I:  # active
                        inh_active_count += 1
                    elif curr_inh_state[v] == inh_S:  # inactive
                        inh_inactive_count += 1
                    elif curr_inh_state[v] == inh_R:  # refractory
                        inh_refractory_count += 1
            print(f"Frame {count} Excitatory: Active(I)={exc_active_count} | Inactive(S)={exc_inactive_count} | Refractory(R)={exc_refractory_count}\n Inhibitory: Active(I)={inh_active_count} | Inactive(S)={inh_inactive_count} | Refractory(R)={inh_refractory_count}")
            
            # Randomly make a few nodes initially infected to start the simulation
            if count == 0 and exc_active_count == 0 and inh_active_count == 0:
                # Infect a small number of random nodes to start the simulation
                vs = list(g.vertices())
                shuffle(vs)

                exc_nodes = [v for v in vs if vertex_index_map[v] in excitatory_nodes]
                inh_nodes = [v for v in vs if vertex_index_map[v] in inhibitory_nodes]
                exc_input_nodes = [v for v in vs if vertex_index_map[v] in excitatory_inputs]
                inh_input_nodes = [v for v in vs if vertex_index_map[v] in inhibitory_inputs]

                num_initial_exc_infections = min(5, len(exc_nodes), len(exc_input_nodes))
                for i in range(num_initial_exc_infections):
                    if i < len(exc_nodes):
                        curr_exc_state[exc_nodes[i]] = exc_I
                        newly_exc_transmited[exc_nodes[i]] = True
                    if i < len(exc_input_nodes):
                        curr_exc_state[exc_input_nodes[i]] = exc_I
                        newly_exc_transmited[exc_input_nodes[i]] = True
                        
                # Infect inhibitory nodes
                num_initial_inh_infections = min(5, len(inh_nodes), len(inh_input_nodes))
                for i in range(num_initial_inh_infections):
                    if i < len(inh_nodes):
                        curr_inh_state[inh_nodes[i]] = inh_I
                        newly_inh_transmited[inh_nodes[i]] = True
                    if i < len(inh_input_nodes):
                        curr_inh_state[inh_input_nodes[i]] = inh_I
                        newly_inh_transmited[inh_input_nodes[i]] = True

            vs = list(g.vertices())
            shuffle(vs)
            for v in vs:
                v_idx = vertex_index_map[v]
                
                is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                is_inhibitory = (v_idx in inhibitory_nodes) or (v_idx in inhibitory_inputs)
                
                # Update excitatory neuron states
                if is_excitatory:
                    if curr_exc_state[v] == exc_I:  # active
                        if random() < r:
                            curr_exc_state[v] = exc_R
                    elif curr_exc_state[v] == exc_S:  # inactive
                        if random() < x:
                            curr_exc_state[v] = exc_I
                            newly_exc_transmited[v] = True
                        else:
                            ns = list(v.out_neighbors())
                            if len(ns) > 0:
                                w = ns[randint(0, len(ns))]  # choose a random neighbor
                                if curr_exc_state[w] == exc_I : 
                                    curr_exc_state[v] = exc_I
                                    newly_exc_transmited[v] = True
                    elif random() < s:
                        curr_exc_state[v] = exc_S
                    if curr_exc_state[v] == exc_R:  # Refractory
                        exc_refractory[v] = True

                # g.set_vertex_filter(exc_refractory.t(lambda x: x))
                
                if is_inhibitory:
                    if curr_inh_state[v] == inh_I:  # active
                        if random() < r:
                            curr_inh_state[v] = inh_R
                    elif curr_inh_state[v] == inh_S:  # inactive
                        if random() < x:
                            curr_inh_state[v] = inh_I
                        else:
                            ns = list(v.out_neighbors())
                            if len(ns) > 0:
                                w = ns[randint(0, len(ns))]  # choose a random neighbor
                                if curr_inh_state[w] == inh_I:  
                                    curr_inh_state[v] = inh_I
                                    newly_inh_transmited[v] = True
                    elif random() < s:
                        curr_inh_state[v] = inh_S
                    if curr_inh_state[v] == inh_R:  # Refractory
                        inh_refractory[v] = True
            
                # g.set_vertex_filter(inh_refractory.t(lambda x: x))

            # Save frame as PNG
            if offscreen:
                # Save every 5th frame for better performance
                if count % 5 != 0 and count != max_count - 1:
                    # Just increment counter and continue without saving
                    count += 1
                    # Store current states as previous states
                    for v in g.vertices():
                        prev_exc_state[v] = curr_exc_state[v]
                        prev_inh_state[v] = curr_inh_state[v]
                    return True
                
                # Save frame as PNG file
                try:
                    # Create a matplotlib figure for this frame with even dimensions
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
                    x_pos = [pos[v][0] for v in g.vertices()]
                    y_pos = [pos[v][1] for v in g.vertices()]
                    colors = []
                    sizes = []
                    for v in g.vertices():
                        v_idx = vertex_index_map[v]
                        is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                        if is_excitatory:
                            exc_state = ini_exc_state[v]
                            colors.append(state_to_rgba(exc_state))
                            sizes.append(dynamic_node_size)
                        else:  
                            inh_state = ini_inh_state[v]
                            colors.append(state_to_rgba(inh_state))
                            sizes.append(dynamic_node_size* 0.75)
                    ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=ee_edge_width * 0.5, zorder=1)

                    # Draw halos around active vertices for better visibility
                    exc_active_x_pos = []
                    exc_active_y_pos = []
                    inh_active_x_pos = []
                    inh_active_y_pos = []
                    exc_active_colors = []
                    exc_active_sizes = []
                    inh_active_colors = []
                    inh_active_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_I:  # If vertex is active (infected)
                            exc_active_x_pos.append(pos[v][0])
                            exc_active_y_pos.append(pos[v][1])
                            exc_active_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_active_sizes.append(dynamic_node_size * 1.2)
                        elif curr_inh_state[v] == inh_I:  # If vertex is active (infected)
                            inh_active_x_pos.append(pos[v][0])
                            inh_active_y_pos.append(pos[v][1])
                            inh_active_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_active_sizes.append(dynamic_node_size * 1.2)
                    if exc_active_x_pos:
                        ax.scatter(exc_active_x_pos, exc_active_y_pos, c=exc_active_colors,
                                   s=exc_active_sizes, alpha=0.5, edgecolors='white', linewidth=ee_edge_width * 0.8, zorder=2)
                    elif inh_active_x_pos:
                        ax.scatter(inh_active_x_pos, inh_active_y_pos, c=inh_active_colors,
                                   s=inh_active_sizes, alpha=0.5, edgecolors='white', linewidth=ee_edge_width * 0.8, zorder=2)

                    exc_inactive_x_pos = []
                    exc_inactive_y_pos = []
                    inh_inactive_x_pos = []
                    inh_inactive_y_pos = []
                    exc_inactive_colors = []
                    exc_inactive_sizes = []
                    inh_inactive_colors = []
                    inh_inactive_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_S:  # If vertex is inactive
                            exc_inactive_x_pos.append(pos[v][0])
                            exc_inactive_y_pos.append(pos[v][1])
                            exc_inactive_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_inactive_sizes.append(dynamic_node_size * 0.8)
                        elif curr_inh_state[v] == inh_S:  # If vertex is inactive
                            inh_inactive_x_pos.append(pos[v][0])
                            inh_inactive_y_pos.append(pos[v][1])
                            inh_inactive_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_inactive_sizes.append(dynamic_node_size * 0.8)
                    if exc_inactive_x_pos:
                        ax.scatter(exc_inactive_x_pos, exc_inactive_y_pos, c=exc_inactive_colors,
                                   s=exc_inactive_sizes, alpha=0.5, edgecolors='white', linewidth=ee_edge_width * 0.5, zorder=2)
                    elif inh_inactive_x_pos:
                        ax.scatter(inh_inactive_x_pos, inh_inactive_y_pos, c=inh_inactive_colors, 
                                   s=inh_inactive_sizes, alpha=0.5, edgecolors='white', linewidth=ee_edge_width * 0.5, zorder=2)

                    exc_refractory_x_pos = []
                    exc_refractory_y_pos = []
                    inh_refractory_x_pos = []
                    inh_refractory_y_pos = []
                    exc_refrac_colors = []
                    exc_refrac_sizes = []
                    inh_refrac_colors = []
                    inh_refrac_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_R:
                            exc_refractory_x_pos.append(pos[v][0])
                            exc_refractory_y_pos.append(pos[v][1])
                            exc_refrac_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_refrac_sizes.append(dynamic_node_size * 1.2)
                        elif curr_inh_state[v] == inh_R:
                            inh_refractory_x_pos.append(pos[v][0])
                            inh_refractory_y_pos.append(pos[v][1])
                            inh_refrac_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_refrac_sizes.append(dynamic_node_size * 1.2)
                    if exc_refractory_x_pos:
                        ax.scatter(exc_refractory_x_pos, exc_refractory_y_pos, c=exc_refrac_colors, 
                                   s=exc_refrac_sizes, alpha=0.7, edgecolors='white', linewidth=ee_edge_width * 0.7, zorder=2)
                    elif inh_refractory_x_pos:
                        ax.scatter(inh_refractory_x_pos, inh_refractory_y_pos, c=inh_refrac_colors, 
                                   s=inh_refrac_sizes, alpha=0.7, edgecolors='white', linewidth=ee_edge_width * 0.7, zorder=2)
                    ax.set_facecolor('white')  # Set axes background to white
                    
                    # Plot edges
                    edges_list = list(g.edges())
                    # Subsample edges for faster plotting - only plot every Nth edge
                    subsample_rate = max(1, len(edges_list) // 5000000)  # the more edges the delicate
                    
                    plotted_ee_edges = 0
                    plotted_ei_edges = 0
                    plotted_ie_edges = 0
                    plotted_ii_edges = 0
                    plotted_input_ee_edges = 0
                    plotted_input_ii_edges = 0
                    # node_types is already defined in the outer scope

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
                    # Draw edges with varying properties based on node states
                    if len(ee_edges) > 0:
                        subsample_rate = max(1, len(ee_edges) // 5000000)  # Adjust subsample rate for EE edges
                        plotted_ee_edges = 0
                        for edge_idx, current_edge in enumerate(ee_edges):
                            current_edge = ee_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ee_edges < 5000000:
                                ees, eet = current_edge
                                ee_x_coords = [pos[ees][0], pos[eet][0]]
                                ee_y_coords = [pos[ees][1], pos[eet][1]]
                                src_e_state = curr_exc_state[ees]
                                dst_e_state = curr_exc_state[eet]
                                src_i_state = curr_inh_state[ees]
                                dst_i_state = curr_inh_state[eet]
                            if src_e_state == exc_I or dst_e_state == exc_I:  # If either node is active
                                ax.plot(ee_x_coords, ee_y_coords, 'red', alpha=0.7, linewidth=ee_edge_width* 0.7 , linestyle='-', zorder=3)
                            elif src_e_state == exc_R or dst_e_state == exc_R:  # If either node is refractory
                                ax.plot(ee_x_coords, ee_y_coords, 'coral', alpha=0.5, linewidth=ee_edge_width* 0.7, linestyle='-', zorder=3)
                            elif src_e_state == exc_S or dst_e_state == exc_S:
                                ax.plot(ee_x_coords, ee_y_coords, 'blue', alpha=0.3, linewidth=ee_edge_width * 0.7, linestyle='-', zorder=3)
                            plotted_ee_edges += 1
                    # Plot Input_EE edges
                    if len(input_ee_edges) > 0:
                        subsample_rate = max(1, len(input_ee_edges) // 5000000)  # Adjust subsample rate for Input_EE edges
                        plotted_input_ee_edges = 0
                        for edge_idx, current_edge in enumerate(input_ee_edges):
                            current_edge = input_ee_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_input_ee_edges < 5000000:
                                ees1, eet1 = current_edge
                                input_ee_x_coords = [pos[ees1][0], pos[eet1][0]]
                                input_ee_y_coords = [pos[ees1][1], pos[eet1][1]]
                                src_e_state = curr_exc_state[ees1]
                                dst_e_state = curr_exc_state[eet1]
                                src_i_state = curr_inh_state[ees1]
                                dst_i_state = curr_inh_state[eet1]
                            if src_e_state == exc_I or dst_e_state == exc_I:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'gold',  alpha=0.7, linewidth=input_ee_edge_width * 0.7, linestyle=(0, (5, 1)), zorder=3)
                            elif src_e_state == exc_R or dst_e_state == exc_R:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'cyan', alpha=0.5, linewidth=input_ee_edge_width* 0.7, linestyle=(0, (5, 1)), zorder=3)
                            else:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'purple', alpha=0.3, linewidth=input_ee_edge_width * 0.7, linestyle=(0, (5, 1)), zorder=3)
                            plotted_input_ee_edges += 1            
                    if len(ii_edges) > 0:
                        subsample_rate = max(1, len(ii_edges) // 5000000)  # Adjust subsample rate for II edges
                        plotted_ii_edges = 0
                        for edge_idx, current_edge in enumerate(ii_edges):
                            current_edge = ii_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ii_edges < 5000000:
                                iis, iit = current_edge
                                ii_x_coords = [pos[iis][0], pos[iit][0]]
                                ii_y_coords = [pos[iis][1], pos[iit][1]]
                                src_e_state = curr_exc_state[iis]
                                dst_e_state = curr_exc_state[iit]
                                src_i_state = curr_inh_state[iis]
                                dst_i_state = curr_inh_state[iit]              
                            if src_i_state == inh_I or dst_i_state == inh_I:
                                ax.plot(ii_x_coords, ii_y_coords, 'red', alpha=0.7, linewidth=ii_edge_width * 0.7, linestyle='--', zorder=3)
                            elif src_i_state == inh_R or dst_i_state == inh_R:
                                ax.plot(ii_x_coords, ii_y_coords, 'coral', alpha=0.5, linewidth=ii_edge_width * 0.7, linestyle='--', zorder=3)
                            elif src_i_state == inh_S or dst_i_state == inh_S:
                                ax.plot(ii_x_coords, ii_y_coords, 'blue', alpha=0.3, linewidth=ii_edge_width * 0.7, linestyle='--', zorder=3)
                            plotted_ii_edges += 1
                    if len(input_ii_edges) > 0:
                        subsample_rate = max(1, len(input_ii_edges) // 5000000)
                        plotted_input_ii_edges = 0
                        for edge_idx, current_edge in enumerate(input_ii_edges):
                            current_edge = input_ii_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_input_ii_edges < 5000000:
                                iis1, iit1 = current_edge
                                input_ii_x_coords = [pos[iis1][0], pos[iit1][0]]
                                input_ii_y_coords = [pos[iis1][1], pos[iit1][1]]
                                src_e_state = curr_exc_state[iis1]
                                dst_e_state = curr_exc_state[iit1]
                                src_i_state = curr_inh_state[iis1]
                                dst_i_state = curr_inh_state[iit1]
                            if src_i_state == inh_I or dst_i_state == inh_I:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'gold',  alpha=0.7, linewidth=input_ii_edge_width * 0.7, linestyle=(5, (10, 3)), zorder=3)
                            elif src_i_state == inh_R or dst_i_state == inh_R:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'cyan', alpha=0.5, linewidth=input_ii_edge_width * 0.7, linestyle=(5, (10, 3)), zorder=3)
                            else:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'purple',  alpha=0.3, linewidth=input_ii_edge_width * 0.7, linestyle=(5, (10, 3)), zorder=3)
                            plotted_input_ii_edges += 1
                    if len(ei_edges) > 0:
                        subsample_rate = max(1, len(ei_edges) // 5000000)  # Adjust subsample rate for EI edges
                        plotted_ei_edges = 0
                        for edge_idx, current_edge in enumerate(ei_edges):
                            current_edge = ei_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ei_edges < 5000000:
                                eis, eit = current_edge
                                ei_x_coords = [pos[eis][0], pos[eit][0]]
                                ei_y_coords = [pos[eis][1], pos[eit][1]]
                                src_e_state = curr_exc_state[eis]
                                dst_e_state = curr_exc_state[eit]
                                src_i_state = curr_inh_state[eis]
                                dst_i_state = curr_inh_state[eit]                                
                            if src_e_state == exc_I or dst_i_state == inh_I:
                                ax.plot(ei_x_coords, ei_y_coords, 'red', alpha=0.7, linewidth=ei_edge_width * 0.7, linestyle=(0, (5, 1)), zorder=3)
                            elif src_e_state == exc_R or dst_i_state == inh_R:
                                ax.plot(ei_x_coords, ei_y_coords, 'coral',  alpha=0.5, linewidth=ei_edge_width * 0.7, linestyle=(0, (5, 1)), zorder=3)
                            elif src_e_state == exc_S or dst_i_state == inh_S:
                                ax.plot(ei_x_coords, ei_y_coords, 'blue',  alpha=0.3, linewidth=ei_edge_width * 0.7, linestyle=(0, (5, 1)), zorder=3)
                            plotted_ei_edges += 1
                    if len(ie_edges) > 0:
                        subsample_rate = max(1, len(ie_edges) // 5000000)  # Adjust subsample rate for IE edges
                        plotted_ie_edges = 0
                        for edge_idx, current_edge in enumerate(ie_edges):
                            current_edge = ie_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ie_edges < 5000000:
                                ies, iet = current_edge
                                ie_x_coords = [pos[ies][0], pos[iet][0]]
                                ie_y_coords = [pos[ies][1], pos[iet][1]]
                                src_e_state = curr_exc_state[ies]
                                dst_e_state = curr_exc_state[iet]
                                src_i_state = curr_inh_state[ies]
                                dst_i_state = curr_inh_state[iet]
                            if src_i_state == inh_I or dst_e_state == exc_I:
                                ax.plot(ie_x_coords, ie_y_coords, 'red', alpha=0.7, linewidth=ie_edge_width* 0.7, linestyle=(5, (10, 3)), zorder=3)
                            elif src_i_state == inh_R or dst_e_state == exc_R:
                                ax.plot(ie_x_coords, ie_y_coords, 'coral', alpha=0.5, linewidth=ie_edge_width* 0.7, linestyle=(5, (10, 3)), zorder=3)
                            elif src_i_state == inh_S or dst_e_state == exc_S:
                                ax.plot(ie_x_coords, ie_y_coords, 'blue', alpha=0.3, linewidth=ie_edge_width * 0.7, linestyle=(5, (10, 3)), zorder=3)
                            plotted_ie_edges += 1
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', lw=2, label='I (EE)', linestyle='-', alpha=0.7),
                        Line2D([0], [0], color='red', lw=2, label='I (II)', linestyle='--', alpha=0.7),
                        Line2D([0], [0], color='red', lw=2, label='I (EI)', linestyle=(0, (5, 1)), alpha=0.7),
                        Line2D([0], [0], color='red', lw=2, label='I (IE)', linestyle=(5, (10, 3)), alpha=0.7),
                        
                        Line2D([0], [0], color='coral', lw=2, label='R (EE)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='coral', lw=2, label='R (II)', linestyle='--', alpha=0.5),
                        Line2D([0], [0], color='coral', lw=2, label='R (EI)', linestyle=(0, (5, 1)), alpha=0.5),
                        Line2D([0], [0], color='coral', lw=2, label='R (IE)', linestyle=(5, (10, 3)), alpha=0.5),
                        
                        Line2D([0], [0], color='blue', lw=2, label='S (EE)', linestyle='-', alpha=0.3),
                        Line2D([0], [0], color='blue', lw=2, label='S (II)', linestyle='--', alpha=0.3),
                        Line2D([0], [0], color='blue', lw=2, label='S (EI)', linestyle=(0, (5, 1)), alpha=0.3),
                        Line2D([0], [0], color='blue', lw=2, label='S (IE)', linestyle=(5, (10, 3)), alpha=0.3),

                        Line2D([0], [0], color='gold', lw=2, label='I (Input_EE)', linestyle=(0, (5, 1)), alpha=0.7),
                        Line2D([0], [0], color='gold', lw=2, label='I (Input_II)', linestyle=(5, (10, 3)), alpha=0.7),
                        Line2D([0], [0], color='cyan', lw=2, label='R (Input_EE)', linestyle=(0, (5, 1)), alpha=0.5),
                        Line2D([0], [0], color='cyan', lw=2, label='R (Input_II)', linestyle=(5, (10, 3)), alpha=0.5),
                        Line2D([0], [0], color='purple', lw=2, label='S (Input_EE)', linestyle=(0, (5, 1)), alpha=0.3),
                        Line2D([0], [0], color='purple', lw=2, label='S (Input_II)', linestyle=(5, (10, 3)), alpha=0.3),
                    ]
                    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=8)
                    
                    ax.set_title(f'{network_name} S->I->R->S epidemic model Frame {count}\n Excitatory: Active(I)={exc_active_count} | Inactive(S)={exc_inactive_count} | Refractory(R)={exc_refractory_count}\n Inhibitory: Active(I)={inh_active_count} | Inactive(S)={inh_inactive_count} | Refractory(R)={inh_refractory_count}')
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
                prev_exc_state[v] = curr_exc_state[v]
                prev_inh_state[v] = curr_inh_state[v]
            
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
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_state_adv.mp4")
            print("\nFor large networks with multiple node types, maintain full resolution for detailed visualization:")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p -crf 18 {network_name}_state_adv.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick:")
            print(f"convert -delay 5 {network_frames_dir}/*.png {network_name}_state_adv.gif")
            print("\nOr create an animated GIF with optimization:")
            print(f"convert -delay 5 -loop 0 {network_frames_dir}/*.png -coalesce -fuzz 5% -layers Optimize {network_name}_state_adv.gif")
            print("\nThis will create an animated GIF from the saved frames.")
            print("\nTo run without saving frames, use: python gt_state_adv.py no_offscreen")
        else:
            print("\nRunning without saving frames.")
            print("To save frames, run without arguments: python gt_state_adv.py")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files: {e}")
        print("Make sure the gt/params directory contains the required files.")
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