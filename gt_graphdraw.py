# %%
import os
import sys
import time
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from tqdm import tqdm
from numpy.linalg import norm
from numpy.random import *
import json
import gc
import random
import scipy
from graph_tool.all import *
import subprocess

from pyneuroml.pynml import read_neuroml2_file
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from graph_tool.all import *
import graph_tool.all as gt
import time
import logging
import gc
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# %% 
TCs_intralaminar = ["TCRil","nRTil"]
TCs_matrix = ["TCR","TCRm","nRTm"]
TCs_core = ["nRT","TCRc","nRTc"]
thalamus = TCs_core + TCs_matrix + TCs_intralaminar

intra_m = {"DAC","NGCDA","NGCSA","HAC","SLAC","MC","BTC","DBC","BP","NGC","LBC", "NBC","SBC","ChC"}
pyr_m = {"PC","SP","SS", "TTPC1","TTPC2","UTPC","STPC","IPC","BPC","TPC_L4","TPC_L1"}

exc_e = {"cADpyr", "cAC", "cNAC", "cSTUT", "cIR","TCR","TCRm","nRTm"}
inh_e = {"bAC","bNAC","dNAC","bSTUT","dSTUT","bIR","TCRil","nRTil","nRT","TCRc","nRTc"}
e_type_list = {"cADpyr", "cAC", "bAC", "cNAC","bNAC", "dNAC", "cSTUT", "bSTUT","dSTUT", "cIR", "bIR"}

layer_list = {"L1","L23","L4","L5","L6"}
Region_list = {"M2a","M2b","M1a","M1b","S1a","S1b"}
gen_list = {"PG","VC","ComInp"}

def get_pop_type(pop_id):
    parts = pop_id.split('_') if "_" in pop_id else [pop_id]
    if len(parts) >= 2 and parts[0] in Region_list:
        for exc_type in exc_e:
            if parts[1].startswith(exc_type):
                return 'exc'
        for inh_type in inh_e:
            if parts[1].startswith(inh_type):
                return 'inh'
    elif len(parts) >= 2 and parts[0] not in Region_list:
        for exc_type in exc_e:
            if parts[0].startswith(exc_type):
                return 'exc'
        for inh_type in inh_e:
            if parts[0].startswith(inh_type):
                return 'inh'
    elif len(parts) == 1:
        for exc_type in exc_e:
            if parts[0] in exc_e:
                return 'exc'
        for inh_type in inh_e:
            if parts[0] in inh_e:
                return 'inh'
    else:
        return parts[0]


def get_Vprefix(pop_id):
    try:
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1 and parts[0] in thalamus: # thalamus
            return parts[0]
        elif len(parts) == 2 and parts[0] in Region_list and parts[1] in thalamus: # regional_thalamus
            return parts[1]
        elif len(parts) == 6 and parts[1] in layer_list: # layered_pop_vprefix
            return '_'.join(parts[1:3])
        elif len(parts) == 7 and parts[0] in Region_list and parts[2] in layer_list: # regional_layered_pop_vprefix
            return '_'.join(parts[2:4]) 
        elif len(parts) == 7 and parts[1] in layer_list and parts[3] in layer_list:
            return '_'.join(parts[1:3])
        return "unknown"
    except Exception as e:
        logging.error(f"Error in get_Vprefix with pop_id {pop_id}: {e}")
        return "unknown"

def get_layer(pop_id):
    try:
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1:
            return "thalamus"
        elif len(parts) == 2 and parts[0] in Region_list:
            return "thalamus"
        elif len(parts) >= 6 and parts[1] in layer_list:
            return parts[1]
        elif len(parts) >= 7 and parts[0] in Region_list:  # Fixed: was 'Regions' which is undefined
            return parts[2] 
        return "unknown"
    except Exception as e:
        logging.error(f"Error in get_layer with pop_id {pop_id}: {e}")
        return "unknown"

def get_Region(pop_id):
    try:
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if parts[0] in Region_list:
            return parts[0]
        return None
    except Exception as e:
        logging.error(f"Error in get_Region with pop_id {pop_id}: {e}")
        return None

def get_input_type(input_id):
    input_id_lower = input_id.lower()
    if 'exc' in input_id_lower:
        return 'exc'
    elif 'inh' in input_id_lower:
        return 'inh'
    
def get_gen_type(ilist_id):
    if '_PG_' in ilist_id:
        return 'pulse_generators'
    elif '_ComInp_' in ilist_id:
        return 'compound_inputs'
    elif '_VC_' in ilist_id:
        return 'voltage_clamp_triples'
    else:
        return 'unknown'        


def visualize_network(nml_net_file, p_intra, p_inter, base_name):
    # Create output directory if it doesn't exist
    output_dir = f"graph_draw/{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    nml_doc = read_neuroml2_file(nml_net_file)
    G = price_network(len(nml_doc.networks[0].populations) 
                      + len(nml_doc.pulse_generators) 
                      + len(nml_doc.compound_inputs) 
                      + len(nml_doc.voltage_clamp_triples), 
                      directed=True)
    
    vprop_name = G.new_vp("string")
    vprop_type = G.new_vp("string")
    vprop_size = G.new_vp("double")
    eprop_name = G.new_ep("string")
    eprop_type = G.new_ep("string")
    eprop_width = G.new_ep("double")
    eprop_color = G.new_ep("vector<double>")
    eprop_dash = G.new_ep("vector<double>")

    try:
        # Create vertices using population properties
        pop_map = {}
        pop_type_stats = {}
        pop_states = {}
        for pop in nml_doc.networks[0].populations:
            v1 = G.add_vertex()
            vprop_name[v1] = pop.id
            vertex_type = get_Vprefix(pop.id)
            pop_type = get_pop_type(pop.id)
            vprop_type[v1] = pop_type
            size = float(pop.size)
            vprop_size[v1] = np.log1p(size) * 2  # Logarithmic scaling for better visibility
            pop_map[pop.id] = v1
            pop_states[vertex_type] = pop_states.get(vertex_type, 0) + 1
            pop_type_stats[pop_type] = pop_type_stats.get(pop_type, 0) + 1
        logging.info(f"Detected {len(pop_map)} pop vertices from {base_name} status : %s", pop_type_stats)
        logging.info(f"pop_states: %s", pop_states)

        group_keys = []
        for v in G.vertices():
            pop_id = vprop_name[v]
            pop_type = get_pop_type(pop.id)
            if vprop_name[v] != pop_id:
                ilist_id = vprop_name[v]
                input_type = get_input_type(ilist_id)
                if isinstance(ilist_id, str): 
                    gkey = ilist_id.replace(f"{input_type}_", "")
                else:
                    gkey = pop_id
                gkey = pop_id
                group_keys.append(gkey)
            
        unique_groups = sorted(dict.fromkeys(group_keys).keys())
        group_pos = {g: i for i, g in enumerate(unique_groups)}
        vprop_group = G.new_vp("int")
        for v, gkey in zip(G.vertices(),group_keys):
            vprop_group[v] = group_pos.get(gkey, 0)

        # Add edges with different types
        edge_count = {'continuous': 0, 'electrical': 0}
        edge_type = G.new_ep("string")  # New property for edge types
        edge_weight = G.new_ep("double")

        # 
        state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(overlap=True))
        gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
        
        tree, prop, vprop = gt.get_hierarchy_tree(state)
        ecount = tree.num_edges()
        vcount = tree.num_vertices()
        print(f"Tree has {vcount} vertices and {ecount} edges")
        
        levels = state.get_levels() # Get the hierarchy levels
        print(f"Detected {len(levels)} hierarchy levels")
        for s in levels:
            print(s)
            if s.get_N() == 1:
                break
        b = levels[0].get_blocks() # Get block structure from highest level
        Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices())
        logging.info(f"Vprefixs: {Vprefixs}")

        for Syn_proj in nml_doc.networks[0].continuous_projections:
            src = Syn_proj.presynaptic_population
            tgt = Syn_proj.postsynaptic_population
            Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
            if (src in pop_map and tgt in pop_map
                and hasattr(Syn_proj, 'continuous_connection_instance_ws')
                and len(Syn_proj.continuous_connection_instance_ws) > 0):
                Vprefix_pre = get_Vprefix(src)
                Vprefix_post = get_Vprefix(tgt)
                prob = p_intra if Vprefix_pre == Vprefix_post else p_inter
                if np.random.rand() < prob:
                    e1 = G.add_edge(pop_map[src], pop_map[tgt])
                    # eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]
                    edge_type[e1] = 'Proj'
                    eprop_type[e1] = 'continuous'
                    edge_weight[e1] = Syn_w
                    eprop_width[e1] = Syn_w
                    eprop_dash[e1] = []
                    edge_count['continuous'] += 1 
        logging.info(f"Added {edge_count['continuous']} Syn_proj edges")

        for elect_proj in nml_doc.networks[0].electrical_projections:
            src = elect_proj.presynaptic_population
            tgt = elect_proj.postsynaptic_population
            elect_w = elect_proj.electrical_connection_instance_ws[0].weight
            
            if (src in pop_map and tgt in pop_map
                and hasattr(elect_proj, 'electrical_connection_instance_ws') 
                and len(elect_proj.electrical_connection_instance_ws) > 0) :
                Vprefix_pre = get_Vprefix(src)
                Vprefix_post = get_Vprefix(tgt)
                prob = p_intra if Vprefix_pre == Vprefix_post else p_inter
                if np.random.rand() < prob:
                    e2 = G.add_edge(pop_map[src], pop_map[tgt])
                    # eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]
                    edge_type[e2] = 'Proj'
                    edge_weight[e2] = elect_w
                    eprop_type[e2] = "electrical"
                    eprop_width[e2] = elect_w
                    eprop_dash[e2] = [0.2, 0.2]
                    edge_count['electrical'] += 1 
        logging.info(f"Added {edge_count['electrical']} elect_proj edges")

        V_intra = 0
        V_inter = 0
        R_intra = 0
        R_inter = 0
        L_intra = 0
        L_inter = 0
        for e in G.edges():
            pre = vprop_name[e.source()]
            post = vprop_name[e.target()]
            if get_Vprefix(pre) == get_Vprefix(post):
                V_intra += 1
            else:
                V_inter += 1
            if get_Region(pre) == get_Region(post):
                R_intra += 1
            else:
                R_inter += 1
            if get_layer(pre) == get_layer(post):
                L_intra += 1
            else:
                L_inter += 1
        logging.info(f"Intra Vprefix-edges: {V_intra}, Inter Vprefix-edges: {V_inter}")
        logging.info(f"Intra Region-edges: {R_intra}, Inter Region-edges: {R_inter}")
        logging.info(f"Intra layer-edges: {L_intra}, Inter layer-edges: {L_inter}")
        #--------------------------------------------------------------------#
        input_map = {}
        input_type_stats = {}
        input_stat = {}
        if hasattr(nml_doc, 'pulse_generators') and nml_doc.pulse_generators:
            for pg in nml_doc.pulse_generators:
                v2 = G.add_vertex()
                vprop_name[v2] = pg.id
                vertex_type = get_gen_type(pg.id)
                pg_type = get_input_type(pg.id)
                vprop_type[v2] = pg_type  # Use consistent type identifier
                vprop_size[v2] = np.log1p(len(pg.id))
                input_map[pg.id] = v2
                input_stat[vertex_type] = input_stat.get(vertex_type, 0) + 1
                input_type_stats[pg_type] = input_type_stats.get(pg_type, 0) + 1
        if hasattr(nml_doc, 'compound_inputs') and nml_doc.compound_inputs:
            for ci in nml_doc.compound_inputs:
                v3 = G.add_vertex()
                vprop_name[v3] = ci.id
                vertex_type = get_gen_type(ci.id)
                ci_type = get_input_type(ci.id)
                vprop_type[v3] = ci_type  # Use consistent type identifier
                vprop_size[v3] = np.log1p(len(ci.id))
                input_map[ci.id] = v3
                input_stat[vertex_type] = input_stat.get(vertex_type, 0) + 1
                input_type_stats[ci_type] = input_type_stats.get(ci_type, 0) + 1  
        
        if hasattr(nml_doc, 'voltage_clamp_triples') and nml_doc.voltage_clamp_triples:
            for vc in nml_doc.voltage_clamp_triples:
                v4 = G.add_vertex()
                vprop_name[v4] = vc.id
                vertex_type = get_gen_type(vc.id)
                vc_type = get_input_type(vc.id)
                vprop_type[v4] = vc_type
                vprop_size[v4] = np.log1p(len(vc.id))
                input_map[vc.id] = v4
                input_stat[vertex_type] = input_stat.get(vertex_type, 0) + 1
                input_type_stats[vc_type] = input_type_stats.get(vc_type, 0) + 1
        logging.info(f"Detected {len(input_map)} input vertices type breakdown : %s", input_type_stats)
        logging.info(f"input_stat: %s", input_stat)

        # Process input edges
        input_edges = {'exc_input': 0, 'inh_input': 0}
        destination_stats = {}
        total_input_edges = 0

        if hasattr(nml_doc.networks[0], 'input_lists') and nml_doc.networks[0].input_lists:
            for ilist in nml_doc.networks[0].input_lists:
                src = ilist.component
                tgt = ilist.populations
                # inputW = ilist.input_ws if isinstance(ilist.input_ws, list) else [ilist.input_ws] if hasattr(ilist, 'input_ws') else []
                if src not in input_map:
                    logging.debug(f"Skipping input_list with unknown component: {src}")
                    continue
                if tgt not in pop_map:
                    logging.debug(f"Skipping input_list targeting unknown population: {tgt}")
                    continue
                inputs = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
                for input_item in inputs:
                    try:
                        destination = getattr(input_item, 'destination', None) 
                        destination_stats[destination] = destination_stats.get(destination, 0) + 1
                        input_w = getattr(input_item, 'weight', 1.0)
                        if destination == 'AMPA_NMDA':
                            e3 = G.add_edge(input_map[src], pop_map[tgt])
                            total_input_edges += 1
                            eprop_name[e3] = destination
                            eprop_type[e3] = "AMPA_NMDA"    
                            edge_type[e3] = "exc_input"
                            input_map[destination] = e3
                            input_edges['exc_input'] += 1
                            edge_weight[e3] = input_w
                            eprop_width[e3] = input_w
                            # eprop_color[e3] = [0.0, 0.8, 0.25, 0.35]
                        elif destination == 'GABA':
                            e4 = G.add_edge(input_map[src], pop_map[tgt])
                            total_input_edges += 1
                            eprop_name[e4] = destination
                            eprop_type[e4] = "GABA"
                            edge_type[e4] = "inh_input"
                            input_map[destination] = e4
                            input_edges['inh_input'] += 1
                            edge_weight[e4] = input_w
                            eprop_width[e4] = input_w
                            # eprop_color[e4] = [0.0, 0.8, 0.25, 0.35]
                        elif destination == 'GapJ':
                            if get_input_type(src) == 'exc':
                                e5 = G.add_edge(input_map[src], pop_map[tgt])
                                total_input_edges += 1
                                eprop_name[e5] = destination
                                eprop_type[e5] = "GapJ"
                                edge_type[e5] = "exc_input"
                                input_map[destination] = e5
                                input_edges['exc_input'] += 1
                                edge_weight[e5] = input_w
                                eprop_width[e5] = input_w
                                # eprop_color[e5] = [0.0, 0.8, 0.25, 0.35]
                            else:
                                e6 = G.add_edge(input_map[src], pop_map[tgt])
                                total_input_edges += 1
                                eprop_name[e6] = destination
                                eprop_type[e6] = "GapJ"
                                edge_type[e6] = "inh_input"
                                input_map[destination] = e6
                                input_edges['inh_input'] += 1
                                edge_weight[e6] = input_w
                                eprop_width[e6] = input_w
                                # eprop_color[e6] = [0.0, 0.8, 0.25, 0.35]
                    except Exception as ex:
                        logging.warning(f"Failed to add input edge from {src} to {tgt}: {ex}")
        # Log detailed statistics
        logging.info("Destination breakdown: %s", destination_stats)
        logging.info("Detected %d input edges: %d exc | %d inh", total_input_edges, input_edges['exc_input'], input_edges['inh_input'])
        #--------------------------------------------------------------------#
        main_vertices = [v for v in G.vertices() if vprop_name[v] in pop_map.keys()]
        main_edges = [e for e in G.edges() if e.source() in main_vertices and e.target() in main_vertices]
        G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices, efilt=lambda e: e in main_edges)
        logging.info("Number of main vertices: %d", G_main.num_vertices())
        logging.info("Number of main edges: %d", G_main.num_edges())
        
        
        input_vfilter = G.new_vertex_property("bool")
        input_vfilter.a = False
        input_vertices = [v for v in G.vertices() if vprop_name[v] in input_map.keys()]
        for v in input_vertices:
            input_vfilter[v] = True
            for w in v.out_neighbors():
                input_vfilter[w] = True
            for w in v.in_neighbors():
                input_vfilter[w] = True
                
        # Create input edge filter property map  
        input_efilter = G.new_edge_property("bool")
        input_efilter.a = False
        for e in G.edges():
            if input_vfilter[e.source()] and input_vfilter[e.target()]:
                if vprop_name[e.source()] in input_map.keys() or vprop_name[e.target()] in input_map.keys():
                    input_efilter[e] = True
        
        G_ilist = gt.GraphView(G, vfilt=input_vfilter, efilt=input_efilter)
        logging.info("Number of input vertices: %d", G_ilist.num_vertices())
        logging.info("Number of input edges: %d", G_ilist.num_edges())
        
        
        full_vertices = [v for v in G.vertices()]
        full_edges = [e for e in G.edges()]
        G = gt.GraphView(G, vfilt=lambda v: v in full_vertices, efilt=lambda e: e in full_edges)
        logging.info("Number of total vertices: %d", G.num_vertices())
        logging.info("Number of total edges: %d", G.num_edges())
        state.print_summary()
        #--------------------------------------------------------------------#
        # Create proper position property map
        pos = G.new_vp("vector<double>")
        G.vp["pos"] = pos
        pos = gt.sfdp_layout(G, pos=pos, groups=vprop_group, C=4.0, K=1.0, p=2.0, gamma=0.1, theta=0.6, max_iter=1000, mu=2, weighted_coarse=True)
        
        # Animation creation
        #############################################################################################################################
        prev_exc_state = G.new_vp("vector<double>")
        curr_exc_state = G.new_vp("vector<double>")
        prev_inh_state = G.new_vp("vector<double>")
        curr_inh_state = G.new_vp("vector<double>")
        exc_transmited = G.new_vp("bool")
        inh_transmited = G.new_vp("bool")
        exc_refractory = G.new_vp("bool")
        inh_refractory = G.new_vp("bool")
        w = gt.max_cardinality_matching(G, edges=True, heuristic=True, brute_force=True)
        def create_graph_tool_animation(G, pos, state, output_file, vertex_fill_color=None,
                                        vertex_color=None, vertex_size=None, edge_color=None, 
                                        edge_pen_width=None, frames=10, mode="graph_draw", **kwargs):
            fixed_pos = gt.sfdp_layout(G, cooling_step=0.99)
            res = gt.max_independent_vertex_set(G)
            frame_files = []
            for i in range(frames):
                progress = i / frames
                exc_pop_vertex = [v for v in G.vertices() if vprop_name[v] == pop.id and vprop_type[v] == 'exc']
                inh_pop_vertex = [v for v in G.vertices() if vprop_name[v] == pop.id and vprop_type[v] == 'inh']
                exc_input_vertex = [v for v in G.vertices() if vprop_name[v] != pop.id and vprop_type[v] == 'exc']
                inh_input_vertex = [v for v in G.vertices() if vprop_name[v] != pop.id and vprop_type[v] == 'inh']
                exc_inactive_pop = int(progress * len(exc_pop_vertex))  
                inh_inactive_pop = int(progress * len(inh_pop_vertex))
                exc_active_pop = int(progress * 0.7 * len(exc_pop_vertex)) 
                inh_active_pop = int(progress * 0.6 * len(inh_pop_vertex)) 
                exc_refrac_pop = int(progress * 0.3 * len(exc_pop_vertex))
                inh_refrac_pop = int(progress * 0.4 * len(inh_pop_vertex))
                exc_inactive_input = int(progress * len(exc_input_vertex))
                inh_inactive_input = int(progress * len(inh_input_vertex))
                exc_active_input = int(progress * 0.7 * len(exc_input_vertex))
                inh_active_input = int(progress * 0.6 * len(inh_input_vertex))
                exc_refrac_input = int(progress * 0.3 * len(exc_input_vertex))
                inh_refrac_input = int(progress * 0.4 * len(inh_input_vertex))

                for idx, v in enumerate(G.vertices()):
                    prev_exc_state[v] = list(matplotlib.cm.viridis(0.1))[:4] # exc_S
                    prev_inh_state[v] = list(matplotlib.cm.viridis_r(0.1))[:4] # inh_S
                    curr_exc_state[v] = prev_exc_state[v]
                    curr_inh_state[v] = prev_inh_state[v]
                    exc_refractory.a = False
                    exc_transmited.a = False
                    inh_refractory.a = False
                    inh_transmited.a = False
                    vertex_type = vprop_type[v]
                    if vertex_type == 'exc':
                        if idx < exc_refrac_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.5))[:4] # exc_R  # Refractory state
                            exc_refractory[v] = True
                        elif idx < exc_active_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.8))[:4] # exc_I  # Active state
                            exc_transmited[v] = True
                        elif idx < exc_inactive_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.3))[:4] # exc_S  # Inactive state
                        elif idx < exc_refrac_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis_r(0.6))[:5]
                            exc_refractory[v] = True
                        elif idx < exc_active_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis_r(0.9))[:5]
                            exc_transmited[v] = True
                        elif idx < exc_inactive_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.4))[:5]

                    elif vertex_type == 'inh':
                        if idx < inh_refrac_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.5))[:4] # inh_R
                            inh_refractory[v] = True
                        elif idx < inh_active_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.8))[:4] # inh_I
                            inh_transmited[v] = True
                        elif idx < inh_inactive_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.3))[:4] # inh_S
                        elif idx < inh_refrac_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.4))[:3]
                            inh_refractory[v] = True
                        elif idx < inh_active_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.7))[:3]
                            inh_transmited[v] = True
                        elif idx < inh_inactive_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.2))[:3]
                    else:
                        curr_exc_state[v] = prev_exc_state[v]
                        curr_inh_state[v] = prev_inh_state[v]
                
                ee_edges = []
                ii_edges = []
                ei_edges = []
                ie_edges = []
                input_ee_edges = []
                input_ii_edges = []
                for idx, e in enumerate(G.edges()):
                    if idx < len(ei_edges):
                        current_edge = ei_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(ei_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(ei_edges[idx])
                            ei_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*len(ei_edges)), .05, ei_edges)
                    elif idx < len(ie_edges):
                        current_edge = ie_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(ie_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(ie_edges[idx])
                            ie_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*len(ie_edges)), .05, ie_edges)
                    elif idx < len(ee_edges):
                        current_edge = ee_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(ee_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(ee_edges[idx])
                            ee_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal(len(exc_pop_vertex)/(2*len(ee_edges)), .05, ee_edges)
                    elif idx < len(ii_edges):
                        current_edge = ii_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(ii_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(ii_edges[idx])
                            ii_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal(len(inh_pop_vertex)/(2*len(ii_edges)), .05, ii_edges)
                    elif idx < len(input_ee_edges):
                        current_edge = input_ee_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(input_ee_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(input_ee_edges[idx])
                            input_ee_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal(len(exc_input_vertex)/(2*len(input_ee_edges)), .05, input_ee_edges)
                    elif idx < len(input_ii_edges):
                        current_edge = input_ii_edges[idx]
                        e = list(current_edge)
                        s1, t1 = e
                        t2 = G.vertex(random.randint(0, int(input_ii_edges[idx].target()) + 1))
                        if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and s1 != t2 and t1.out_degree() > 0 and t2 not in s1.out_neighbors()): 
                            G.remove_edge(input_ii_edges[idx])
                            input_ii_edges[i] = G.add_edge(s1, t2)
                        eprop_color[e] = np.random.normal(len(inh_input_vertex)/(2*len(input_ii_edges)), .05, input_ii_edges)
                    else:
                        eprop_color[e] = np.random.normal(G.num_vertices()/(2*G.num_edges()), .05, G.num_edges())
                
                cnorm = matplotlib.colors.Normalize(vmin=-abs(edge_weight.fa).max(), vmax=abs(edge_weight.fa).max())
                frame_file = f"graph_draw/{base_name}/{base_name}_frame_{i:03d}.png"
                if vertex_fill_color is None:
                    if hasattr(state, 's'):
                        state_prop = state.s
                    elif hasattr(state, 'b'):
                        state_prop = state.b
                    else:
                        state_prop = G.vertex_index
                    computed_vertex_fill_color = gt.perfect_prop_hash([state_prop])[0]
                else:
                    computed_vertex_fill_color = vertex_fill_color
                computed_vertex_shape = computed_vertex_fill_color
                gt.graph_draw(
                    G, pos=fixed_pos,
                    vertex_fill_color=computed_vertex_fill_color,
                    vertex_shape=computed_vertex_shape,
                    vertex_color=res,
                    # vertex_color=curr_exc_state if vprop_type == "exc" else curr_inh_state,
                    edge_color=eprop_color,
                    edge_pen_width= w.t(lambda x: x + 1),
                    ecnorm=cnorm,
                    ecmap=matplotlib.cm.autumn,
                    bg_color=[0.98, 0.98, 0.98, 1],
                    output=frame_file,
                    output_size=(800, 800)  # Fixed size for animation frames
                )
                frame_files.append(frame_file)
            # Create animation using ImageMagick
            cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_file]
            subprocess.run(cmd, check=True)
            for f in frame_files:
                if os.path.exists(f):
                    os.remove(f)
            gc.collect()
        
        #############################################################################################################################
        deg_main = G_main.degree_property_map("in")
        deg_main.a = 9 * (np.sqrt(deg_main.a) * 0.9 + 0.9)
        ebet_main = gt.betweenness(G_main)[1]
        # Safe normalization with protection against empty arrays and division by zero
        if len(ebet_main.a) > 0 and ebet_main.a.max() > 0:
            ebet_main.a /= ebet_main.a.max() / 18.
        else:
            # If no edges or all betweenness values are zero, set to default value
            ebet_main.a[:] = 1.0
        eorder_main = ebet_main.copy()
        eorder_main.a *= -1
        pos_main = gt.sfdp_layout(G_main, cooling_step=0.99)
        control_main = G_main.new_edge_property("vector<double>")
        for e in G_main.edges():
            d_main = np.sqrt(sum((pos_main[e.source()].a - pos_main[e.target()].a) ** 2)) / 5
            control_main[e] = [0.3, d_main, 0.7, d_main]
        
        gt.remove_parallel_edges(G_main)
        gt.graph_draw(G_main, pos_main, vertex_anchor=0, 
                      edge_color=ebet_main, edge_pen_width=w.t(lambda x: x + 1), 
                       vorder=deg_main, vertex_fill_color=deg_main, vertex_shape=deg_main,
                      eorder=eorder_main,  edge_control_points=control_main,  
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"graph_draw/{base_name}/{base_name}_main.png")
        create_graph_tool_animation(G_main, pos_main, state, output_file=f"graph_draw/{base_name}/{base_name}_main.gif",
                                    vertex_fill_color=deg_main, edge_color=ebet_main,edge_pen_width=w.t(lambda x: x + 1)
                                )
        
        deg_ilist = G_ilist.degree_property_map("out")
        deg_ilist.a = 3 * (np.sqrt(deg_ilist.a) * 0.3 + 0.3)
        ebet_ilist = gt.betweenness(G_ilist)[1]
        # Safe normalization with protection against empty arrays and division by zero
        if len(ebet_ilist.a) > 0 and ebet_ilist.a.max() > 0:
            ebet_ilist.a /= ebet_ilist.a.max() / 5.
        else:
            # If no edges or all betweenness values are zero, set to default value
            ebet_ilist.a[:] = 1.0
        eorder_ilist = ebet_ilist.copy()
        eorder_ilist.a *= -1
        pos_ilist = gt.sfdp_layout(G_ilist, cooling_step=0.99)
        control_ilist = G_ilist.new_edge_property("vector<double>")
        for e in G_ilist.edges():
            d_ilist = np.sqrt(sum((pos_ilist[e.source()].a - pos_ilist[e.target()].a) ** 2)) / 5
            control_ilist[e] = [0.3, d_ilist, 0.7, d_ilist]
        
        gt.remove_parallel_edges(G_ilist)
        gt.graph_draw(G_ilist, pos_ilist, vertex_anchor=0, 
                      edge_color=ebet_ilist, edge_pen_width=w.t(lambda x: x + 1),   
                       vorder=deg_ilist, vertex_fill_color=deg_ilist, vertex_shape=deg_ilist,
                      eorder=eorder_ilist, edge_control_points=control_ilist, 
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"graph_draw/{base_name}/{base_name}_ilist.png")
        create_graph_tool_animation(G_ilist, pos_ilist, state, output_file=f"graph_draw/{base_name}/{base_name}_ilist.gif",
                                    vertex_fill_color=deg_ilist,  
                                    edge_color=ebet_ilist, edge_pen_width=w.t(lambda x: x + 1), 
                                )
        deg = G.degree_property_map("total")
        deg.a = 7 * (np.sqrt(deg.a) * 0.7 + 0.7)
        ebet = gt.betweenness(G)[1]
        ebet.a /= ebet.a.max() / 15.
        eorder = ebet.copy()
        eorder.a *= -1
        pos = gt.sfdp_layout(G, cooling_step=0.99)
        control = G.new_edge_property("vector<double>")
        for e in G.edges():
            d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
            control[e] = [0.3, d, 0.7, d]
        
        gt.remove_parallel_edges(G)
        gt.graph_draw(G, pos, vertex_anchor=0, 
                       edge_color=ebet, edge_pen_width=w.t(lambda x: x + 1),  
                      vorder=deg, vertex_shape=deg,  vertex_fill_color=deg, 
                      eorder=eorder, edge_control_points=control,   
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"graph_draw/{base_name}/{base_name}_graph.png")
        
        create_graph_tool_animation(G, pos, state, output_file=f"graph_draw/{base_name}/{base_name}_graph.gif",
                                    vertex_fill_color=deg, edge_color=ebet, edge_pen_width=w.t(lambda x: x + 1), 
                                )
        #############################################################################################################################
        def create_blockstate_animation(bg, pos, block_state, output_file, vertex_fill_color=None,
                                      vcmap=None, edge_color=None, edge_pen_width=None):
            """
            Create animation for BlockState objects using block membership (.b) instead of spin states (.s)
            """
            import subprocess
            b = block_state.get_blocks() # Get block membership
            b_hashed = gt.perfect_prop_hash([b])[0]
            
            frame_files = []
            num_frames = 20  # Number of animation frames
            
            for i in range(num_frames):
                # Create cycling effect by shifting block colors
                color_shift = i * (256 // num_frames)
                shifted_colors = (b_hashed.fa + color_shift) % 256
                
                frame_file = f"graph_draw/{base_name}/{base_name}_block_frame_{i:03d}.png"
                
                # Create vertex color property map with shifted colors
                vcolor = bg.new_vp("int")
                vcolor.fa = shifted_colors
                
                gt.graph_draw(
                    bg, pos=pos,
                    vertex_fill_color=vcolor,
                    vertex_shape=vcolor,
                    vertex_size=vertex_fill_color if vertex_fill_color is not None else bg.vertex_index,
                    vertex_text=bg.vertex_index,  # Add vertex indices as text labels
                    vertex_font_size=12,  # Set readable font size
                    edge_color=edge_color if edge_color is not None else bg.edge_index,
                    edge_pen_width=edge_pen_width,
                    bg_color=[0.98, 0.98, 0.98, 1],
                    output=frame_file,
                    output_size=(800, 800)
                )
                frame_files.append(frame_file)
            
            # Create animation using ImageMagick
            cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_file]
            subprocess.run(cmd, check=True)
            
            # Clean up frame files
            for f in frame_files:
                os.remove(f)
        state_condesne = gt.BlockState(G)
        gt.mcmc_equilibrate(state_condesne, wait=1000)
        b = state_condesne.get_blocks()
        b = gt.perfect_prop_hash([b])[0]
        # Fix ResourceWarning by ensuring file is properly closed
        block_output = f"graph_draw/{base_name}/{base_name}_block.png"
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=b, vertex_shape=b,
                    bg_color=[0.98, 0.98, 0.98, 1],output=block_output)
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_condesne, output_file=f"graph_draw/{base_name}/{base_name}_block.gif",
                                    vertex_fill_color=b, vcmap=matplotlib.cm.viridis,
                                    edge_color=w,edge_pen_width=w.t(lambda x: x + 1)
                                )
        # Compute condensation graph without inheriting positions to avoid geometric invalidity
        condensation_result = gt.condensation_graph(G, b)
        bg = condensation_result[0]
        vcount = condensation_result[1] 
        ecount = condensation_result[2]
        
        pos = gt.sfdp_layout(bg, cooling_step=0.99)
        condense_output = f"graph_draw/{base_name}/{base_name}_block_condesne.png"
        draw_args = {
            'pos': pos,
            'vertex_fill_color': bg.vertex_index,
            'vertex_shape': bg.vertex_index,
            'vertex_size': gt.prop_to_size(vcount, mi=40, ma=100),
            'vertex_text': bg.vertex_index,  # Add vertex indices as text labels
            'vertex_font_size': 12,  # Set readable font size
            'edge_pen_width': gt.prop_to_size(ecount, mi=2, ma=10),
            'bg_color': [0.98, 0.98, 0.98, 1],
            'output': condense_output
        }
            
        gt.graph_draw(bg, **draw_args)
        # Create animation for condensation graph using BlockState-specific animation
        create_blockstate_animation(bg, pos, state_condesne, output_file=f"graph_draw/{base_name}/{base_name}_block_condesne.gif",
                                  vertex_fill_color=gt.prop_to_size(vcount, mi=40, ma=100),
                                  edge_color=bg.edge_index,
                                  edge_pen_width=gt.prop_to_size(ecount, mi=2, ma=10))
        
        #############################################################################################################################
        return 

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        if 'pop' in locals():
            print(f"Population causing error: {pop.id}")
        raise

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nml_net_files = [
        os.path.join(script_dir, "net_files/TC2CT.net.nml"),
        # os.path.join(script_dir, "net_files/TC2PT.net.nml"),
        os.path.join(script_dir, "net_files/TC2IT4_IT2CT.net.nml"),
        os.path.join(script_dir, "net_files/TC2IT2PTCT.net.nml"),
        # os.path.join(script_dir, "net_files/C2T_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/iC_max.net.nml"),
        # os.path.join(script_dir, "net_files/T2C_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/iT_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/loop_iT_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/loop_L1.net.nml"),
        # os.path.join(script_dir, "net_files/loop_L23.net.nml"),
        # os.path.join(script_dir, "net_files/loop_L4.net.nml"),
        # os.path.join(script_dir, "net_files/loop_L5.net.nml"),
        # os.path.join(script_dir, "net_files/loop_L6.net.nml"),
        # os.path.join(script_dir, "net_files/max_CTC_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M1b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M1_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2aM1aS1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1bM1bM2b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2M1S1_max_plus.net.nml")
    ]
    for nml_net_file in nml_net_files:
        if not os.path.exists(nml_net_file):
            print(f"Warning: File '{nml_net_file}' does not exist. Skipping...")
            continue
        base_name = os.path.basename(nml_net_file).split(".")[0]
        visualize_network(nml_net_file, p_intra=0.9, p_inter=0.1,base_name=base_name)


