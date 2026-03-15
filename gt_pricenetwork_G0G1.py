# %%
import os
import sys
import time
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ['MPLBACKEND'] = 'Agg'
from cv2 import merge
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from collections import defaultdict
from tqdm import tqdm
import json
import gc
import random
import scipy
from graph_tool.all import *
import graph_tool.all as gt
import subprocess
from pyneuroml.pynml import read_neuroml2_file
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from tqdm import tqdm
import time
import logging
import gc

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

layer_list = {"L1","L23","L4","L5","L6","thalamus"}
Region_list = {"M2a","M2b","M1a","M1b","S1a","S1b"}
gen_list = {"PG", "VC", "ComInp"}
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
        if len(parts) == 1 and parts[0] in thalamus:
            return parts[0]
        elif len(parts) == 2 and parts[1] in thalamus:
            return parts[1]
        elif len(parts) >= 6 and parts[0] not in Region_list:
            return '_'.join(parts[1:3])
        elif len(parts) >= 7 and parts[0] in Region_list:
            return '_'.join(parts[2:4]) 
        elif len(parts) >= 7 and parts[0] not in Region_list:
            return '_'.join(parts[2:5])
        return pop_id
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
        return pop_id
    except Exception as e:
        logging.error(f"Error in get_layer with pop_id {pop_id}: {e}")
        return "unknown"

def get_Region(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return None
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
def visualize_network(nml_net_file, V_intra, V_inter, L_intra, L_inter, base_name):
    # Create output directory if it doesn't exist
    output_dir = f"gt_plots/{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    nml_doc = read_neuroml2_file(nml_net_file)
    G0 = complete_graph(len(nml_doc.networks[0].populations), directed=True)
    G1 = complete_graph(len(nml_doc.pulse_generators), directed=True)
    
    pop_vprop_name = G0.new_vp("string")
    pop_vprop_type = G0.new_vp("string")
    pop_vprop_size = G0.new_vp("double")
    pop_eprop_type = G0.new_ep("string")
    pop_eprop_width = G0.new_ep("double")
    edge_weight = G0.new_ep("int", vals=np.random.normal(G0.num_vertices()/(2*G0.num_edges()), .05, G0.num_edges()))
    input_vprop_name = G1.new_vp("string")
    input_vprop_type = G1.new_vp("string")
    input_vprop_size = G1.new_vp("double")
    input_eprop_type = G1.new_ep("string")
    input_eprop_width = G1.new_ep("double")
    input_edge_weight = G1.new_ep("int", vals=np.random.normal(G1.num_vertices()/(2*G1.num_edges()), .05, G1.num_edges()))
    
    try:
        # Step 1: Create vertices and edges using population properties
        pop_map = {}  # Maps population IDs to vertex indices
        pop_type_stats = {}
        pop_states = {}
        for pop in nml_doc.networks[0].populations:
            v_pop = G0.add_vertex()
            pop_vprop_name[v_pop] = pop.id
            vertex_type = get_Vprefix(pop.id)
            pop_type = get_pop_type(pop.id)
            pop_vprop_type[v_pop] = pop_type
            size = float(pop.size) if hasattr(pop, 'size') else 1.0
            pop_vprop_size[v_pop] = np.log1p(size) * 2
            pop_map[pop.id] = v_pop  # Map population ID to vertex
            pop_states[vertex_type] = pop_states.get(vertex_type, 0) + 1
            pop_type_stats[pop_type] = pop_type_stats.get(pop_type, 0) + 1
        logging.info(f"Detected {len(pop_map)} population vertices from {base_name} network")
        logging.info(f"pop_states: %s", pop_states)
        
        # Step 2: create input vertices and edges
        input_vertices = {}
        input_type_stats = {}
        input_states = {}
        for pg in nml_doc.pulse_generators:
            v_pg = G1.add_vertex()
            input_vprop_name[v_pg] = pg.id
            pg_type = get_input_type(pg.id)
            input_vprop_type[v_pg] = pg_type  # Use consistent type identifier
            input_vprop_size[v_pg] = 2
            input_vertices[pg.id] = v_pg
            vertex_type = get_gen_type(pg.id)
            input_type_stats[pg_type] = input_type_stats.get(pg_type, 0) + 1
            input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
        logging.info("Detected %d input vertices type breakdown : %s", len(input_vertices), input_type_stats)

        # Step 3: Community detection first
        G = price_network(G0.num_vertices() + G1.num_vertices(), directed=True)
        
        state = minimize_nested_blockmodel_dl(G, state_args=dict(overlap=True))
        gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
        
        edge_count = {'continuous': 0, 'electrical': 0}
        edge_type_prop = G0.new_ep("string")  # New property for edge types
        for Syn_proj in nml_doc.networks[0].continuous_projections:
            pre = Syn_proj.presynaptic_population
            post = Syn_proj.postsynaptic_population
            if (pre in pop_map and post in pop_map 
                and hasattr(Syn_proj, 'continuous_connection_instance_ws') 
                and len(Syn_proj.continuous_connection_instance_ws) > 0) :
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)
                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)
                inter_module = (Vprefix_pre != Vprefix_post)
                if inter_module:
                    if (v_pass and l_pass and np.random.rand() < 0.9):  
                        Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                        e1 = G0.add_edge(pop_map[pre], pop_map[post])
                        if v_pass and l_pass:
                            pop_eprop_width[e1] = 0.2  
                        else:
                            pop_eprop_width[e1] = 0.9 
                        
                        edge_weight[e1] = Syn_w
                        edge_type_prop[e1] = 'Proj'
                        pop_eprop_type[e1] = 'continuous'
                        edge_count['continuous'] += 1
                    elif (v_pass and l_pass):
                        Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                        e1 = G0.add_edge(pop_map[pre], pop_map[post])
                
                        if v_pass and l_pass:
                            pop_eprop_width[e1] = 0.2  
                        else:
                            pop_eprop_width[e1] = 0.9

                        edge_weight[e1] = Syn_w
                        pop_eprop_width[e1] = Syn_w
                        edge_type_prop[e1] = 'Proj'
                        pop_eprop_type[e1] = 'continuous'
                        edge_count['continuous'] += 1
        
        print(f"Added {edge_count['continuous']} Syn_proj edges")

        # Add electrical projections with community-aware edge weights
        for elect_proj in nml_doc.networks[0].electrical_projections:
            pre = elect_proj.presynaptic_population
            post = elect_proj.postsynaptic_population
            elect_w = elect_proj.electrical_connection_instance_ws[0].weight
            if (pre in pop_map and post in pop_map
                and hasattr(elect_proj, 'electrical_connection_instance_ws') 
                and len(elect_proj.electrical_connection_instance_ws) > 0) :
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)
                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)
                inter_module = (Vprefix_pre != Vprefix_post) or (layer_pre != layer_post)
                if inter_module:
                    if (v_pass and l_pass and np.random.rand() < 0.9):  
                        
                        e2 = G0.add_edge(pop_map[elect_proj.presynaptic_population], 
                                        pop_map[elect_proj.postsynaptic_population])
                        
                        if v_pass and l_pass:
                            pop_eprop_width[e2] = 0.25 
                        else:
                            pop_eprop_width[e2] = 0.12

                        edge_type_prop[e2] = 'Proj'
                        edge_weight[e2] = elect_w
                        pop_eprop_width[e2] = elect_w
                        pop_eprop_type[e2] = 'electrical'
                        edge_count['electrical'] += 1
                elif (v_pass and l_pass):
                    e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                pop_map[elect_proj.postsynaptic_population])
                    if v_pass and l_pass:
                        pop_eprop_width[e2] = 0.25 
                    else:
                        pop_eprop_width[e2] = 0.12
                    edge_weight[e2] = elect_w
                    edge_type_prop[e2] = 'Proj'
                    pop_eprop_type[e2] = 'electrical'
                    edge_count['electrical'] += 1     
        print(f"Added {edge_count['electrical']} elect_proj edges")

        intra = 0
        inter = 0
        for e_pop in G0.edges():
            pre_pop = pop_vprop_name[e_pop.source()]
            post_pop = pop_vprop_name[e_pop.target()]
            if get_Vprefix(pre_pop) == get_Vprefix(post_pop):
                intra += 1
            else:
                inter += 1
        print(f"Intra Vprefix-edges: {intra}, Inter Vprefix-edges: {inter}")
        
        # input edges
        input_edges = {'GABA': 0, 'AMPA_NMDA': 0}
        input_edge_type_prop = G1.new_ep("string")
        total_input_edges = 0
        destination_stats = {}
        for ilist in nml_doc.networks[0].input_lists:
            pre = ilist.component  # This is the pulse generator (e.g., "inh_PG_nRT")
            post = ilist.populations  # This is the target population (e.g., "nRT")
            inputs = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
            
            if pre in input_vertices and post in pop_map:
                for input_item in inputs:
                    try:
                        destination = input_item.destination if hasattr(input_item, 'destination') else None
                        destination_stats[destination] = destination_stats.get(destination, 0) + 1
                        if destination == 'AMPA_NMDA':
                            e3 = G1.add_edge(input_vertices[pre], pop_map[post])
                            input_w = input_item.weight if hasattr(input_item, 'weight') else 1.0
                            input_eprop_width[e3] = input_w
                            input_eprop_type[e3] = "AMPA_NMDA"    
                            input_edge_type_prop[e3] = "AMPA_NMDA"
                            input_edges['AMPA_NMDA'] += 1
                            # input_edge_weight[e3] = input_w
                            total_input_edges += 1
                        elif destination == 'GABA':
                            e4 = G1.add_edge(input_vertices[pre], pop_map[post])
                            input_eprop_width[e4] = 0.1
                            input_eprop_type[e4] = "GABA"
                            input_edge_type_prop[e4] = "GABA"
                            input_edges['GABA'] += 1
                            input_w = input_item.weight if hasattr(input_item, 'weight') else 1.0
                            input_edge_weight[e4] = input_w
                            total_input_edges += 1
                    except Exception as ex:
                        logging.warning(f"Failed to add input edge from {pre} to {post}: {ex}")
                    
        logging.info("Destination breakdown: %s", destination_stats)
        logging.info("Detected %d input edges: %d GABA | %d AMPA_NMDA", total_input_edges, input_edges['GABA'], input_edges['AMPA_NMDA'])

        tree, prop, vprop = gt.get_hierarchy_tree(state)
        ecount = tree.num_edges()
        vcount = tree.num_vertices()
        print(f"Tree has {vcount} vertices and {ecount} edges")
        levels = state.get_levels() # Get the hierarchy levels
        print(f"Detected {len(levels)} hierarchy levels")
        b = levels[0].get_blocks() # Get block structure from highest level
        Vprefixs = set(get_Vprefix(pop_vprop_name[v_pop]) for v_pop in G0.vertices())
        logging.info(f"Vprefixs: {Vprefixs}")

        group_keys = []
        for v_pop in G0.vertices():
            pop_id = pop_vprop_name[v_pop]
            pop_type = get_pop_type(pop_id)
            gkey = pop_id
            group_keys.append(gkey)
        for v_pg in G1.vertices():    
            pg_id = input_vprop_name[v_pg]
            input_type = get_input_type(pg_id)
            if input_vprop_type[v_pg] == 'exc':
                gkey = 'exc'
            elif input_vprop_type[v_pg] == 'inh':
                gkey = 'inh'
            group_keys.append(gkey)
        unique_groups = sorted(dict.fromkeys(group_keys).keys())
        group_pos = {g: i for i, g in enumerate(unique_groups)}
        vprop_group = G.new_vp("int")
        for v, gkey in zip(G.vertices(),group_keys):
            vprop_group[v] = group_pos.get(gkey, 0)

        pop_vertices = [v for v in G0.vertices() if pop_vprop_type[v] == pop_type]
        pop_edges = [e for e in G0.edges() if e.source() in pop_vertices and e.target() in pop_vertices]
        G0 = gt.GraphView(G0, vfilt=lambda v: v in pop_vertices, efilt=lambda e: e in pop_edges)
        # if G0.num_vertices() > 0:
        #     state_ndc_pop = gt.minimize_nested_blockmodel_dl(G0, state_args=dict(deg_corr=False))
        #     state_dc_pop  = gt.minimize_nested_blockmodel_dl(G0, state_args=dict(deg_corr=True))
        #     # logging.info("Number of edges: %d", G.num_edges())
        #     logging.info("Block counts (ndc): %s", np.unique(state_ndc_pop.get_levels()[0].get_blocks().a, return_counts=True))
        #     logging.info("Block counts (dc): %s", np.unique(state_dc_pop.get_levels()[0].get_blocks().a, return_counts=True))
        #     pop_ndc_b = state_ndc_pop.get_levels()[0].get_blocks()
        #     pop_dc_b = state_dc_pop.get_levels()[0].get_blocks()
        # else:
        #     logging.warning("G0 has no vertices, skipping blockmodel analysis")
        #     # Create empty property maps to prevent downstream errors
        #     pop_ndc_b = G0.new_vp("int")
        #     pop_dc_b = G0.new_vp("int")
        # pop_comm_ndc = len(set(pop_ndc_b.a)) if G0.num_vertices() > 0 else 0
        # pop_comm_dc = len(set(pop_dc_b.a)) if G0.num_vertices() > 0 else 0
        # logging.info(f"pop_vertices_ndc: {pop_comm_ndc} communities | pop_vertices_dc: {pop_comm_dc} communities")

        input_vertices = [v for v in G1.vertices() if input_vprop_type[v] == pg_type]
        ilist_edges = [e for e in G1.edges() if e.source() in input_vertices]
        G1 = gt.GraphView(G1, vfilt=lambda v: v in input_vertices, efilt=lambda e: e in ilist_edges)
        # if G1.num_vertices() > 0:
        #     state_ndc_ilist = gt.minimize_nested_blockmodel_dl(G1, state_args=dict(deg_corr=False))
        #     state_dc_ilist = gt.minimize_nested_blockmodel_dl(G1, state_args=dict(deg_corr=True))
        #     logging.info("Non-degree-corrected ILIST:%s", state_ndc_ilist.entropy())
        #     logging.info("Degree-corrected ILIST:%s", state_dc_ilist.entropy())
        #     logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_ilist.entropy() - state_dc_ilist.entropy())
        #     ilist_ndc_b = state_ndc_ilist.get_levels()[0].get_blocks()
        #     ilist_dc_b = state_dc_ilist.get_levels()[0].get_blocks()
        #     logging.info("ilist Block counts (ndc): %s", np.unique(ilist_ndc_b.a, return_counts=True))
        #     logging.info("ilist Block counts (dc): %s", np.unique(ilist_dc_b.a, return_counts=True))
        # else:
        #     logging.warning("G1 has no vertices, skipping blockmodel analysis")
        #     # Create empty property maps to prevent downstream errors
        #     ilist_ndc_b = G1.new_vp("int")
        #     ilist_dc_b = G1.new_vp("int")
        # ilist_comm_ndc = len(set(ilist_ndc_b.a)) if G1.num_vertices() > 0 else 0
        # ilist_comm_dc = len(set(ilist_dc_b.a)) if G1.num_vertices() > 0 else 0
        # logging.info(f"input_vertices_ndc: {ilist_comm_ndc} communities | input_vertices_dc: {ilist_comm_dc} communities")
        
        # Step 4: Create proper position property map
        pos = G.new_vp("vector<double>")
        G.vp["pos"] = pos

        # Step 5: Fine-tune layout with sfdp/frl/al/rtl/pl/rl using proper position initialization
        pos = gt.sfdp_layout(G, groups=vprop_group)
        # Helper: validate/repair pos to avoid degenerate transforms
        def _ensure_pos_valid(pos_prop, G, min_jitter=1e-3):
            try:
                # build numpy array of shape (n,2)
                arr = np.vstack([np.asarray(pos_prop[v]) for v in G.vertices()]) if G.num_vertices() > 0 else np.zeros((0,2))
            except Exception:
                arr = None
            
            if arr is None or arr.size == 0 or not np.isfinite(arr).all():
                new_pos = gt.sfdp_layout(G)
                for v in G.vertices():
                    pos_prop[v] = list(new_pos[v])
                return pos_prop
            
            # if all x or all y identical -> add tiny jitter
            if np.ptp(arr[:,0]) == 0.0:
                jitter = np.random.uniform(-min_jitter, min_jitter, size=arr.shape[0])
                for i, v in enumerate(G.vertices()):
                    p = list(pos_prop[v])
                    p[0] = float(p[0]) + float(jitter[i])
                    pos_prop[v] = p
            if np.ptp(arr[:,1]) == 0.0:
                jitter = np.random.uniform(-min_jitter, min_jitter, size=arr.shape[0])
                for i, v in enumerate(G.vertices()):
                    p = list(pos_prop[v])
                    p[1] = float(p[1]) + float(jitter[i])
                    pos_prop[v] = p
            for i, v in enumerate(G.vertices()):
                pos_prop[v][0] += np.random.normal(0, 1e-6)
                pos_prop[v][1] += np.random.normal(0, 1e-6)
            # final sanity: if still degenerate, recompute sfdp
            arr = np.vstack([np.asarray(pos_prop[v]) for v in G.vertices()]) if G.num_vertices() > 0 else np.zeros((0,2))
            if np.ptp(arr[:,0]) == 0.0 or np.ptp(arr[:,1]) == 0.0:
                new_pos = gt.sfdp_layout(G)
                for v in G.vertices():
                    pos_prop[v] = list(new_pos[v])
            return pos_prop

        # Safe graph draw wrapper with fallback to recompute layout
        def safe_graph_draw(*args, pos_prop=None, Gref=None, retries=1, **kwargs):
            if pos_prop is not None and Gref is not None:
                _ensure_pos_valid(pos_prop, Gref)
            try:
                gt.graph_draw(*args, pos=pos_prop, **kwargs) if 'pos' in gt.graph_draw.__code__.co_varnames else gt.graph_draw(*args, **kwargs)
                return True
            except RuntimeError as re:
                logging.warning(f"graph_draw failed with RuntimeError: {re}. Attempting fallback layout.")
                try:
                    new_pos = gt.sfdp_layout(Gref)
                    if Gref is not None:
                        for v in Gref.vertices():
                            pos_prop[v] = list(new_pos[v])
                    gt.graph_draw(*args, pos=pos_prop, **kwargs)
                    return True
                except Exception as e2:
                    logging.error(f"Fallback graph_draw also failed: {e2}")
                    return False
            except Exception as e:
                logging.error(f"graph_draw unexpected error: {e}")
                return False
        
        # Step 6: Draw with community colors and edge types
        # #############################################################################################################################
        # def animate_hierarchy(state, output_gif, frames=5, sweeps_per_frame=50, **draw_kwargs):
        #     frame_files = []
        #     for i in range(frames):
        #         for j in range(sweeps_per_frame):
        #             state.multiflip_mcmc_sweep(niter=10)
        #         frame_file = f"hierarchy_frame_{i:03d}.png"
        #         state.draw(output=frame_file, **draw_kwargs)
        #         frame_files.append(frame_file)
        #     # Create GIF
        #     cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
        #     subprocess.run(cmd, check=True)
        #     # Clean up
        #     for f in frame_files:
        #         if os.path.exists(f):
        #             os.remove(f)
        #     gc.collect()
        
        # # state_hierarchy = gt.minimize_nested_blockmodel_dl(G,state_args=dict(recs=[edge_weight],rec_types=["real-exponential"]))
        # state_hierarchy = gt.NestedBlockState(G,base_type=gt.RankedBlockState, state_args=dict(eweight=gt.contract_parallel_edges(G)))
        # gt.mcmc_equilibrate(state_hierarchy, force_niter=100, mcmc_args=dict(niter=10))
        
        # gt.mcmc_equilibrate(state_hierarchy, wait=10, mcmc_args=dict(niter=10))
        # for i in range(10):
        #     for j in range(100):
        #         state_hierarchy.multiflip_mcmc_sweep(niter=10,beta=np.inf)
        # state_hierarchy.draw(pos=pos,
        #     # edge_color=gt.prop_to_size(edge_weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
        #     # eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1, log=True),
        #     # edge_gradient=[], 
        #     bg_color=[1, 1, 1, 1],output=f"gt_plots/{base_name}/{base_name}_hierarchy.png" ,empty_branches=False)
        # animate_hierarchy(state_hierarchy,output_gif=f"gt_plots/{base_name}/{base_name}_hierarchy.gif",frames=5,sweeps_per_frame=50,bg_color=[1, 1, 1, 1])

        # ############################################################################################################################
        # def animate_ghcp(state_ghcp, tpos, shape, cts, eprop_color, output_gif,frames=5, sweeps_per_frame=50, vertex_fill_color=None, 
        #                 vertex_size=None, **kwargs):
        #     frame_files = []
        #     G = state_ghcp.g
        #     # Get top-level block state
        #     top_level = state_ghcp.get_levels()[0]
        #     num_blocks = len(set(top_level.get_blocks().a))
        #     community_colors = [
        #         [
        #             0.5 + 0.5 * np.cos(2 * np.pi * i / max(num_blocks, 1)),
        #             0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 1/3)),
        #             0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 2/3)),
        #             0.8
        #         ]
        #         # for i in range(max(num_blocks, 1))
        #         for i in range(num_blocks)
        #     ]
        #     vprop_block_color = G.new_vp("vector<double>")
        #     # Get current block assignments from top level
            
        #     for i in range(frames):
        #         for j in range(sweeps_per_frame):
        #             state_ghcp.multiflip_mcmc_sweep(niter=10)
        #         blocks = top_level.get_blocks()
        #         for v in G.vertices():
        #             vprop_block_color[v] = community_colors[blocks[v] % len(community_colors)]

        #         frame_file = f"ghcp_frame_{i:03d}.png"
        #         gt.graph_draw(
        #             G, pos=tpos,
        #             vertex_shape=shape,
        #             edge_control_points=cts,
        #             edge_color=edge_weight,
        #             vertex_fill_color=vprop_block_color,
        #             vertex_size=vertex_size,
        #             bg_color=[1, 1, 1, 1],
        #             output=frame_file,
        #             **kwargs
        #         )
        #         frame_files.append(frame_file)
        #     # Create GIF
        #     cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
        #     subprocess.run(cmd, check=True)
        #     # Clean up
        #     for f in frame_files:
        #         if os.path.exists(f):
        #             os.remove(f)

        # g_ghcp = gt.GraphView(G, vfilt=gt.label_largest_component(G))
        # state_ghcp = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[edge_weight],rec_types=["discrete-binomial"]))
        # gt.mcmc_equilibrate(state_ghcp, wait=10, mcmc_args=dict(niter=10))
        # tree, prop_map, vprop = gt.get_hierarchy_tree(state_ghcp)
        # root = tree.vertex(tree.num_vertices() - 1, use_index=False)
        # tpos = gt.radial_tree_layout(tree, root, weighted=True)
        # cts = gt.get_hierarchy_control_points(G, tree, tpos)
        # shape = b.copy()
        # shape.a %= 14
        # gt.graph_draw(g_ghcp, pos=G.own_property(tpos), 
        #             vertex_fill_color=b, 
        #             vertex_shape=shape,edge_control_points=cts,edge_color=edge_weight,
        #             vertex_pen_width=2.5,vertex_anchor=0, 
        #             bg_color=[1, 1, 1, 1],output=f"gt_plots/{base_name}/{base_name}_ghcp.png")
        # for i in range(100):
        #     ret = state_ghcp.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        # state_ghcp.draw(edge_color=edge_weight.copy("double"), ecmap=matplotlib.cm.plasma,
        #                 eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1),
        #                 edge_gradient=[], bg_color=[1, 1, 1, 1],output=f"gt_plots/{base_name}/{base_name}_ghcp_wsbm.png")
        # animate_ghcp(state_ghcp, tpos, shape, cts, edge_weight,output_gif=f"gt_plots/{base_name}/{base_name}_ghcp.gif",frames=5, sweeps_per_frame=50)

        ##############################################################################################################################
        pre_exc_vprop_state = G.new_vp("int")
        curr_exc_vprop_state = G.new_vp("int")
        pre_inh_vprop_state = G.new_vp("int")
        curr_inh_vprop_state = G.new_vp("int")
        exc_transmited = G.new_vp("bool")
        inh_transmited = G.new_vp("bool")
        exc_refractory = G.new_vp("bool")
        inh_refractory = G.new_vp("bool")
        pre_eprop_state = G.new_ep("int")
        curr_eprop_state = G.new_ep("int")
        w = gt.max_cardinality_matching(G, edges=True, heuristic=True, brute_force=True)
        
        def create_graph_tool_animation(G, pos, state, output_file, vertex_fill_color=None,
                                        vertex_color=None, vertex_size=None, edge_color=None, 
                                        edge_pen_width=None, frames=20,
                                        mode="graph_draw", **kwargs):
            
            fixed_pos = gt.sfdp_layout(G, cooling_step=0.99)
            res = gt.max_independent_vertex_set(G)
            frame_files = []
            for i in range(frames):
                progress = i / frames
                exc_vertices = [v for v in G.vertices() if pop_vprop_type[v] == 'exc' and input_vprop_type[v] == 'exc']
                inh_vertices = [v for v in G.vertices() if pop_vprop_type[v] == 'inh' and input_vprop_type[v] == 'inh']
                input_exc_vertices = [v for v in G.vertices() if input_vprop_type[v] == 'exc']
                input_inh_vertices = [v for v in G.vertices() if input_vprop_type[v] == 'inh']

                exc_active_vertices = int(progress * (len(exc_vertices)+len(input_exc_vertices)) * 0.5)
                inh_active_vertices = int(progress * (len(inh_vertices)+len(input_inh_vertices)) * 0.5)
                exc_refractory_vertices = int(progress * G.num_vertices() * 0.2) 
                inh_refractory_vertices = int(progress * G.num_vertices() * 0.2) 
                exc_inactive_vertices = int(progress *(G.num_vertices() - exc_active_vertices - exc_refractory_vertices))
                inh_inactive_vertices = int(progress *(G.num_vertices() - inh_active_vertices - inh_refractory_vertices))
                for idx, v in enumerate(G.vertices()):
                    pre_exc_vprop_state[v] = 1
                    pre_inh_vprop_state[v] = 2
                    exc_refractory.a = False
                    exc_transmited.a = False
                    inh_refractory.a = False
                    inh_transmited.a = False
                    if idx < exc_refractory_vertices:
                        curr_exc_vprop_state[v] = 3 # R: Refractory state
                        exc_refractory[v] = True
                    elif idx < inh_refractory_vertices:
                        curr_inh_vprop_state[v] = 3
                        inh_refractory[v] = True
                    elif idx < exc_active_vertices:
                        curr_exc_vprop_state[v] = 5 # I: Active state
                        exc_transmited[v] = True
                    elif idx < inh_active_vertices:
                        curr_inh_vprop_state[v] = 5
                        inh_transmited[v] = True
                    elif idx < exc_inactive_vertices:
                        curr_exc_vprop_state[v] = 1 # S: Inactive state
                    elif idx < inh_inactive_vertices:
                        curr_inh_vprop_state[v] = 1
                    else:
                        curr_exc_vprop_state[v] = pre_exc_vprop_state[v]
                        curr_inh_vprop_state[v] = pre_inh_vprop_state[v]
                
                ee = [e for e in G.edges() if pop_vprop_type[e.source()] == 'exc' and pop_vprop_type[e.target()] == 'exc']
                ei = [e for e in G.edges() if pop_vprop_type[e.source()] == 'exc' and pop_vprop_type[e.target()] == 'inh']
                ie = [e for e in G.edges() if pop_vprop_type[e.source()] == 'inh' and pop_vprop_type[e.target()] == 'exc']
                ii = [e for e in G.edges() if pop_vprop_type[e.source()] == 'inh' and pop_vprop_type[e.target()] == 'inh']
                input_ee = [e for e in G.edges() if input_vprop_type[e.source()] == 'exc']
                input_ii = [e for e in G.edges() if input_vprop_type[e.source()] == 'inh']
                
                ee_edges = int(progress * len(ee) * 0.5)
                ii_edges = int(progress * len(ii) * 0.5)
                ei_edges = int(progress * len(ei) * 0.6)
                ie_edges = int(progress * len(ie) * 0.7)
                input_ee_edges = int (progress * len(input_ee) * 0.5)
                input_ii_edges = int (progress * len(input_ii) * 0.5)
                
                for idx, e in enumerate(G.edges()):
                    pre_eprop_state[e] = 0.5
                    if idx < ei_edges:
                        curr_eprop_state[e] = 1   
                    elif idx < ie_edges:
                        curr_eprop_state[e] = 1   
                    elif idx < ee_edges:
                        curr_eprop_state[e] = 5 
                    elif idx < ii_edges:
                        curr_eprop_state[e] = 3
                    elif idx < input_ee_edges:
                        curr_eprop_state[e] = 5
                    elif idx < input_ii_edges:
                        curr_eprop_state[e] = 3
                    else:
                        curr_eprop_state[e] = pre_eprop_state[e]
                cnorm = matplotlib.colors.Normalize(vmin=-abs(edge_weight.fa).max(), vmax=abs(input_edge_weight.fa).max())     
                frame_file = f"frame_{i:03d}.png"
                gt.graph_draw(
                    G, pos=fixed_pos,
                    vertex_fill_color=res,
                    vertex_shape=gt.perfect_prop_hash([state.s])[0],
                    vertex_color=curr_exc_vprop_state if pop_vprop_type == "exc" and input_vprop_type == "exc" else curr_inh_vprop_state,
                    edge_color=curr_eprop_state,
                    edge_pen_width=w.t(lambda x: 0.2*x + 1),
                    ecnorm=cnorm,
                    bg_color=[1, 1, 1, 1],
                    output=frame_file,
                    ecmap=matplotlib.cm.winter,
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
        gt.remove_parallel_edges(G)
        f = np.eye(4) * 0.1
        state_graph = gt.PottsGlauberState(G, f)
        ret_graph = state_graph.iterate_async(niter=1000 * G.num_vertices())
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_anchor=0, 
                      vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0], 
                      vertex_shape=gt.perfect_prop_hash([state_graph.s])[0], 
                      edge_color=w, edge_pen_width=w.t(lambda x: 0.2*x + 1),
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[1, 1, 1, 1], output=f"gt_plots/{base_name}/{base_name}_graph.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_graph, output_file=f"gt_plots/{base_name}/{base_name}_graph_basic.gif",
                                    vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0], 
                                    edge_color=w,edge_pen_width=w.t(lambda x: 0.2*x + 1)
                                )
        kcore = gt.kcore_decomposition(gt.GraphView(G, vfilt=gt.label_largest_component(G)))
        state_kcore = gt.NormalState(G, sigma=0.001, w=-100)
        ret_kcore = state_kcore.iterate_sync(niter=1000)
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=state_kcore.s, 
                      vertex_shape=state_kcore.s,output_size=(800, 800),
                      edge_color=w, edge_pen_width=w.t(lambda x: 0.2*x + 1), ecmap=matplotlib.cm.tab20c,
                      bg_color=[1, 1, 1, 1], output=f"gt_plots/{base_name}/{base_name}_kcore.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_kcore, 
                                    output_file=f"gt_plots/{base_name}/{base_name}_kcore_basic.gif",
                                    vertex_fill_color=state_kcore.s, 
                                    vertex_shape=state_kcore.s,# vcapmap=matplotlib.cm.coolwarm,
                                    edge_color=w, edge_pen_width=w.t(lambda x: 0.2*x + 1), ecmap=matplotlib.cm.tab20c,
                                )
        try:
            similarity = gt.vertex_similarity(GraphView(G, reversed=True),"inv-log-weight")
            color = G.new_vp("double")
            color.a = similarity[0].a
            state_sim = gt.CIsingGlauberState(G, beta=.2)
            ret_sim = state_sim.iterate_async(niter=1000 * G.num_vertices())
            gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=state_sim.s,# vertex_text=G.vertex_index,
                          vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],output_size=(800, 800),
                          edge_color=w, edge_pen_width=w.t(lambda x: 0.2*x + 1), ecmap=matplotlib.cm.Set3,
                          bg_color=[1, 1, 1, 1], output=f"gt_plots/{base_name}/{base_name}_similarity.png")
            
            create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99),state_sim, 
                                        vertex_fill_color=state_sim.s, 
                                        vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],
                                        vcmap=matplotlib.cm.magma,
                                        output_file=f"gt_plots/{base_name}/{base_name}_similarity_basic.gif",
                                        edge_color=w, edge_pen_width=w.t(lambda x: 0.2*x + 1), ecmap=matplotlib.cm.Set3,
                                    )
        except Exception as e:
            logging.info(f"[WARNING] Failed to calculate vertex similarity: {e}")        
        #############################################################################################################################
        G = price_network(G.num_vertices())
        deg = G.degree_property_map("in")
        deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
        ebet = gt.betweenness(G)[1]

        gt.graphviz_draw(G, pos=gt.sfdp_layout(G, cooling_step=0.99), maxiter=100, ratio="compress", overlap=False, layout="sfdp",
                        vcolor=deg, vorder=vprop_group, elen=10, vcmap=matplotlib.cm.gist_heat,
                        ecolor=ebet, eorder=edge_weight, output=f"gt_plots/{base_name}/{base_name}_graphviz.png")
        
        #############################################################################################################################
        metrics = {}
        centrality_metrics = {
            'pr': 'pagerank',
            'bt': 'betweenness',
            'V': 'eigenvector',
            'katz': 'katz',
            'hitsX': 'hits_authority',
            'hitsY': 'hits_hub',
            't': 'eigentrust',
            'tt': 'trust_transitivity',
            'c': 'closeness',
        }
        try:
            G.save(f"{base_name}.gt")
            logging.info(f"Saved graph file: {base_name}.gt")
        except Exception as _e:
            logging.warning(f"Could not save graph to {base_name}.gt: {_e}")
        # Compute metrics once (in-memory) and convert returned numpy arrays to graph-tool vertex properties
        try:
            from metrics.generator import compute_and_save_metrics
        except ImportError:
            import importlib.util
            spec_path = os.path.join(os.getcwd(), "metrics_analysis_project", "src", "metrics", "generator.py")
            # spec = importlib.util.spec_from_file_location("metrics.generator", spec_path)
            if os.path.exists(spec_path):
                spec = importlib.util.spec_from_file_location("metrics.generator", spec_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    compute_and_save_metrics = getattr(mod, "compute_and_save_metrics")
                else:
                    raise ImportError("Cannot load metrics.generator from spec")
            else:
                raise

        out_dir = os.path.join(os.getcwd(), "metrics_out")
        try:
            # Force single thread for metrics computation with additional safety measures
            metrics_dict, npz_path, csv_path = compute_and_save_metrics(
                G, out_dir=out_dir, prefix=base_name, normalize=True, nthreads=1, save_files=True)
            logging.info(f"Computed metrics once (saved: {npz_path is not None})")
            gc.collect()  # Force garbage collection after metrics computation
        except Exception as e:
            logging.warning(f"Failed to compute metrics via generator: {e}")
            metrics_dict = {}
        
        # Convert metrics dictionary to vertex properties
        for k, arr in metrics_dict.items():
            try:
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.size != G.num_vertices():
                    tmp = np.full(G.num_vertices(), np.nan, dtype=float)
                    tmp[:min(arr_np.size, G.num_vertices())] = arr_np[:min(arr_np.size, G.num_vertices())]
                    arr_np = tmp
                vp = G.new_vp("double")
                vp.a = arr_np
                metrics[k] = vp
            except Exception as e:
                logging.warning(f"Failed to convert metric {k} to vertex_property; filling NaNs: {e}")
                vp = G.new_vp("double")
                vp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                metrics[k] = vp
                
        # Ensure expected metric keys exist (fill missing with NaNs)
        for key in set(centrality_metrics.values()):
            if key not in metrics:
                logging.info(f"Metric '{key}' missing — filling with NaNs")
                vp = G.new_vp("double")
                vp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                metrics[key] = vp

        # Process centrality metrics with enhanced error handling
        for metric_name, metric_key in tqdm(centrality_metrics.items(), desc="Centrality metrics"):
            logging.info(f"\n[INFO] Calculating and plotting {metric_name}...")
            try:
                t0 = time.time()
                metric = metrics[metric_key]
                for v in G.vertices():
                    metric[v] = metric[v]
                # Ensure metric is valid
                if not hasattr(metric, 'a') or len(metric.a) != G.num_vertices():
                    tmp = G.new_vp("double")
                    tmp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                    try:
                        arr = np.asarray(metric)
                        for i, v in enumerate(G.vertices()):
                            tmp[v] = float(arr[i]) if i < arr.size else np.nan
                        metric = tmp
                    except Exception:
                        metric = tmp
                _ensure_pos_valid(gt.sfdp_layout(G, cooling_step=0.99), G)
                # Try to draw with graph-tool (with fallbacks)
                output_file = f"gt_plots/{base_name}/{base_name}_graph_{metric_name}.png"
                draw_success = safe_graph_draw(
                    G, pos_prop=gt.sfdp_layout(G, cooling_step=0.99), Gref=G,
                    output=output_file,
                    vertex_fill_color=metric,
                    # vertex_size=gt.prop_to_size(metric, mi=5, ma=15),
                    vcmap=matplotlib.cm.twilight,ecmap=matplotlib.cm.tab20c,
                    bg_color=[1, 1, 1, 1]
                )
                if draw_success:
                    logging.info(f"[INFO] Saved {output_file} in {time.time() - t0:.2f} seconds")
                else:
                    logging.info(f"[ERROR] Failed to save {output_file}")
                
                # Also try graphviz draw as backup
                t1 = time.time()
                try:
                    pos = G.own_property(pos) if hasattr(G, 'own_property') else gt.sfdp_layout(G)
                    _ensure_pos_valid(pos, G)
                    gt.graphviz_draw(
                        G, pos, maxiter=100, ratio="compress", overlap=False,layout="sfdp",
                        vcolor=metric, vorder=metric, elen=10,
                        vcmap=matplotlib.cm.twilight,
                        ecolor=ebet, eorder=edge_weight, ecmap=matplotlib.cm.gist_heat,
                        output=f"gt_plots/{base_name}/{base_name}_graphviz_{metric_name}.png")
                    logging.info(f"[INFO] Saved gt_plots/{base_name}/{base_name}_graphviz_{metric_name}.png in {time.time() - t1:.2f} seconds")
                except Exception as e:
                    logging.warning(f"graphviz_draw failed for {metric_name}: {e}")

            except Exception as e:
                logging.error(f"Failed to process metric {metric_name}: {e}")
                continue
        
        #############################################################################################################################
        return 

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        if 'pop' in locals():
            print(f"Population causing error: {pop.id}")
        raise



if __name__ == "__main__":
    nml_net_files = [
        # "TC2CT.net.nml" ,
        # "TC2PT.net.nml",
        # "TC2IT4_IT2CT.net.nml",
        "TC2IT2PTCT.net.nml",
        "max_CTC_plus.net.nml",
        "M1a_max_plus.net.nml",
        "M1_max_plus.net.nml",
        "M2_max_plus.net.nml",
        "M2M1S1_max_plus.net.nml",
        "S1bM1bM2b_max_plus.net.nml",
        "M2aM1aS1a_max_plus.net.nml",
    ]
    for nml_net_file in nml_net_files:
        base_name = nml_net_file.split('.')[0]
        visualize_network(nml_net_file, V_intra=0.5, V_inter=0.5,L_inter=0.5, L_intra=0.5, base_name=base_name)
        graph_types = ['graph', 'graphviz']
        metric_names = [
            'pr', 'bt', 'V', 'katz','hitsX', 'hitsY', 't', 'tt','c'
        ]
        file_paths = []
        for graph_type in graph_types:
            for metric in metric_names:
                file_name = f"gt_plots/{base_name}/{base_name}_{graph_type}_{metric}.png"
                file_paths.append(file_name)

        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"File '{file_path}' size: {os.path.getsize(file_path)/(1024*1024):.2f} MB")
            else:
                print(f"File '{file_path}' not found")

