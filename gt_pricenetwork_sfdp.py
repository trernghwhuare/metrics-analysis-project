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
from collections import defaultdict
from tqdm import tqdm
import xml.etree.ElementTree as ET
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
from tqdm import tqdm
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
    output_dir = f"gt_plots/{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    nml_doc = read_neuroml2_file(nml_net_file)
    G = price_network(len(nml_doc.networks[0].populations) + len(nml_doc.networks[0].input_lists), directed=True)
    vprop_name = G.new_vertex_property("string")
    vprop_type = G.new_vertex_property("string")
    vprop_size = G.new_vertex_property("double")
    eprop_name = G.new_edge_property("string")
    eprop_type = G.new_edge_property("string")
    eprop_width = G.new_edge_property("double")
    eprop_color = G.new_edge_property("vector<double>")
    eprop_dash = G.new_edge_property("vector<double>")

    try:
        # Step 1: Create vertices using population properties
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

        input_map = {}
        input_type_stats = {}
        input_states = {}
        if hasattr(nml_doc, 'pulse_generators') and nml_doc.pulse_generators:
            for pg in nml_doc.pulse_generators:
                v2 = G.add_vertex()
                vprop_name[v2] = pg.id
                vertex_type = get_gen_type(pg.id)
                pg_type = get_input_type(pg.id)
                vprop_type[v2] = pg_type  # Use consistent type identifier
                vprop_size[v2] = 2
                input_map[pg.id] = v2
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[pg_type] = input_type_stats.get(pg_type, 0) + 1
        if hasattr(nml_doc, 'compound_inputs') and nml_doc.compound_inputs:
            for ci in nml_doc.compound_inputs:
                v3 = G.add_vertex()
                vprop_name[v3] = ci.id
                vertex_type = get_gen_type(ci.id)
                ci_type = get_input_type(ci.id)
                vprop_type[v3] = ci_type  # Use consistent type identifier
                vprop_size[v3] = 2
                input_map[ci.id] = v3
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[ci_type] = input_type_stats.get(ci_type, 0) + 1  
        
        if hasattr(nml_doc, 'voltage_clamp_triples') and nml_doc.voltage_clamp_triples:
            for vc in nml_doc.voltage_clamp_triples:
                v4 = G.add_vertex()
                vprop_name[v4] = vc.id
                vertex_type = get_gen_type(vc.id)
                vc_type = get_input_type(vc.id)
                vprop_type[v4] = vc_type
                vprop_size[v4] = 2
                input_map[vc.id] = v4
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[vc_type] = input_type_stats.get(vc_type, 0) + 1
        logging.info(f"Detected {len(input_map)} input vertices type breakdown : %s", input_type_stats)
        logging.info(f"input_states: %s", input_states)

        # Step 2: Add edges with different types
        edge_count = {'continuous': 0, 'electrical': 0}
        edge_type = G.new_edge_property("string")  # New property for edge types
        edge_weight = G.new_edge_property("double")

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
                    eprop_color[e2] = [1.0, 0.0, 0.0, 0.4]  
                    edge_type[e2] = 'Proj'
                    edge_weight[e2] = elect_w
                    eprop_type[e2] = "electrical"
                    eprop_width[e2] = elect_w
                    eprop_dash[e2] = [0.2, 0.2]
                    edge_count['electrical'] += 1 
        logging.info(f"Added {edge_count['electrical']} elect_proj edges")

        intra = 0
        inter = 0
        for e in G.edges():
            pre = vprop_name[e.source()]
            post = vprop_name[e.target()]
            if get_Vprefix(pre) == get_Vprefix(post):
                intra += 1
            else:
                inter += 1
        print(f"Intra Vprefix-edges: {intra}, Inter Vprefix-edges: {inter}")

        # Process input edges
        input_edges = {'exc_input': 0, 'inh_input': 0}
        destination_stats = {}
        total_input_edges = 0

        if hasattr(nml_doc.networks[0], 'input_lists') and nml_doc.networks[0].input_lists:
            for ilist in nml_doc.networks[0].input_lists:
                src = ilist.component
                tgt = ilist.populations
                inputW = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
                if src not in input_map:
                    logging.debug(f"Skipping input_list with unknown component: {src}")
                    continue
                if tgt not in pop_map:
                    logging.debug(f"Skipping input_list targeting unknown population: {tgt}")
                    continue
                # inputs = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
                for input_item in inputW:
                    try:
                        destination = getattr(input_item, 'destination', None) 
                        destination_stats[destination] = destination_stats.get(destination, 0) + 1

                        if destination == 'AMPA_NMDA':
                            e3 = G.add_edge(input_map[src], pop_map[tgt])
                            total_input_edges += 1
                            eprop_name[e3] = destination
                            eprop_type[e3] = "AMPA_NMDA"    
                            edge_type[e3] = "exc_input"
                            input_map[destination] = e3
                            input_edges['exc_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e3] = input_w
                            eprop_width[e3] = 0.1
                            # eprop_color[e3] = [1.0, 0.5, 0.0, 0.5]
                        elif destination == 'GABA':
                            e4 = G.add_edge(input_map[src], pop_map[tgt])
                            total_input_edges += 1
                            eprop_name[e4] = destination
                            eprop_type[e4] = "GABA"
                            edge_type[e4] = "inh_input"
                            input_map[destination] = e4
                            input_edges['inh_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e4] = input_w
                            eprop_width[e4] = 0.1
                            # eprop_color[e4] = [1.0, 0.0, 0.0, 0.5]  # Reddish for inputs
                        elif destination == 'GapJ':
                            # For GapJ, we need to determine type from the pre-component
                            if get_input_type(pre) == 'exc':
                                e5 = G.add_edge(input_map[src], pop_map[tgt])
                                total_input_edges += 1
                                eprop_name[e5] = destination
                                eprop_type[e5] = "GapJ"
                                edge_type[e5] = "exc_input"
                                input_map[destination] = e5
                                input_edges['exc_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e5] = input_w
                            else:
                                e6 = G.add_edge(input_map[src], pop_map[tgt])
                                total_input_edges += 1
                                eprop_name[e6] = destination
                                eprop_type[e6] = "GapJ"
                                edge_type[e6] = "inh_input"
                                input_map[destination] = e6
                                input_edges['inh_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e6] = input_w
                    except Exception as ex:
                        logging.warning(f"Failed to add input edge from {src} to {tgt}: {ex}")
        # Log detailed statistics
        logging.info("Destination breakdown: %s", destination_stats)
        logging.info("Detected %d input edges: %d exc | %d inh", total_input_edges, input_edges['exc_input'], input_edges['inh_input'])

        # Step 3: Community detection first
        state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(overlap=True))
        gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
        
        tree, prop, vprop = gt.get_hierarchy_tree(state)
        ecount = tree.num_edges()
        vcount = tree.num_vertices()
        print(f"Tree has {vcount} vertices and {ecount} edges")
        
        levels = state.get_levels() # Get the hierarchy levels
        print(f"Detected {len(levels)} hierarchy levels")
        
        b = levels[0].get_blocks() # Get block structure from highest level
        Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices())
        logging.info(f"Vprefixs: {Vprefixs}")
        
        main_vertices = [v for v in G.vertices() if vprop_name[v] in pop_map.keys()]
        main_edges = [e for e in G.edges() if e.source() in main_vertices and e.target() in main_vertices]
        # G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices, efilt=lambda e: e in main_edges)
        # state_ndc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=False))
        # state_dc_main  = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=True))
        # logging.info("Block counts (ndc): %s", np.unique(state_ndc_main.get_levels()[0].get_blocks().a, return_counts=True))
        # logging.info("Block counts (dc): %s", np.unique(state_dc_main.get_levels()[0].get_blocks().a, return_counts=True))
        
        # main_ndc_b = state_ndc_main.get_levels()[0].get_blocks()
        # main_dc_b = state_dc_main.get_levels()[0].get_blocks()
        # main_comm_ndc = len(set(main_ndc_b.a))
        # main_comm_dc = len(set(main_dc_b.a))
        # logging.info(f"main_vertices_ndc: {main_comm_ndc} communities | main_vertices_dc: {main_comm_dc} communities")

        input_vertices = [v for v in G.vertices() if vprop_name[v] in input_map.keys()]
        ilist_edges = [e for e in G.edges() if e.source() in input_vertices or e.target() in input_vertices]
        # G_ilist = gt.GraphView(G, vfilt=lambda v: v in input_vertices, efilt=lambda e: e in ilist_edges)
        full_vertices = [v for v in G.vertices()]
        full_edges = [e for e in G.edges()]
        

        # Step 4: Create proper position property map
        pos = G.new_vertex_property("vector<double>")
        G.vp["pos"] = pos

        # Step 5: Fine-tune layout with sfdp/frl/al/rtl/pl/rl using proper position initialization
        pos = gt.sfdp_layout(G, pos=pos, C=4.0, K=1.0, p=2.0, gamma=0.1, theta=0.6, max_iter=1000, mu=2, weighted_coarse=True)
        # pos = gt.sfdp_layout(G, groups=vprop_group)
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
        #############################################################################################################################
        # def animate_hierarchy(state, output_gif, frames=5, swee_edgess_per_frame=50, **draw_kwargs):
        #     frame_files = []
        #     for i in range(frames):
        #         for j in range(swee_edgess_per_frame):
        #             state.multiflip_mcmc_swee_edges(niter=10)
        #         frame_file = f"gt_plots/{base_name}/{base_name}_hierarchy_frame_{i:03d}.png"
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

        # state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
        # gt.mcmc_equilibrate(state_ndc_full, wait=10, mcmc_args=dict(niter=10))
        # for i in range(10):
        #     for j in range(100):
        #         state_ndc_full.multiflip_mcmc_swee_edges(niter=10)
        # state_ndc_full.draw(
        #     bg_color=[0.98, 0.98, 0.98, 1],
        #     output=f"gt_plots/{base_name}/{base_name}_ndc_hierarchy.png",
        #     empty_branches=False,
        #     # Noneoverlapping=True,
        # )
        # animate_hierarchy(state_ndc_full, output_gif=f"gt_plots/{base_name}/{base_name}_ndc_hierarchy.gif",
        #                         frames=5,
        #                         swee_edgess_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
        
        # state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
        # gt.mcmc_equilibrate(state_dc_full, wait=10, mcmc_args=dict(niter=10))
        # for i in range(10):
        #     for j in range(100):
        #         state_dc_full.multiflip_mcmc_swee_edges(niter=10)
        # state_dc_full.draw(
        #     bg_color=[0.98, 0.98, 0.98, 1],
        #     output=f"gt_plots/{base_name}/{base_name}_dc_hierarchy.png",
        #     empty_branches=False,
        #     # Noneoverlapping=True,
        # )
        # animate_hierarchy(state_dc_full, output_gif=f"gt_plots/{base_name}/{base_name}_dc_hierarchy.gif",
        #                         frames=5,
        #                         swee_edgess_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
        
        # # state_hierarchy = gt.minimize_nested_blockmodel_dl(G,state_args=dict(recs=[edge_weight],rec_types=["real-exponential"]))
        # state_hierarchy = gt.NestedBlockState(G,base_type=gt.RankedBlockState, state_args=dict(eweight=gt.contract_parallel_edges(G)))
        # gt.mcmc_equilibrate(state_hierarchy, force_niter=100, mcmc_args=dict(niter=10))
        
        # gt.mcmc_equilibrate(state_hierarchy, wait=10, mcmc_args=dict(niter=10))
        # for i in range(10):
        #     for j in range(100):
        #         state_hierarchy.multiflip_mcmc_swee_edges(niter=10,beta=np.inf)
        # state_hierarchy.draw(pos=pos,
        #     # edge_color=gt.prop_to_size(edge_weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
        #     # eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1, log=True),
        #     # edge_gradient=[], 
        #     bg_color=[0.98, 0.98, 0.98, 1],output=f"gt_plots/{base_name}/{base_name}_hierarchy.png" ,empty_branches=False)
        # animate_hierarchy(state_hierarchy,output_gif=f"gt_plots/{base_name}/{base_name}_hierarchy.gif",frames=5,swee_edgess_per_frame=50,bg_color=[0.98, 0.98, 0.98, 1])

        
        # #############################################################################################################################
        # def animate_ghcp(state_ghcp, tpos, shape, cts, eprop_color, output_gif,frames=5, swee_edgess_per_frame=50, vertex_fill_color=None, 
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
        #     vprop_block_color = G.new_vertex_property("vector<double>")
        #     # Get current block assignments from top level
            
        #     for i in range(frames):
        #         for j in range(swee_edgess_per_frame):
        #             state_ghcp.multiflip_mcmc_swee_edges(niter=10)
        #         blocks = top_level.get_blocks()
        #         for v in G.vertices():
        #             vprop_block_color[v] = community_colors[blocks[v] % len(community_colors)]

        #         frame_file = f"gt_plots/{base_name}/{base_name}_ghcp_frame_{i:03d}.png"
        #         gt.graph_draw(
        #             G, pos=tpos,
        #             vertex_shape=shape,
        #             edge_control_points=cts,
        #             edge_color=eprop_color,
        #             vertex_fill_color=vprop_block_color,
        #             vertex_size=vertex_size,
        #             bg_color=[0.98, 0.98, 0.98, 1],
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
        #             vertex_shape=shape,edge_control_points=cts,edge_color=eprop_color,
        #             vertex_pen_width=2.5,vertex_anchor=0, 
        #             bg_color=[0.98, 0.98, 0.98, 1],output=f"gt_plots/{base_name}/{base_name}_ghcp.png")
        # for i in range(100):
        #     ret = state_ghcp.multiflip_mcmc_swee_edges(niter=10, beta=np.inf)
        # state_ghcp.draw(edge_color=edge_weight.copy("double"), ecmap=matplotlib.cm.PiYG,
        #                 eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1),
        #                 edge_gradient=[], bg_color=[0.98, 0.98, 0.98, 1],output=f"gt_plots/{base_name}/{base_name}_ghcp_wsbm.png")
        # animate_ghcp(state_ghcp, tpos, shape, cts, eprop_color,output_gif=f"gt_plots/{base_name}/{base_name}_ghcp.gif",frames=5, swee_edgess_per_frame=50)

        #############################################################################################################################
        prev_exc_state = G.new_vertex_property("vector<double>")
        curr_exc_state = G.new_vertex_property("vector<double>")
        prev_inh_state = G.new_vertex_property("vector<double>")
        curr_inh_state = G.new_vertex_property("vector<double>")
        exc_transmited = G.new_vertex_property("bool")
        inh_transmited = G.new_vertex_property("bool")
        exc_refractory = G.new_vertex_property("bool")
        inh_refractory = G.new_vertex_property("bool")
        w = gt.max_cardinality_matching(G, edges=True, heuristic=True, brute_force=True)
        # edge_weight is already created after graph replacement, so we don't need to create it again
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
                
                ee = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'exc' and vprop_type[e.target()] == 'exc']
                ei = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'exc' and vprop_type[e.target()] == 'inh']
                ie = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'inh' and vprop_type[e.target()] == 'exc']
                ii = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'inh' and vprop_type[e.target()] == 'inh']
                input_ee = [e for e in G.edges() if vprop_name[e.source()] != pop.id and vprop_type[e.source()] == 'exc']
                input_ii = [e for e in G.edges() if vprop_name[e.source()] != pop.id and vprop_type[e.source()] == 'inh']
                
                ee_edges = int(progress * len(ee) * 0.5)
                ii_edges = int(progress * len(ii) * 0.5)
                ei_edges = int(progress * len(ei) * 0.6)
                ie_edges = int(progress * len(ie) * 0.7)
                input_ee_edges = int (progress * len(input_ee) * 0.5)
                input_ii_edges = int (progress * len(input_ii) * 0.5)
                
                for idx, e in enumerate(G.edges()):
                    if idx < ei_edges:
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*ei_edges), .05, ei_edges)
                    elif idx < ie_edges:
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*ie_edges), .05, ie_edges)
                    elif idx < ee_edges:
                        eprop_color[e] = np.random.normal(len(exc_pop_vertex)/(2*ee_edges), .05, ee_edges)
                    elif idx < ii_edges:
                        eprop_color[e] = np.random.normal(len(inh_pop_vertex)/(2*ii_edges), .05, ii_edges)
                    elif idx < input_ee_edges:
                        eprop_color[e] = np.random.normal(len(exc_input_vertex)/(2*input_ee_edges), .05, input_ee_edges)
                    elif idx < input_ii_edges:
                        eprop_color[e] = np.random.normal(len(exc_input_vertex)/(2*input_ii_edges), .05, input_ii_edges)
                    else:
                        eprop_color[e] = np.random.normal(G.num_vertices()/(2*G.num_edges()), .05, G.num_edges())
                cnorm = matplotlib.colors.Normalize(vmin=-abs(edge_weight.fa).max(), vmax=abs(edge_weight.fa).max())
                frame_file = f"gt_plots/{base_name}/{base_name}_frame_{i:03d}.png"
                gt.graph_draw(
                    G, pos=fixed_pos,
                    vertex_fill_color=res,
                    vertex_shape=b,
                    vertex_color=curr_exc_state if vprop_type == "exc" else curr_inh_state,
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
        gt.remove_parallel_edges(G)
        f = np.eye(4) * 0.1
        state_graph = gt.PottsGlauberState(G, f)
        ret_graph = state_graph.iterate_async(niter=1000 * G.num_vertices())
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_anchor=0, 
                      vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0],  
                      vertex_shape=gt.perfect_prop_hash([state_graph.s])[0], 
                      edge_color=w, edge_pen_width=w.t(lambda x: x + 1),
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"gt_plots/{base_name}/{base_name}_graph.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_graph, output_file=f"gt_plots/{base_name}/{base_name}_graph_basic.gif",
                                    vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0], # vcmap=matplotlib.cm.viridis,
                                    edge_color=w,edge_pen_width=w.t(lambda x: x + 1)
                                )
        kcore = gt.kcore_decomposition(G)
        state_kcore = gt.NormalState(G, sigma=0.001, w=-100)
        ret_kcore = state_kcore.iterate_sync(niter=1000)
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=kcore.t(lambda x: x + 1), 
                      vertex_shape=state_kcore.s,output_size=(800, 800),
                      edge_color=w, edge_pen_width=w.t(lambda x: x + 1), ecmap=matplotlib.cm.viridis,
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"gt_plots/{base_name}/{base_name}_kcore.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_kcore, 
                                    output_file=f"gt_plots/{base_name}/{base_name}_kcore_basic.gif",
                                    vertex_fill_color=state_kcore.s, 
                                    vertex_shape=state_kcore.s,# vcapmap=matplotlib.cm.viridis,
                                    edge_color=w, edge_pen_width=w.t(lambda x: x + 1), ecmap=matplotlib.cm.Set1,
                                )
        try:
            similarity = gt.vertex_similarity(GraphView(G, reversed=True),"inv-log-weight")
            color = G.new_vp("double")
            color.a = similarity[0].a
            state_sim = gt.CIsingGlauberState(G, beta=.2)
            G = state_sim.g
            ret_sim = state_sim.iterate_async(niter=1000 * G.num_vertices())
            gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=state_sim.s,
                          vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],output_size=(800, 800),
                          edge_color=w, edge_pen_width=w.t(lambda x: x + 1), ecmap=matplotlib.cm.Set3,
                          bg_color=[0.98, 0.98, 0.98, 1], output=f"gt_plots/{base_name}/{base_name}_similarity.png")
            
            create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_sim, 
                                        vertex_fill_color=state_sim.s, 
                                        vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],
                                        output_file=f"gt_plots/{base_name}/{base_name}_similarity_basic.gif",
                                        edge_color=w, 
                                        edge_pen_width=w.t(lambda x: x + 1), ecmap=matplotlib.cm.Set3,
                                    )
        except Exception as e:
            logging.info(f"[WARNING] Failed to calculate vertex similarity: {e}")        
        
        #############################################################################################################################
        G = price_network(G.num_vertices())
        deg = G.degree_property_map("in")
        deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
        ebet = gt.betweenness(G)[1]
        gt.graphviz_draw(G, pos=gt.sfdp_layout(G, cooling_step=0.99), maxiter=100, ratio="compress", overlap=False, layout="sfdp",
                        vcolor=deg, vorder=deg, elen=10, vcmap=matplotlib.cm.gist_heat,
                        ecolor=ebet, eorder=ebet, output=f"gt_plots/{base_name}/{base_name}_graphviz.png")
        
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
                vp = G.new_vertex_property("double")
                vp.a = arr_np
                metrics[k] = vp
            except Exception as e:
                logging.warning(f"Failed to convert metric {k} to vertex_property; filling NaNs: {e}")
                vp = G.new_vertex_property("double")
                vp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                metrics[k] = vp
                
        # Ensure expected metric keys exist (fill missing with NaNs)
        for key in set(centrality_metrics.values()):
            if key not in metrics:
                logging.info(f"Metric '{key}' missing — filling with NaNs")
                vp = G.new_vertex_property("double")
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
                    tmp = G.new_vertex_property("double")
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
                    vertex_fill_color=metric, vertex_shape=metric,
                    # vertex_size=gt.prop_to_size(metric, mi=5, ma=15),
                    vcmap=matplotlib.cm.gist_heat,
                    bg_color=[0.98, 0.98, 0.98, 1]
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
                        vcmap=matplotlib.cm.gist_heat,
                        ecolor=ebet, eorder=ebet,
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
        "TC2CT.net.nml" ,
        "TC2PT.net.nml",
        "TC2IT4_IT2CT.net.nml",
        "TC2IT2PTCT.net.nml",
    ]
    for nml_net_file in nml_net_files:
        base_name = nml_net_file.split(".")[0]
        visualize_network(nml_net_file, p_intra=0.9, p_inter=0.1,base_name=base_name)
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

