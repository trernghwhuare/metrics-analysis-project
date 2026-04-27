# %%
import os
import sys
import time
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
import numpy as np
from pyneuroml.pynml import read_neuroml2_file
import gi
gi.require_version('Gtk', '3.0')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from graph_tool.all import *
import graph_tool.all as gt
from tqdm import tqdm
import time
import logging
import gc
from concurrent.futures import ThreadPoolExecutor
import subprocess
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

layer_list = {"L1","L23","L4","L5","L6","thalamus"}
Region_list = {"M2a","M2b","M1a","M1b","S1a","S1b"}
gen_list = {"PG","VC","ComInp"}

def get_pop_type(pop_id):
    if not isinstance(pop_id, str) or pop_id == "":
        return pop_id
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
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1 and parts[0] in thalamus:
            return parts[0]
        elif len(parts) == 2 and parts[0] in Region_list and parts[1] in thalamus:
            return parts[1]
        elif len(parts) >= 6 and parts[1] in layer_list:
            return '_'.join(parts[1:3])
        elif len(parts) >= 7 and parts[0] in Region_list and parts[2] in layer_list:
            return '_'.join(parts[2:4]) 
        elif len(parts) >= 7 and parts[1] in layer_list:
            return '_'.join(parts[2:5])
        return "unknown"
    except Exception as e:
        logging.error(f"Error in get_Vprefix with pop_id {pop_id}: {e}")
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

def get_layer(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
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

def get_input_type(ilist_id):
    if ilist_id.startswith('exc_'):
        return 'exc'
    elif ilist_id.startswith('inh_'):
        return 'inh'
    else:
        return 'default'

def get_gen_type(ilist_id):
    if '_PG_' in ilist_id:
        return 'pulse_generators'
    elif '_ComInp_' in ilist_id:
        return 'compound_inputs'
    elif '_VC_' in ilist_id:
        return 'voltage_clamp_triples'
    else:
        return 'unknown'

def visualize_network(nml_net_file, R_intra, R_inter, V_intra, V_inter,L_intra, L_inter, base_name):
    import os
    output_dir = f"metrics/{base_name}"
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
            # v1.population = pop
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
        
        # Step 2: Add edges with different types
        edge_count = {'continuous': 0, 'electrical': 0}
        edge_type = G.new_edge_property("string")  # New property for edge types
        edge_weight = G.new_edge_property("double")


        # Step 3: Community detection first
        state = gt.minimize_nested_blockmodel_dl(G)
        gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
        
        tree, prop, vprop = gt.get_hierarchy_tree(state)
        ecount = tree.num_edges()
        vcount = tree.num_vertices()
        # Get the hierarchy levels
        levels = state.get_levels()
        b = levels[0].get_blocks()
       
        logging.info(f"Tree has {vcount} vertices and {ecount} edges")
        logging.info(f"Detected {len(levels)} hierarchy levels")
        Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices())
        Regions = set(get_Region(vprop_name[v]) for v in G.vertices())
        logging.info(f"Regions: {Regions}, Vprefixs: {Vprefixs}")
        
        
        for Syn_proj in nml_doc.networks[0].continuous_projections:
            pre = Syn_proj.presynaptic_population
            post = Syn_proj.postsynaptic_population
            if (pre in pop_map and post in pop_map
                and hasattr(Syn_proj, 'continuous_connection_instance_ws')
                and len(Syn_proj.continuous_connection_instance_ws) > 0):
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)

                Region_pre = get_Region(pre)
                Region_post = get_Region(post)
                Rprob = R_intra if Region_pre == Region_post else R_inter
                r_pass = (np.random.rand() < Rprob)

                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)

                # Modified: Use AND instead of OR to make modular structure more apparent
                inter_module = (Vprefix_pre != Vprefix_post) or (Region_pre != Region_post) or (layer_pre != layer_post)
                if inter_module:
                    # For inter-module connections, use a much lower probability
                    if (r_pass and v_pass and l_pass and np.random.rand() < 0.5):  # Additional 10% sampling
                        Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                        e1 = G.add_edge(pop_map[pre], pop_map[post])
                        # source_block = b[e1.source()]
                        # target_block = b[e1.target()]
                    
                        if r_pass and v_pass:
                            eprop_width[e1] = 0.2   # if source_block != target_block else 0.3
                            eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                        elif r_pass:
                            # region-driven
                            eprop_width[e1] = 0.1  # if source_block != target_block else 0.15
                            eprop_color[e1] = [0.0, 0.35, 0.8, 0.65] 
                        else:
                            # Vprefix-driven
                            eprop_width[e1] = 0.9 
                            eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]

                        edge_weight[e1] = Syn_w
                        edge_type[e1] = 'proj'
                        eprop_type[e1] = 'continuous'
                        edge_count['continuous'] += 1 
                elif (r_pass and v_pass and l_pass):  # For intra-module connections
                    Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                    e1 = G.add_edge(pop_map[pre], pop_map[post])
                    # source_block = b[e1.source()]
                    # target_block = b[e1.target()]
                
                    if r_pass and v_pass:
                        eprop_width[e1] = Syn_w * 2  # if source_block != target_block else Syn_w * 3
                        eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                    elif r_pass:
                        # region-driven
                        eprop_width[e1] = Syn_w * 10  # if source_block != target_block else Syn_w * 15
                        eprop_color[e1] = [0.0, 0.35, 0.8, 0.65] 
                    else:
                        # Vprefix-driven
                        eprop_width[e1] = Syn_w * 5 
                        eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]

                    edge_weight[e1] = Syn_w
                    edge_type[e1] = 'Proj'
                    eprop_type[e1] = 'continuous'
                    edge_count['continuous'] += 1 
        logging.info(f"Detected {edge_count['continuous']} Syn_proj edges")

        for elect_proj in nml_doc.networks[0].electrical_projections:
            pre = elect_proj.presynaptic_population
            post = elect_proj.postsynaptic_population
            if (pre in pop_map and post in pop_map
                and hasattr(elect_proj, 'electrical_connection_instance_ws') 
                and len(elect_proj.electrical_connection_instance_ws) > 0) :
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)

                Region_pre = get_Region(pre)
                Region_post = get_Region(post)
                Rprob = R_intra if Region_pre == Region_post else R_inter
                r_pass = (np.random.rand() < Rprob)

                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)
                
                inter_module = (Vprefix_pre != Vprefix_post) or (Region_pre != Region_post) or (layer_pre != layer_post)
                if inter_module:
                    # For inter-module connections, use a much lower probability
                    if (r_pass and v_pass and l_pass and np.random.rand() < 0.1):  # Additional 10% sampling
                        elect_w = elect_proj.electrical_connection_instance_ws[0].weight
                        e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                    pop_map[elect_proj.postsynaptic_population])
                        # source_block = b[e2.source()]
                        # target_block = b[e2.target()]
                        
                        if r_pass and v_pass:
                            eprop_width[e2] = elect_w * 5 # if source_block != target_block else elect_w * 4
                            eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                        elif r_pass:
                            eprop_width[e2] = elect_w * 8  # if source_block != target_block else elect_w * 6
                            eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
                        else:
                            eprop_width[e2] = elect_w * 10
                            eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]

                        edge_weight[e2] = elect_w
                        edge_type[e2] = 'Proj'
                        eprop_type[e2] = 'electrical'
                        edge_count['electrical'] += 1 
                elif (r_pass and v_pass and l_pass):  # For intra-module connections
                    elect_w = elect_proj.electrical_connection_instance_ws[0].weight
                    e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                pop_map[elect_proj.postsynaptic_population])
                    # source_block = b[e2.source()]
                    # target_block = b[e2.target()]
                    
                    if r_pass and v_pass:
                        eprop_width[e2] = elect_w * 5 # if source_block != target_block else elect_w * 4
                        eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                    elif r_pass:
                        eprop_width[e2] = elect_w * 8  # if source_block != target_block else elect_w * 6
                        eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
                    else:
                        eprop_width[e2] = elect_w * 10
                        eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]

                    edge_weight[e2] = elect_w
                    edge_type[e2] = 'Proj'
                    eprop_type[e2] = 'electrical'
                    edge_count['electrical'] += 1     
        logging.info(f"Detected {edge_count['electrical']} elect_proj edges")

        # Parallelize the edge calculations using ThreadPoolExecutor
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

        input_edge_count = {'exc_input': 0, 'inh_input': 0}
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
                            input_edge_count['exc_input'] += 1
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
                            input_edge_count['inh_input'] += 1
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
                                input_edge_count['exc_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e5] = input_w
                            else:
                                e6 = G.add_edge(input_map[src], pop_map[tgt])
                                total_input_edges += 1
                                eprop_name[e6] = destination
                                eprop_type[e6] = "GapJ"
                                edge_type[e6] = "inh_input"
                                input_map[destination] = e6
                                input_edge_count['inh_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e6] = input_w
                    except Exception as ex:
                        logging.warning(f"Failed to add input edge from {src} to {tgt}: {ex}")
        
        # Log detailed statistics
        logging.info("Destination breakdown: %s", destination_stats)
        logging.info("Detected %d input edges: %d exc | %d inh", total_input_edges, input_edge_count['exc_input'], input_edge_count['inh_input'])
        #--------------------------------------------------------------------#
        main_vertices = [v for v in G.vertices() if vprop_name[v] in pop_map.keys()]
        main_edges = [e for e in G.edges() if e.source() in main_vertices and e.target() in main_vertices]
        G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices, efilt=lambda e: e in main_edges)
        logging.info("Number of G_main vertices: %d", G_main.num_vertices())
        logging.info("Number of G_main edges: %d", G_main.num_edges())
        
        input_vfilter = G.new_vertex_property("bool")
        input_vfilter.a = False
        input_vertices = [v for v in G.vertices() if vprop_name[v] in input_map.keys()]
        for v in input_vertices:
            input_vfilter[v] = True
            for w in v.out_neighbors():
                input_vfilter[w] = True
            for w in v.in_neighbors():
                input_vfilter[w] = True
        input_efilter = G.new_edge_property("bool")
        input_efilter.a = False
        # Create input edge filter property map - include edges connected to input vertices
        for e in G.edges():
            if input_vfilter[e.source()] and input_vfilter[e.target()]:
                if vprop_name[e.source()] in input_map.keys() or vprop_name[e.target()] in input_map.keys():
                    input_efilter[e] = True
        # Use consistent filtering with input_vfilter and input_efilter like in gt_pricenetwork_sfdp.py
        G_ilist = gt.GraphView(G, vfilt=input_vfilter, efilt=input_efilter)
        logging.info("Number of G_ilist vertices: %d", G_ilist.num_vertices())
        logging.info("Number of G_ilist edges: %d", G_ilist.num_edges())
        
        full_vertices = [v for v in G.vertices()]
        full_edges = [e for e in G.edges()]
        G_full = gt.GraphView(G, vfilt=lambda v: v in full_vertices, efilt=lambda e: e in full_edges)
        logging.info("Number of total vertices: %d", G_full.num_vertices())
        logging.info("Number of total edges: %d", G_full.num_edges())
        
        # Step 4: Create proper position property map with improved biological organization
        pos = G.new_vertex_property("vector<double>")
        G.vp["pos"] = pos
        # Fine-tune layout with sfdp using proper position initialization
        # pos = gt.sfdp_layout(G, pos=pos, groups=vprop_group, C=30.0, K=5.0, p=5.0, gamma=0.005, theta=0.5, max_iter=5000, mu=10, weighted_coarse=True)
        pos = gt.sfdp_layout(G, groups=vprop_group)
        vprop_state = G.new_vertex_property("int")
        eprop_state = G.new_edge_property("int")
        w = gt.max_cardinality_matching(G, edges=True, heuristic=True, brute_force=True)
        res = gt.max_independent_vertex_set(G)
        def create_graph_tool_animation(G, pos, state, output_file, vertex_fill_color=None,
                                        vertex_color=None, vertex_size=None, edge_color=None, edge_pen_width=None, frames=20,
                                        mode="graph_draw", **kwargs):
            fixed_pos = gt.sfdp_layout(G, cooling_step=0.99)
            # Create property maps for coloring
            vprop_color = G.new_vertex_property("vector<double>")
            eprop_color = G.new_edge_property("vector<double>")
            
            frame_files = []
            for i in range(frames):
                progress = i / frames
                active_vertices = int(progress * G.num_vertices())
                active_edges = int(progress * G.num_edges())
                refractory_vertices = int(0.1 * active_vertices)
                refractory_edges = int(0.1 * active_edges)
                inactive_vertices = G.num_vertices() - active_vertices - refractory_vertices
                inactive_edges = G.num_edges() - active_edges - refractory_edges
                for idx, v in enumerate(G.vertices()):
                    if idx < refractory_vertices:
                        vprop_state[v] = 1  # Refractory state
                        vprop_color[v] = matplotlib.cm.twilight(0.5)  # Refractory color
                    elif idx < active_vertices:
                        vprop_state[v] = 2  # Active state
                        vprop_color[v] = matplotlib.cm.twilight(0.8)
                    elif idx < inactive_vertices:
                        vprop_state[v] = 0  # Inactive state
                        vprop_color[v] = matplotlib.cm.twilight(0.3)
                    else:
                        vprop_state[v] = 1  
                        vprop_color[v] = matplotlib.cm.twilight(0.2)  
                        
                for idx, e in enumerate(G.edges()):
                    if idx < refractory_edges:
                        eprop_state[e] = 0.5  # Refractory state
                        eprop_color[e] = matplotlib.cm.twilight_r(0.5)
                    elif idx < active_edges:
                        eprop_state[e] = 1  # Active state
                        eprop_color[e] = matplotlib.cm.twilight_r(0.8)
                    elif idx < inactive_edges:
                        eprop_state[e] = 0  # Inactive state
                        eprop_color[e] = matplotlib.cm.twilight_r(0.3)
                    else:
                        eprop_state[e] = 0.1  # Inactive state
                        eprop_color[e] = matplotlib.cm.twilight_r(0.2)  
                 
                frame_file = f"metrics/{base_name}/{base_name}_frame_{i:03d}.png"
                
                gt.graph_draw(
                    G, pos=fixed_pos,
                    vertex_fill_color=b,
                    vertex_shape=res,
                    vertex_color=vprop_state,
                    edge_color=eprop_color,
                    edge_pen_width=w.t(lambda x: 0.2*x + 1),
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
        metrics = {}
        gt.remove_parallel_edges(G)
        edge_weight.a = np.random.random(len(edge_weight.a)) * 42 
        def metric(G, metric_func, *args, **kwargs):
            """Compute metric only on the largest component and assign to all vertices (NaN elsewhere)."""
            # G = gt.GraphView(G, vfilt=gt.label_largest_component(G))
            result = metric_func(G, *args, **kwargs)
            if isinstance(result, tuple):
                result = result[1]
            metric_full = G.new_vertex_property("double")
            for v in G.vertices():
                metric_full[v] = result[v]
            return metric_full
        
        with gt.openmp_context(nthreads=8):
            metrics['pagerank'] = gt.pagerank(G)

            vp, ep = gt.betweenness(G)
            metrics['betweenness'] = vp

            metrics['eigenvector'] = metric(G, lambda g: gt.eigenvector(g, edge_weight))
            metrics['katz'] = metric(G, lambda g: gt.katz(g, weight=edge_weight))
            metrics['hits_authority'] = metric(G, lambda g: gt.hits(g)[1])
            metrics['hits_hub'] = metric(G, lambda g: gt.hits(g)[2])
            metrics['eigentrust'] = metric(G, lambda g: gt.eigentrust(g, edge_weight))
            def trust_trans(g):
                vs = list(g.vertices())
                if not vs:
                    return g.new_vertex_property("double")
                return gt.trust_transitivity(g, edge_weight, source=vs[0])
            metrics['trust_transitivity'] = metric(G, trust_trans)
            # Closeness 
            metrics['closeness'] = metric(G, gt.closeness)
        ############################################################
        centrality_metrics = {
            'pr': 'pagerank',
            'betweenness': 'betweenness',
            'eigenvector': 'eigenvector',
            'katz': 'katz',
            'hitsX': 'hits_authority',
            'hitsY': 'hits_hub',
            'eigentrust': 'eigentrust',
            'trust_transitivity': 'trust_transitivity',
            'closeness': 'closeness',
        }
        for metric_name, metric_key in tqdm(centrality_metrics.items(), desc="Centrality metrics"):
            print(f"\n[INFO] Calculating and plotting {metric_name}...")
            t0 = time.time()
            metric = metrics[metric_key]
            gt.graph_draw(
                G, gt.sfdp_layout(G, cooling_step=0.99),  vertex_anchor=0, 
                vertex_fill_color=metric, vorder=metric, vertex_shape=res, vcmap=matplotlib.cm.Set3,
                edge_color=edge_weight, edge_pen_width=w.t(lambda x: 0.2*x + 1),
                edge_start_marker="bar", edge_end_marker="arrow",
                bg_color=[0.98, 0.98, 0.98, 1], output=f"metrics/{base_name}/{base_name}_{metric_name}.png"
            )
            print(f"[INFO] Saved metrics/{base_name}/{base_name}_{metric_name}.png in {time.time() - t0:.2f} seconds")
            create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state, output_file=f"metrics/{base_name}/{base_name}_{metric_name}.gif",
                                    vertex_fill_color=metric, edge_color=edge_weight, edge_pen_width=w.t(lambda x: 0.2*x + 1)
                                )
            print(f"[INFO] Saved metrics/{base_name}/{base_name}_{metric_name}.gif in {time.time() - t0:.2f} seconds")
            
        #############################################################################################################################
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        degrees = G.degree_property_map("in").get_array()
        betweenness = metrics['betweenness'].get_array()
        pagerank = metrics['pagerank'].get_array()
        eigenvector = metrics['eigenvector'].get_array()
        closeness = metrics['closeness'].get_array()
        hits_authority = metrics['hits_authority'].get_array()
        hits_hub = metrics['hits_hub'].get_array()
        eigentrust = metrics['eigentrust'].get_array()
        katz = metrics['katz'].get_array()
        trust_transitivity = metrics['trust_transitivity'].get_array()
        
        vertex_indices = [int(v) for v in G.vertices()]
        nc = len(vertex_indices)
        logging.info(f"Using all components: total vertices = {nc}")
        logging.info(f"Number of valid metric series: {0 if 'aligned_metric_arrays' not in locals() else len(aligned_metric_arrays)}")
        
        if nc == 0:
            logging.warning("Largest component empty — skipping all metric plots.")
            return
        # Helper aligns metric to largest component indices
        def _align_mask(metric_arr):
            arr = np.array([metric_arr[i] for i in vertex_indices], dtype=float)
            valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
            if not np.any(valid_mask):
                return None
            return np.array([degrees[i] for i, _ in enumerate(vertex_indices)])[valid_mask], arr[valid_mask]
        # collect per-metric aligned pairs (degree, metric)
        aligned_metric_arrays = {}
        for name, arr in (
            ('bt', betweenness),
            ('pr', pagerank),
            ('V', eigenvector),
            ('katz', katz),
            ('hitsX', hits_authority),
            ('hitsY', hits_hub),
            ('t', eigentrust),
            ('tt', trust_transitivity),
            ('c', closeness),
        ):
            pair = _align_mask(arr)
            if pair is not None and len(pair[0]) > 0:
                aligned_metric_arrays[name] = pair

        logging.info(f"Number of valid metric series: {len(aligned_metric_arrays)}")
        logging.info(f"Total vertices considered: {nc}")

        #----------------facetplot-----------------------------------
        data_list = []
        for metric_name, (deg_arr, metric_arr) in aligned_metric_arrays.items():
            df = pd.DataFrame({'Degree': deg_arr, 'Centrality': metric_arr, 'Metric': metric_name})
            data_list.append(df)

        if not data_list:
            logging.warning("No valid metric data for faceted plots.")
        else:
            combined_df = pd.concat(data_list, ignore_index=True)
            sns.set_style('whitegrid')
            g = sns.FacetGrid(combined_df, col='Metric', hue='Metric',
                              palette='tab20', sharey=False, height=3, aspect=1.2, col_wrap=3)
            g.map(sns.scatterplot, 'Degree', 'Centrality', alpha=0.7, s=20)
            def safe_kdeplot(x, y, **kwargs):
                try:
                    xa = np.asarray(x)
                    ya = np.asarray(y)
                    if xa.size < 2 or ya.size < 2:
                        return
                    if np.unique(xa[~np.isnan(xa)]).size < 2 or np.unique(ya[~np.isnan(ya)]).size < 2:
                        return
                    sns.kdeplot(x=xa, y=ya, **kwargs)
                except Exception as kw_ex:
                    logging.warning("Skipping kdeplot for facet due to: %s", kw_ex)

            g.map(safe_kdeplot, 'Degree', 'Centrality', fill=False, levels=10, alpha=0.3, warn_singular=False)
            try:
                g.map(safe_kdeplot, 'Degree', 'Centrality', fill=True, levels=10, alpha=0.3, warn_singular=False)
            except Exception:
                pass
            g.add_legend()
            g.set_axis_labels('Degree', 'Centrality')
            
            g.fig.suptitle(f"{base_name}", fontsize=12)
            plt.tight_layout()
            # Save the Graph-tool graph to disk for reproducibility before plotting
            try:
                gt_path = os.path.join(output_dir, f"{base_name}.gt")
                G.save(gt_path)
                logging.info("Saved Graph-tool file: %s", gt_path)
            except Exception as save_ex:
                logging.warning("Failed to save Graph-tool file %s: %s", gt_path, save_ex)
            g.savefig(f"metrics/{base_name}/{base_name}_facetgrid.png", dpi=300, bbox_inches='tight')
            plt.close(g.fig)
        return  
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        if 'pop' in locals():
            print(f"Population causing error: {pop.id}")
        raise

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    nml_net_files = [
        # os.path.join(script_dir, "net_files/TC2CT.net.nml"),
        # os.path.join(script_dir, "net_files/TC2PT.net.nml"),
        # os.path.join(script_dir, "net_files/TC2IT4_IT2CT.net.nml"),
        # os.path.join(script_dir, "net_files/TC2IT2PTCT.net.nml"),
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
        os.path.join(script_dir, "net_files/max_CTC_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M1b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2b_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1b_max_plus.net.nml"),
        os.path.join(script_dir, "net_files/M1_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/M2_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1_max_plus.net.nml"),
        os.path.join(script_dir, "net_files/M2aM1aS1a_max_plus.net.nml"),
        # os.path.join(script_dir, "net_files/S1bM1bM2b_max_plus.net.nml"),
        os.path.join(script_dir, "net_files/M2M1S1_max_plus.net.nml")
    ]

    for nml_net_file in nml_net_files:
        if not os.path.exists(nml_net_file):
            print(f"Warning: File '{nml_net_file}' does not exist. Skipping...")
            continue
        base_name = os.path.basename(nml_net_file).split(".")[0]
        visualize_network(nml_net_file, 
                          R_intra=0.9, R_inter=0.1, 
                          V_intra=0.9, V_inter=0.1,
                          L_intra=0.9, L_inter=0.1,
                          base_name=base_name)

