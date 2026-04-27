import os
from matplotlib import cm
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
import subprocess
import numpy as np
from pyneuroml.pynml import read_neuroml2_file
import subprocess
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

def visualize_network(nml_net_file, V_intra, V_inter, R_intra, R_inter, L_intra, L_inter, base_name):
    output_dir = f"hierarchy/{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    nml_doc = read_neuroml2_file(nml_net_file)
    input_groups = {}
    G = gt.Graph()
    vprop_name = G.new_vertex_property("string")
    vprop_type = G.new_vertex_property("string")
    vprop_size = G.new_vertex_property("double")
    vprop_color = G.new_vertex_property("vector<double>")
    eprop_type = G.new_edge_property("string")
    eprop_width = G.new_edge_property("double")
    eprop_color = G.new_edge_property("vector<double>")
    eprop_dash = G.new_edge_property("vector<double>")
    edge_weight = G.new_edge_property("double")
    edge_type_prop = G.new_edge_property("string")
    vprop_group = G.new_vertex_property("int")
    
    def get_pop_type(pop_id):
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) >= 2 and parts[0] in Region_list:
            for exc_type in exc_e:
                if parts[1].startswith(exc_type):
                    return exc_type
            for inh_type in inh_e:
                if parts[1].startswith(inh_type):
                    return inh_type
        elif len(parts) >= 2 and parts[0] not in Region_list:
            for exc_type in exc_e:
                if parts[0].startswith(exc_type):
                    return exc_type
            for inh_type in inh_e:
                if parts[0].startswith(inh_type):
                    return inh_type
        elif len(parts) == 1:
            for exc_type in exc_e:
                if parts[0] in exc_e:
                    return exc_type
            for inh_type in inh_e:
                if parts[0] in inh_e:
                    return inh_type
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
    def get_input_type(pg_id):
        pg_id_lower = pg_id.lower()
        if 'exc' in pg_id_lower:
            return 'exc'
        elif 'inh' in pg_id_lower:
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
    # Step 1: Create vertices using population properties
    pop_map = {}
    for pop in nml_doc.networks[0].populations:
        v1 = G.add_vertex()
        vprop_name[v1] = pop.id
        pop_type = get_pop_type(pop.id)
        vprop_type[v1] = pop_type
        size = float(pop.size) if hasattr(pop, 'size') else 1.0
        vprop_size[v1] = np.log1p(size) * 2  # Logarithmic scaling for better visibility
        v1.population = pop
        pop_map[pop.id] = v1
    logging.info(f"Total {len(pop_map)} population vertices from {base_name} network")
    
    group_keys = []
    for v in G.vertices():
        pop_id = vprop_name[v]
        pop_type = get_pop_type(pop.id)
        if vprop_name[v] != pop_id:
            ilist_id = vprop_name[v]
            input_type = get_input_type(ilist_id)
            if isinstance(ilist_id, str) and ilist_id.startswith(f"{vprop_type[v]}_"):
                gkey = ilist_id.replace(f"{vprop_type[v]}_", "")
            else:
                gkey = pop_id
            gkey = pop_id
        else:
            g_from_map = input_groups.get(pop.id) if isinstance(input_groups, dict) else None
            gkey = g_from_map if g_from_map is not None else get_Vprefix(pop_id)
        group_keys.append(gkey)
    unique_groups = sorted(dict.fromkeys(group_keys).keys())
    group_pos = {g: i for i, g in enumerate(unique_groups)}
    vprop_group = G.new_vp("int")
    for v, gkey in zip(G.vertices(),group_keys):
        vprop_group[v] = group_pos.get(gkey, 0)

    # Step 2: Add edges with different types
    edge_count = {'continuous': 0, 'electrical': 0}
    edge_type_prop = G.new_edge_property("string")  # New property for edge types
    edge_weight = G.new_edge_property("double")

    # Step 3: Community detection first
    state = gt.minimize_nested_blockmodel_dl(G)
    gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
    
    tree, prop, vprop = gt.get_hierarchy_tree(state)
    ecount = tree.num_edges()
    vcount = tree.num_vertices()
    logging.info(f"Tree has {vcount} vertices and {ecount} edges")
    
    levels = state.get_levels() # Get the hierarchy levels
    logging.info(f"Detected {len(levels)} hierarchy levels")
    for s in levels:
        print(s)
        if s.get_N() == 1:
            break
    b = levels[0].get_blocks() # Get block structure from highest level
    Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices())
    Regions = set(get_Region(vprop_name[v]) for v in G.vertices())
    Layers = set(get_layer(vprop_name[v]) for v in G.vertices())
    logging.info(f"Layers: {Layers} | Regions: {Regions} | Vprefixs: {Vprefixs}")
    
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

            inter_module = (Vprefix_pre != Vprefix_post) or (Region_pre != Region_post) or (layer_pre != layer_post)
            if inter_module:
                # For inter-module connections, use a much lower probability
                if (r_pass and v_pass and l_pass and np.random.rand() < 0.5):  # Additional 10% sampling
                    Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                    e1 = G.add_edge(pop_map[pre], pop_map[post])
                    source_block = b[e1.source()]
                    target_block = b[e1.target()]
                
                    if r_pass and v_pass:
                        eprop_width[e1] = 0.2  if source_block != target_block else 0.3
                        eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                        eprop_dash[e1] = []
                    elif r_pass:
                        # region-driven
                        eprop_width[e1] = 0.1  if source_block != target_block else 0.15
                        eprop_color[e1] = [0.0, 0.35, 0.8, 0.65] 
                        eprop_dash[e1] = [0.2, 0.2] 
                    else:
                        # Vprefix-driven
                        eprop_width[e1] = 0.9 
                        eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]
                        eprop_dash[e1] = [0.05, 0.18]
                    edge_weight[e1] = Syn_w
                    edge_type_prop[e1] = "continuous"
                    eprop_type[e1] = "continuous"
                    eprop_dash[e1] = []
                    edge_count['continuous'] += 1 * len(Syn_proj.continuous_connection_instance_ws)
                        
            elif (r_pass and v_pass and l_pass):  # For intra-module connections
                Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                e1 = G.add_edge(pop_map[pre], pop_map[post])
                source_block = b[e1.source()]
                target_block = b[e1.target()]
            
                if r_pass and v_pass:
                    eprop_width[e1] = Syn_w * 2  if source_block != target_block else Syn_w * 3
                    eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                    eprop_dash[e1] = []
                elif r_pass:
                    # region-driven
                    eprop_width[e1] = Syn_w * 10  if source_block != target_block else Syn_w * 15
                    eprop_color[e1] = [0.0, 0.35, 0.8, 0.65] 
                    eprop_dash[e1] = [0.2, 0.2] 
                else:
                    # Vprefix-driven
                    eprop_width[e1] = Syn_w * 5 
                    eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]
                    eprop_dash[e1] = [0.05, 0.18]

                edge_weight[e1] = Syn_w
                edge_type_prop[e1] = "continuous"
                eprop_type[e1] = "continuous"
                eprop_dash[e1] = []
                edge_count['continuous'] += 1 * len(Syn_proj.continuous_connection_instance_ws)
    logging.info(f"Detected {edge_count['continuous']} Syn_proj edges (modular, sparse)")

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
            
            # Modified: Use AND instead of OR to make modular structure more apparent
            # Additionally, make inter-module connections even more sparse
            inter_module = (Vprefix_pre != Vprefix_post) or (Region_pre != Region_post) or (layer_pre != layer_post)
            if inter_module:
                # For inter-module connections, use a much lower probability
                if (r_pass and v_pass and l_pass and np.random.rand() < 0.1):  # Additional 10% sampling
                    elect_w = elect_proj.electrical_connection_instance_ws[0].weight
                    e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                pop_map[elect_proj.postsynaptic_population])
                    source_block = b[e2.source()]
                    target_block = b[e2.target()]
                    
                    if r_pass and v_pass:
                        eprop_width[e2] = elect_w * 5 if source_block != target_block else elect_w * 4
                        eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                        eprop_dash[e2] = []
                    elif r_pass:
                        eprop_width[e2] = elect_w * 8  if source_block != target_block else elect_w * 6
                        eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
                        eprop_dash[e2] = [0.18, 0.18]
                    else:
                        eprop_width[e2] = elect_w * 2
                        eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]
                        eprop_dash[e2] = [0.08, 0.22]
                    edge_weight[e2] = elect_w
                    edge_type_prop[e2] = "electrical"
                    eprop_type[e2] = "electrical"
                    edge_count['electrical'] += 1 * len(elect_proj.electrical_connection_instance_ws)    
                    
            elif (r_pass and v_pass and l_pass):  # For intra-module connections
                elect_w = elect_proj.electrical_connection_instance_ws[0].weight
                e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                            pop_map[elect_proj.postsynaptic_population])
                source_block = b[e2.source()]
                target_block = b[e2.target()]
                
                if r_pass and v_pass:
                    eprop_width[e2] = elect_w * 5 if source_block != target_block else elect_w * 4
                    eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                    eprop_dash[e2] = []
                elif r_pass:
                    eprop_width[e2] = elect_w * 8  if source_block != target_block else elect_w * 6
                    eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
                    eprop_dash[e2] = [0.18, 0.18]
                else:
                    eprop_width[e2] = elect_w * 10
                    eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]
                    eprop_dash[e2] = [0.08, 0.22]

                edge_weight[e2] = elect_w
                edge_type_prop[e2] = "electrical"
                eprop_type[e2] = "electrical"
                edge_count['electrical'] += 1 * len(elect_proj.electrical_connection_instance_ws)    
    logging.info(f"Detected {edge_count['electrical']} elect_proj edges (modular, sparse)")

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
            vprop_size[v2] = np.log1p(len(pg.id))
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
            vprop_size[v3] = np.log1p(len(ci.id))
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
            vprop_size[v4] = np.log1p(len(vc.id))
            input_map[vc.id] = v4
            input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
            input_type_stats[vc_type] = input_type_stats.get(vc_type, 0) + 1
    logging.info(f"Detected {len(input_map)} input vertices type breakdown : %s", input_type_stats)
    logging.info(f"input_states: %s", input_states)

    
    input_edges = {'GABA': 0, 'AMPA_NMDA': 0}
    input_edge_type_prop = G.new_edge_property("string")
    input_edge_weight = G.new_edge_property("double")
    total_input_edges = 0
    
    # Statistics for debugging
    destination_stats = {}
    input_type_stats = {}
    
    for ilist in nml_doc.networks[0].input_lists:
        pre = ilist.component
        posts = ilist.populations if isinstance(ilist.populations, list) else [ilist.populations]
        inputs = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
        
        # Track input type statistics
        input_type = get_input_type(pre)
        input_type_stats[input_type] = input_type_stats.get(input_type, 0) + 1
        
        for input_item in inputs:
            for post in posts:
                # Check if both source and target exist
                if pre in input_map and post in pop_map:
                    try:
                        e3 = G.add_edge(input_map[pre], pop_map[post])
                        total_input_edges += 1
                        
                        # Set edge properties based on input characteristics
                        input_edge_type_prop[e3] = input_type
                        
                        # Set weight if available
                        if hasattr(input_item, 'weight'):
                            input_edge_weight[e3] = float(input_item.weight)
                        else:
                            input_edge_weight[e3] = 1.0  # Default weight
                        
                        # Determine destination
                        destination = None
                        if hasattr(input_item, 'destination'):
                            destination = input_item.destination
                            # Track destination statistics
                            destination_stats[destination] = destination_stats.get(destination, 0) + 1
                        else:
                            if input_type == 'exc':
                                destination = 'AMPA_NMDA'
                            elif input_type == 'inh':
                                destination = 'GABA'
                            else:
                                # Default to GABA for unknown types to maintain consistency with previous behavior
                                destination = 'GABA'
                            
                            # Track inferred destinations
                            key = f"Inferred:{destination}"
                            destination_stats[key] = destination_stats.get(key, 0) + 1
                        
                        # Update counters based on destination
                        # Fix the typo in destination name
                        if destination == 'GABA':
                            input_edges["GABA"] += 1
                        elif destination == 'AMPA_NMDA' or destination == 'APMA_NMDA':  # Handle the typo
                            input_edges["AMPA_NMDA"] += 1
                            # Fix the typo for consistency
                            destination = 'AMPA_NMDA'
                        else:
                            # Handle any other destination types that might appear
                            if destination not in input_edges:
                                input_edges[destination] = 0
                            input_edges[destination] += 1
                            
                    except Exception as e:
                        logging.warning(f"Failed to create edge from {pre} to {post}: {e}")
    
    # Log detailed statistics
    logging.info(f"Detected {len(input_map)} input vertices type breakdown : %s", input_type_stats)
    logging.info("Destination breakdown: %s", destination_stats)
    logging.info(f"Detected %d input edges: {input_edges['GABA']} | {input_edges['AMPA_NMDA']}", total_input_edges)
    
    main_vertices = [v for v in G.vertices() if vprop_type[v] != "input"]
    G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices)
    state_ndc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=False))
    state_ndc_main.print_summary()
    logging.info("Block counts (ndc): %s", np.unique(state_ndc_main.get_levels()[0].get_blocks().a, return_counts=True))

    state_dc_main  = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=True))
    state_dc_main.print_summary()
    logging.info("Block counts (dc): %s", np.unique(state_dc_main.get_levels()[0].get_blocks().a, return_counts=True))
    logging.info("Number of edges: %d", G.num_edges())
    logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_main.entropy() - state_dc_main.entropy())
    
    
    input_vertices = [v for v in G.vertices() if vprop_type[v] != pop_type]
    ilist_edges = [e for e in G.edges() if e.source() in input_vertices]
    G_ilist = gt.GraphView(G, vfilt=lambda v: v in input_vertices, efilt=lambda e: e in ilist_edges)
    state_ndc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=False))
    state_ndc_ilist.print_summary()
    logging.info("Non-degree-corrected ILIST:%s", state_ndc_ilist.entropy())
    state_dc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=True))
    state_dc_ilist.print_summary()
    logging.info("Degree-corrected ILIST:%s", state_dc_ilist.entropy())
    logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_ilist.entropy() - state_dc_ilist.entropy())
    
    full_vertices = [v for v in G.vertices()]
    full_edges = [e for e in G.edges()]
    G_full = gt.GraphView(G, vfilt=lambda v: v in full_vertices, efilt=lambda e: e in full_edges)
    state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
    state_ndc_full.print_summary()
    logging.info("Non-degree-corrected FULL:%s", state_ndc_full.entropy())
    state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
    state_dc_full.print_summary()
    logging.info("Degree-corrected FULL:%s", state_dc_full.entropy())
    logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_full.entropy() - state_dc_full.entropy())
    
    # Step 4: Create proper position property map with improved biological organization
    pos = G.new_vertex_property("vector<double>")
    G.vp["pos"] = pos

    # Step 5: Fine-tune layout with sfdp using proper position initialization
    pos = gt.sfdp_layout(G, pos=pos, groups=vprop_group, C=30.0, K=5.0, p=5.0, gamma=0.005, theta=0.5, max_iter=5000, mu=10, weighted_coarse=True)
  
    # Step 6: Draw with community colors and edge types
    deg = G.degree_property_map("in")
    deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
    v_prop_map = G.new_vertex_property("double")
    for v in G.vertices():
        v_prop_map[v] = deg[v]
    ebet = gt.betweenness(G)[1]
    e_prop_map = G.new_edge_property("double")
    for e in G.edges():
        e_prop_map[e] = ebet[e]
    
    #############################################################################################################################
    def animate_hierarchy(state, output_gif, frames=5, sweeps_per_frame=50, **draw_kwargs):
        frame_files = []
        for i in range(frames):
            for j in range(sweeps_per_frame):
                state.multiflip_mcmc_sweep(niter=10)
            frame_file = f"hierarchy/{base_name}/hierarchy_frame_{i:03d}.png"
            state.draw(output=frame_file, **draw_kwargs)
            frame_files.append(frame_file)
        # Create GIF
        cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
        subprocess.run(cmd, check=True)
        # Clean up
        for f in frame_files:
            if os.path.exists(f):
                os.remove(f)
        gc.collect()

    state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
    gt.mcmc_equilibrate(state_ndc_full, wait=10, mcmc_args=dict(niter=10))
    for i in range(4):
        for j in range(250):
            state_ndc_full.multiflip_mcmc_sweep(niter=10)
        state_ndc_full.draw(
            bg_color=[0.98, 0.98, 0.98, 1],
            output=f"hierarchy/{base_name}/{base_name}_ndc_hierarchy_{i}.png",
            empty_branches=False,
        )
    animate_hierarchy(state_ndc_full, output_gif=f"hierarchy/{base_name}/{base_name}_ndc_hierarchy.gif",
                    frames=5, sweeps_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
    
    state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
    gt.mcmc_equilibrate(state_dc_full, wait=10, mcmc_args=dict(niter=10))
    for i in range(4):
        for j in range(250):
            state_dc_full.multiflip_mcmc_sweep(niter=10)
        state_dc_full.draw(
            bg_color=[0.98, 0.98, 0.98, 1],
            output=f"hierarchy/{base_name}/{base_name}_dc_hierarchy_{i}.png",
            empty_branches=False,
        )
    animate_hierarchy(state_dc_full, output_gif=f"hierarchy/{base_name}/{base_name}_dc_hierarchy.gif",
                    frames=5, sweeps_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
    
    state_hierarchy = gt.NestedBlockState(G,base_type=gt.RankedBlockState, state_args=dict(eweight=gt.contract_parallel_edges(G_full)))
    bs = []
    def collect_partitions(s):
        nonlocal bs
        bs.append(s.get_bs())
    gt.mcmc_equilibrate(state_hierarchy, force_niter=1000, mcmc_args=dict(niter=10), callback=collect_partitions)
    pmode = gt.ModeClusterState(bs, nested=True)
    gt.mcmc_equilibrate(state_hierarchy, wait=1, mcmc_args=dict(niter=1))
    modes = pmode.get_modes()
    for i, mode in enumerate(modes):
        b = mode.get_max_nested()    # mode's maximum
        pv = mode.get_marginal(G)    # mode's marginal distribution
        logging.info(f"Mode {i} with size {mode.get_M()/len(bs)}")
        state_hierarchy = state_hierarchy.copy(bs=b)
        state_hierarchy.draw(vertex_shape="pie", vertex_pie_fractions=pv,
                             bg_color=[0.98, 0.98, 0.98, 1],output=f"hierarchy/{base_name}/{base_name}_hierarchy_{i}.png" ,empty_branches=False)
    animate_hierarchy(state_hierarchy,output_gif=f"hierarchy/{base_name}/{base_name}_hierarchy.gif",frames=5,sweeps_per_frame=50,bg_color=[0.98, 0.98, 0.98, 1])
    
    
    #############################################################################################################################
    def animate_ghcp(state, tpos, shape, cts, eprop_color, output_gif,frames=5, sweeps_per_frame=50, vertex_fill_color=None, 
                    vertex_size=None, **kwargs):
        frame_files = []
        top_level = state.get_levels()[0]
        num_blocks = len(set(top_level.get_blocks().a))
        community_colors = [
            [
                0.5 + 0.5 * np.cos(2 * np.pi * i / max(num_blocks, 1)),
                0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 1/3)),
                0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 2/3)),
                0.8
            ]
            for i in range(num_blocks)
        ]
        vprop_block_color = G.new_vertex_property("vector<double>")
        
        for i in range(frames):
            for j in range(sweeps_per_frame):
                state.multiflip_mcmc_sweep(niter=10)
            blocks = top_level.get_blocks()
            for v in G.vertices():
                vprop_block_color[v] = community_colors[blocks[v] % len(community_colors)]

            frame_file = f"hierarchy/{base_name}/ghcp_frame_{i:03d}.png"
            gt.graph_draw(
                G, pos=tpos,
                vertex_shape=shape,
                edge_control_points=cts,
                edge_color=eprop_color,
                vertex_color=b,
                vertex_fill_color=vprop_block_color,
                vertex_size=vertex_size,
                bg_color=[0.98, 0.98, 0.98, 1],
                output=frame_file,
                **kwargs
            )
            frame_files.append(frame_file)
        # Create GIF
        cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
        subprocess.run(cmd, check=True)
        # Clean up
        for f in frame_files:
            if os.path.exists(f):
                os.remove(f)

    state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[edge_weight],rec_types=["discrete-binomial"]))
    gt.mcmc_equilibrate(state, wait=10, mcmc_args=dict(niter=10))
    tree, prop_map, vprop = gt.get_hierarchy_tree(state)
    root = tree.vertex(tree.num_vertices() - 1, use_index=False)
    tpos = gt.radial_tree_layout(tree, root, weighted=True)
    cts = gt.get_hierarchy_control_points(G, tree, tpos)
    b = state.levels[0].b
    shape = b.copy()
    shape.a %= 14
    gt.graph_draw(G, pos=G.own_property(tpos), 
                vertex_fill_color=b, 
                vertex_shape=shape,edge_control_points=cts,edge_color=eprop_color,
                vertex_pen_width=2.5,vertex_anchor=0, 
                bg_color=[0.98, 0.98, 0.98, 1],output=f"hierarchy/{base_name}/{base_name}_ghcp.png")
    for i in range(100):
        ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
    state.draw(edge_color=edge_weight.copy("double"), ecmap=matplotlib.cm.plasma,
                    eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=0.2),
                    edge_gradient=[], bg_color=[0.98, 0.98, 0.98, 1],output=f"hierarchy/{base_name}/{base_name}_ghcp_wsbm.png")
    animate_ghcp(state, tpos, shape, cts, eprop_color,output_gif=f"hierarchy/{base_name}/{base_name}_ghcp.gif",frames=5, sweeps_per_frame=50)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nml_net_files = [
        # os.path.join(script_dir, "net_files/TC2CT.net.nml"),
        os.path.join(script_dir, "net_files/TC2PT.net.nml"),
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
        visualize_network(nml_net_file, V_intra=0.5, V_inter=0.5, R_intra=0.5, R_inter=0.5,L_intra=0.5, L_inter=0.5, base_name=base_name)
    
                
