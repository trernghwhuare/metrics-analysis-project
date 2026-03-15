#!/usr/bin/env python3
"""
Graph-Tool Graph Generator

This script generates various graph models using graph-tool and saves them as .gt files.
It supports multiple graph generation models including random graphs, scale-free networks,
and more.

Usage:
    python -m network_metrics_package.gt_generator --model price --output test.gt --vertices 1000
"""

import argparse
import logging
import os
import sys
import numpy as np

# Add src to path for package imports if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import graph_tool.all as gt
    HAS_GRAPH_TOOL = True
except ImportError as e:
    print(f"graph-tool not available: {e}")
    HAS_GRAPH_TOOL = False

logger = logging.getLogger(__name__)


def create_price_network(n_vertices, c=0.8, m=1, directed=False):
    """
    Create a Price network (scale-free directed graph).
    
    Args:
        n_vertices (int): Number of vertices
        c (float): Constant factor for edge probability
        m (int): Number of edges to attach from new vertex
        directed (bool): Whether to create directed graph
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    logger.info(f"Creating Price network with {n_vertices} vertices, c={c}, m={m}, directed={directed}")
    return gt.price_network(n_vertices, c=c, m=m, directed=directed)


def create_random_graph(n_vertices, n_edges, directed=False, parallel_edges=False, self_loops=False):
    """
    Create a random graph (Erdős–Rényi model).
    
    Args:
        n_vertices (int): Number of vertices
        n_edges (int): Number of edges
        directed (bool): Whether to create directed graph
        parallel_edges (bool): Allow parallel edges
        self_loops (bool): Allow self loops
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    logger.info(f"Creating random graph with {n_vertices} vertices, {n_edges} edges")
    # Use graph-tool's random_graph with a constant degree distribution
    def deg_sample():
        return max(1, int(2 * n_edges / n_vertices))
    
    # random_graph returns (graph, vertex_property_map)
    g, _ = gt.random_graph(
        n_vertices, 
        deg_sample,
        directed=directed,
        parallel_edges=parallel_edges,
        self_loops=self_loops,
        model="configuration"
    )
    
    # Adjust edge count if needed
    current_edges = g.num_edges()
    if current_edges < n_edges:
        # Add more edges randomly
        vertices = list(g.vertices())
        while g.num_edges() < n_edges:
            v1, v2 = np.random.choice(vertices, 2, replace=False)
            if not g.edge(v1, v2):
                g.add_edge(v1, v2)
                if not directed and not g.edge(v2, v1):
                    g.add_edge(v2, v1)
    elif current_edges > n_edges:
        # Remove excess edges
        edges = list(g.edges())
        np.random.shuffle(edges)
        for e in edges[n_edges:]:
            g.remove_edge(e)
            
    return g


def create_complete_graph(n_vertices, directed=False):
    """
    Create a complete graph.
    
    Args:
        n_vertices (int): Number of vertices
        directed (bool): Whether to create directed graph
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    logger.info(f"Creating complete graph with {n_vertices} vertices")
    g = gt.Graph(directed=directed)
    g.add_vertex(n_vertices)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            g.add_edge(g.vertex(i), g.vertex(j))
            if directed:
                g.add_edge(g.vertex(j), g.vertex(i))
    return g


def create_lattice_graph(dimensions, periodic=False):
    """
    Create a lattice graph.
    
    Args:
        dimensions (list): List of dimensions [d1, d2, ...]
        periodic (bool): Whether to make it periodic (toroidal)
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    logger.info(f"Creating lattice graph with dimensions {dimensions}, periodic={periodic}")
    return gt.lattice(dimensions, periodic=periodic)


def create_geometric_graph(n_vertices, radius, dim=2):
    """
    Create a random geometric graph.
    
    Args:
        n_vertices (int): Number of vertices
        radius (float): Connection radius
        dim (int): Dimension of embedding space
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    logger.info(f"Creating geometric graph with {n_vertices} vertices, radius={radius}, dim={dim}")
    points = np.random.random((n_vertices, dim))
    g = gt.geometric_graph(points, radius)[0]
    return g


def add_basic_properties(g, model_name, **kwargs):
    """
    Add basic vertex and edge properties to the graph for identification.
    
    Args:
        g (graph_tool.Graph): Graph to add properties to
        model_name (str): Name of the graph model used
        **kwargs: Additional parameters used in generation
    """
    # Add graph-level properties
    g.gp["model"] = g.new_gp("string", val=model_name)
    g.gp["vertices"] = g.new_gp("int", val=g.num_vertices())
    g.gp["edges"] = g.new_gp("int", val=g.num_edges())
    
    # Add vertex properties
    vprop_id = g.new_vertex_property("int")
    for v in g.vertices():
        vprop_id[v] = int(v)
    g.vp["vertex_id"] = vprop_id
    
    # Add edge properties
    eprop_id = g.new_edge_property("int")
    for i, e in enumerate(g.edges()):
        eprop_id[e] = i
    g.ep["edge_id"] = eprop_id
    
    # Store generation parameters
    param_str = ";".join([f"{k}={v}" for k, v in kwargs.items()])
    g.gp["parameters"] = g.new_gp("string", val=param_str)
    
    logger.info(f"Added basic properties to graph: {model_name}")


def generate_graph(model, **kwargs):
    """
    Generate a graph based on the specified model.
    
    Args:
        model (str): Graph model to use
        **kwargs: Model-specific parameters
        
    Returns:
        graph_tool.Graph: Generated graph
    """
    model_functions = {
        'price': create_price_network,
        'random': create_random_graph,
        'complete': create_complete_graph,
        'lattice': create_lattice_graph,
        'geometric': create_geometric_graph,
    }
    
    if model not in model_functions:
        raise ValueError(f"Unknown model: {model}. Available models: {list(model_functions.keys())}")
    
    g = model_functions[model](**kwargs)
    add_basic_properties(g, model, **kwargs)
    
    return g


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph-Tool Graph Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models and their parameters:
  price: --vertices N --c C --m M --directed
  random: --vertices N --edges E --directed --parallel-edges --self-loops  
  complete: --vertices N --directed
  lattice: --dimensions D1 D2 ... --periodic
  geometric: --vertices N --radius R --dim D

Examples:
  # Generate a Price network
  python -m network_metrics_package.gt_generator --model price --vertices 1000 --output price_1000.gt
  
  # Generate a random graph
  python -m network_metrics_package.gt_generator --model random --vertices 500 --edges 2000 --output random_500_2000.gt
  
  # Generate a geometric network
  python -m network_metrics_package.gt_generator --model geometric --vertices 100 --radius 0.2 --output geo_100.gt
        """
    )
    
    parser.add_argument("--model", required=True, 
                       choices=['price', 'random', 'complete', 'lattice', 'geometric'],
                       help="Graph generation model to use")
    parser.add_argument("--output", required=True, help="Output .gt file path")
    
    # Common parameters
    parser.add_argument("--vertices", type=int, default=100, help="Number of vertices (default: 100)")
    parser.add_argument("--directed", action="store_true", help="Create directed graph")
    
    # Model-specific parameters
    parser.add_argument("--c", type=float, default=0.8, help="Price network constant (default: 0.8)")
    parser.add_argument("--m", type=int, default=1, help="Price network edges per new vertex (default: 1)")
    parser.add_argument("--edges", type=int, help="Number of edges for random graph")
    parser.add_argument("--parallel-edges", action="store_true", help="Allow parallel edges in random graph")
    parser.add_argument("--self-loops", action="store_true", help="Allow self loops in random graph")
    parser.add_argument("--dimensions", type=int, nargs='+', help="Lattice dimensions (e.g., 10 10 for 2D)")
    parser.add_argument("--periodic", action="store_true", help="Make lattice periodic (toroidal)")
    parser.add_argument("--radius", type=float, help="Connection radius for geometric graph")
    parser.add_argument("--dim", type=int, default=2, help="Dimension for geometric graph (default: 2)")
    parser.add_argument("--k", type=int, help="Nearest neighbors for small-world graph")
    parser.add_argument("--p", type=float, help="Rewiring probability for small-world graph")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if not HAS_GRAPH_TOOL:
        print("graph-tool is required but not available. Please install it.")
        sys.exit(1)
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Prepare model-specific parameters
    model_params = {}
    
    if args.model == 'price':
        model_params.update({
            'n_vertices': args.vertices,
            'c': args.c,
            'm': args.m,
            'directed': args.directed
        })
    elif args.model == 'random':
        if args.edges is None:
            parser.error("--edges is required for random model")
        model_params.update({
            'n_vertices': args.vertices,
            'n_edges': args.edges,
            'directed': args.directed,
            'parallel_edges': args.parallel_edges,
            'self_loops': args.self_loops
        })
    elif args.model == 'complete':
        model_params.update({
            'n_vertices': args.vertices,
            'directed': args.directed
        })
    elif args.model == 'lattice':
        if args.dimensions is None:
            parser.error("--dimensions is required for lattice model")
        model_params.update({
            'dimensions': args.dimensions,
            'periodic': args.periodic
        })
    elif args.model == 'geometric':
        if args.radius is None:
            parser.error("--radius is required for geometric model")
        model_params.update({
            'n_vertices': args.vertices,
            'radius': args.radius,
            'dim': args.dim
        })
    
    try:
        # Generate graph
        logger.info(f"Generating {args.model} graph...")
        g = generate_graph(args.model, **model_params)
        
        # Save graph
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        os.makedirs(output_dir, exist_ok=True)
        g.save(args.output)
        logger.info(f"Graph saved to {args.output}")
        logger.info(f"Graph stats: {g.num_vertices()} vertices, {g.num_edges()} edges")
        
        # Print graph properties
        logger.info("Graph properties:")
        for prop_name in g.gp.keys():
            logger.info(f"  {prop_name}: {g.gp[prop_name]}")
            
    except Exception as e:
        logger.error(f"Failed to generate graph: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()