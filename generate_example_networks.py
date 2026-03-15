#!/usr/bin/env python3
"""
Example script for generating common network models used in neuroscience research.

This script demonstrates how to generate various graph models that are commonly
used in neural network analysis and save them as .gt files for further analysis.
"""

import os
import sys
import logging

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import graph_tool.all as gt
    from network_metrics_package.gt_generator import generate_graph
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)


def generate_neuroscience_networks(output_dir="example_networks"):
    """
    Generate example networks commonly used in neuroscience research.
    
    Args:
        output_dir (str): Directory to save generated networks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scale-free network (Price model) - mimics real neural connectivity
    logger.info("Generating scale-free Price network...")
    g_price = generate_graph(
        'price',
        n_vertices=1000,
        c=0.8,
        m=2,
        directed=True
    )
    price_path = os.path.join(output_dir, "scale_free_1000.gt")
    g_price.save(price_path)
    logger.info(f"Saved scale-free network to {price_path}")
    
    # 2. Random geometric graph - spatial embedding model
    logger.info("Generating random geometric network...")
    g_geometric = generate_graph(
        'geometric',
        n_vertices=300,
        radius=0.15,
        dim=2
    )
    geo_path = os.path.join(output_dir, "geometric_300.gt")
    g_geometric.save(geo_path)
    logger.info(f"Saved geometric network to {geo_path}")
    
    # 4. Complete graph - fully connected baseline
    logger.info("Generating complete network...")
    g_complete = generate_graph(
        'complete',
        n_vertices=50
    )
    complete_path = os.path.join(output_dir, "complete_50.gt")
    g_complete.save(complete_path)
    logger.info(f"Saved complete network to {complete_path}")
    
    # 5. Lattice network - regular grid structure
    logger.info("Generating lattice network...")
    g_lattice = generate_graph(
        'lattice',
        dimensions=[20, 20],  # 400 vertices total
        periodic=True
    )
    lattice_path = os.path.join(output_dir, "lattice_20x20.gt")
    g_lattice.save(lattice_path)
    logger.info(f"Saved lattice network to {lattice_path}")
    
    # 5. Random graph - Erdős–Rényi model
    logger.info("Generating random network...")
    g_random = generate_graph(
        'random',
        n_vertices=500,
        n_edges=2000,
        directed=False
    )
    random_path = os.path.join(output_dir, "random_500_2000.gt")
    g_random.save(random_path)
    logger.info(f"Saved random network to {random_path}")
    
    logger.info("All example networks generated successfully!")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not HAS_DEPENDENCIES:
        print("Required dependencies not available. Please install requirements.txt")
        sys.exit(1)
    
    try:
        generate_neuroscience_networks()
        print("Example networks generated! Check the 'example_networks' directory.")
    except Exception as e:
        logger.error(f"Failed to generate networks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()