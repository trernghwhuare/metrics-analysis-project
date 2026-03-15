#!/usr/bin/env python3
"""
Simple test script to verify GT generator functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from network_metrics_package import generate_graph
    print("✓ Successfully imported generate_graph")
    
    # Test generating a small graph
    g = generate_graph('price', n_vertices=10, c=0.8, m=1, directed=True)
    print(f"✓ Successfully generated Price network with {g.num_vertices()} vertices and {g.num_edges()} edges")
    
    # Test saving
    g.save("test_small_network.gt")
    print("✓ Successfully saved graph to test_small_network.gt")
    
    # Clean up
    os.remove("test_small_network.gt")
    print("✓ Cleanup completed")
    
    print("\n🎉 All tests passed! GT generator is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)