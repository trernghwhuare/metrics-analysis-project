#!/usr/bin/env python3
"""
Tests for the GT graph generator functionality.
"""

import os
import tempfile
import pytest
import sys

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import graph_tool.all as gt
    from network_metrics_package.gt_generator import generate_graph
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False


@pytest.mark.skipif(not HAS_GRAPH_TOOL, reason="graph-tool not available")
class TestGTGenerator:
    """Test cases for graph generation functionality."""
    
    def test_price_network_generation(self):
        """Test Price network generation."""
        g = generate_graph('price', n_vertices=100, c=0.8, m=1, directed=True)
        
        assert g.num_vertices() == 100
        assert g.num_edges() > 0
        assert g.is_directed()
        assert g.gp["model"] == "price"
        
    def test_random_graph_generation(self):
        """Test random graph generation."""
        g = generate_graph('random', n_vertices=50, n_edges=100, directed=False)
        
        assert g.num_vertices() == 50
        assert g.num_edges() == 100
        assert not g.is_directed()
        assert g.gp["model"] == "random"
        
    def test_complete_graph_generation(self):
        """Test complete graph generation."""
        n_vertices = 10
        g = generate_graph('complete', n_vertices=n_vertices, directed=False)
        
        expected_edges = n_vertices * (n_vertices - 1) // 2
        assert g.num_vertices() == n_vertices
        assert g.num_edges() == expected_edges
        assert g.gp["model"] == "complete"
        
    def test_geometric_graph_generation(self):
        """Test geometric graph generation."""
        g = generate_graph('geometric', n_vertices=50, radius=0.2, dim=2)
        
        assert g.num_vertices() == 50
        assert g.num_edges() >= 0  # Could be 0 if radius is too small
        assert g.gp["model"] == "geometric"
        
    def test_lattice_graph_generation(self):
        """Test lattice graph generation."""
        dimensions = [5, 5]  # 25 vertices total
        g = generate_graph('lattice', dimensions=dimensions, periodic=False)
        
        assert g.num_vertices() == 25
        assert g.num_edges() > 0
        assert g.gp["model"] == "lattice"
        
    def test_graph_properties(self):
        """Test that generated graphs have proper properties."""
        g = generate_graph('price', n_vertices=10, c=0.8, m=1, directed=True)
        
        # Check graph properties
        assert "model" in g.gp
        assert "vertices" in g.gp
        assert "edges" in g.gp
        assert "parameters" in g.gp
        
        # Check vertex properties
        assert "vertex_id" in g.vp
        for v in g.vertices():
            assert g.vp["vertex_id"][v] == int(v)
            
        # Check edge properties
        assert "edge_id" in g.ep
        for i, e in enumerate(g.edges()):
            assert g.ep["edge_id"][e] == i
            
    def test_save_and_load(self):
        """Test saving and loading generated graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_graph.gt")
            
            # Generate and save graph
            g1 = generate_graph('price', n_vertices=20, c=0.8, m=1, directed=True)
            g1.save(output_path)
            
            # Load graph
            g2 = gt.load_graph(output_path)
            
            # Verify properties are preserved
            assert g2.num_vertices() == g1.num_vertices()
            assert g2.num_edges() == g1.num_edges()
            assert g2.is_directed() == g1.is_directed()
            assert g2.gp["model"] == g1.gp["model"]