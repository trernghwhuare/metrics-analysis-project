"""
Comprehensive test suite for the Network Metrics Generator.

This test suite validates the core functionality of the network metrics analysis framework,
ensuring robustness, accuracy, and reliability of computed metrics.
"""

import unittest
import sys
import os
import tempfile
import numpy as np

# Add the src directory to the path for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import graph_tool.all as gt
    from network_metrics_package.metrics.generator import compute_and_save_metrics, _metric_per_component_mapped
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False


@unittest.skipUnless(HAS_GRAPH_TOOL, "graph-tool not available")
class TestMetricsGenerator(unittest.TestCase):
    """Test suite for network metrics generator functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test graph
        self.test_graph = gt.Graph()
        self.test_graph.add_vertex(5)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        for source, target in edges:
            self.test_graph.add_edge(source, target)

    def test_compute_and_save_metrics_basic(self):
        """Test basic functionality of compute_and_save_metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, npz_path, csv_path = compute_and_save_metrics(
                self.test_graph, 
                out_dir=temp_dir, 
                prefix="test", 
                normalize=True, 
                save_files=True
            )
            
            # Check that metrics dictionary is returned
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)
            
            # Check that all metric arrays have correct length
            for metric_name, metric_array in metrics.items():
                self.assertEqual(len(metric_array), self.test_graph.num_vertices())
                self.assertIsInstance(metric_array, np.ndarray)
            
            # Check that files were saved
            self.assertTrue(os.path.exists(npz_path))
            self.assertTrue(os.path.exists(csv_path))
            
            # Verify file contents
            loaded_metrics = np.load(npz_path)
            self.assertEqual(set(loaded_metrics.files), set(metrics.keys()))

    def test_compute_and_save_metrics_no_save(self):
        """Test compute_and_save_metrics without saving files."""
        metrics, npz_path, csv_path = compute_and_save_metrics(
            self.test_graph, 
            out_dir=".", 
            prefix="test", 
            normalize=True, 
            save_files=False
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIsNone(npz_path)
        self.assertIsNone(csv_path)

    def test_compute_and_save_metrics_no_normalize(self):
        """Test compute_and_save_metrics without normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_norm, _, _ = compute_and_save_metrics(
                self.test_graph, 
                out_dir=temp_dir, 
                prefix="test_norm", 
                normalize=True, 
                save_files=True
            )
            
            metrics_no_norm, _, _ = compute_and_save_metrics(
                self.test_graph, 
                out_dir=temp_dir, 
                prefix="test_no_norm", 
                normalize=False, 
                save_files=True
            )
            
            # Normalized metrics should be between 0 and 1 (or NaN)
            for metric_name, metric_array in metrics_norm.items():
                valid_values = metric_array[~np.isnan(metric_array)]
                if len(valid_values) > 0:
                    self.assertGreaterEqual(np.min(valid_values), 0.0)
                    self.assertLessEqual(np.max(valid_values), 1.0)

    def test_metric_per_component_mapped_with_real_graph(self):
        """Test _metric_per_component_mapped with actual graph-tool graph."""
        # Create a disconnected graph
        disconnected_graph = gt.Graph()
        disconnected_graph.add_vertex(6)
        # Component 1: vertices 0,1,2
        disconnected_graph.add_edge(0, 1)
        disconnected_graph.add_edge(1, 2)
        # Component 2: vertices 3,4,5  
        disconnected_graph.add_edge(3, 4)
        disconnected_graph.add_edge(4, 5)
        
        # Test with a simple metric function
        def simple_metric(graph_view):
            return graph_view.num_vertices()
        
        result = _metric_per_component_mapped(disconnected_graph, simple_metric)
        
        self.assertEqual(len(result), disconnected_graph.num_vertices())
        # All vertices in component 1 should have value 3
        # All vertices in component 2 should have value 3
        for i in range(6):
            if not np.isnan(result[i]):
                self.assertEqual(result[i], 3.0)

    def test_error_handling(self):
        """Test error handling in metric computation."""
        # Create a graph that might cause issues
        problematic_graph = gt.Graph()
        problematic_graph.add_vertex(1)  # Single vertex, no edges
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, _, _ = compute_and_save_metrics(
                problematic_graph, 
                out_dir=temp_dir, 
                prefix="problematic", 
                normalize=True, 
                save_files=True
            )
            
            # Should handle gracefully without crashing
            self.assertIsInstance(metrics, dict)
            # Some metrics might be all NaN, but that's acceptable
            for metric_name, metric_array in metrics.items():
                self.assertEqual(len(metric_array), 1)


@unittest.skipUnless(HAS_GRAPH_TOOL, "graph-tool not available")  
class TestMetricsGeneratorEdgeCases(unittest.TestCase):
    """Test edge cases for network metrics generator."""

    def test_empty_graph(self):
        """Test with empty graph."""
        empty_graph = gt.Graph()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, _, _ = compute_and_save_metrics(
                empty_graph, 
                out_dir=temp_dir, 
                prefix="empty", 
                normalize=True, 
                save_files=True
            )
            
            self.assertIsInstance(metrics, dict)
            # All metrics should be empty arrays or arrays of NaN
            for metric_name, metric_array in metrics.items():
                self.assertEqual(len(metric_array), 0)

    def test_single_vertex_graph(self):
        """Test with single vertex graph."""
        single_vertex = gt.Graph()
        single_vertex.add_vertex(1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, _, _ = compute_and_save_metrics(
                single_vertex, 
                out_dir=temp_dir, 
                prefix="single", 
                normalize=True, 
                save_files=True
            )
            
            self.assertIsInstance(metrics, dict)
            for metric_name, metric_array in metrics.items():
                self.assertEqual(len(metric_array), 1)


if __name__ == '__main__':
    # Run with verbose output for better debugging
    unittest.main(verbosity=2)