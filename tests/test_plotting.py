"""
Comprehensive test suite for the Network Metrics Plotting functionality.

This test suite validates the visualization capabilities of the network metrics analysis framework,
ensuring robustness and correctness of all plotting functions.
"""

import unittest
import sys
import os
import tempfile
import numpy as np

# Add the src directory to the path for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from network_metrics_package.plotting.compare_plots import (
        load_metrics,
        plot_violin, 
        plot_box,
        plot_heatmap_corr,
        plot_clustermap
    )
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


@unittest.skipUnless(HAS_PLOTTING, "Plotting module not available")
class TestPlottingFunctions(unittest.TestCase):
    """Test suite for plotting functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock metrics data
        np.random.seed(42)  # For reproducible tests
        self.metrics_dict = {
            'pagerank': np.random.random(100),
            'betweenness': np.random.random(100),
            'closeness': np.random.random(100),
            'eigenvector': np.random.random(100)
        }
        
        # Create metrics with NaN values
        self.metrics_with_nan = self.metrics_dict.copy()
        self.metrics_with_nan['pagerank'][10:20] = np.nan
        self.metrics_with_nan['betweenness'][30:40] = np.nan

    def test_load_metrics(self):
        """Test loading metrics from NPZ file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            npz_path = os.path.join(temp_dir, "test_metrics.npz")
            np.savez_compressed(npz_path, **self.metrics_dict)
            
            # Load and verify
            loaded_metrics = load_metrics(npz_path)
            self.assertEqual(set(loaded_metrics.keys()), set(self.metrics_dict.keys()))
            for key in self.metrics_dict.keys():
                np.testing.assert_array_equal(loaded_metrics[key], self.metrics_dict[key])

    def test_plot_violin_basic(self):
        """Test basic violin plot functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "violin_test.png")
            plot_violin(self.metrics_dict, out=output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_violin_with_nan(self):
        """Test violin plot with NaN values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "violin_nan_test.png")
            plot_violin(self.metrics_with_nan, out=output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_violin_empty_data(self):
        """Test violin plot with empty or invalid data."""
        empty_metrics = {'empty_metric': np.array([])}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "violin_empty_test.png")
            plot_violin(empty_metrics, out=output_path)
            # Should handle gracefully without crashing
            # File might not be created, but function shouldn't raise exception

    def test_plot_box_basic(self):
        """Test basic box plot functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "box_test.png")
            plot_box(self.metrics_dict, out=output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_box_with_nan(self):
        """Test box plot with NaN values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "box_nan_test.png")
            plot_box(self.metrics_with_nan, out=output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_heatmap_corr_basic(self):
        """Test basic heatmap correlation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "heatmap_test.png")
            plot_heatmap_corr(self.metrics_dict, out=output_path, annot=True)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_heatmap_corr_with_nan(self):
        """Test heatmap correlation with NaN values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "heatmap_nan_test.png")
            plot_heatmap_corr(self.metrics_with_nan, out=output_path, annot=True)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_heatmap_single_metric(self):
        """Test heatmap with single metric (edge case)."""
        single_metric = {'single': np.random.random(50)}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "heatmap_single_test.png")
            plot_heatmap_corr(single_metric, out=output_path, annot=True)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_clustermap_basic(self):
        """Test basic clustermap functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "clustermap_test.png")
            plot_clustermap(self.metrics_dict, out=output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_plot_clustermap_single_metric(self):
        """Test clustermap with single metric (should skip)."""
        single_metric = {'single': np.random.random(50)}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "clustermap_single_test.png")
            plot_clustermap(single_metric, out=output_path)
            # Should handle gracefully, file might not be created but no exception

    def test_all_plots_together(self):
        """Test running all plot types together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_violin(self.metrics_dict, out=os.path.join(temp_dir, "all_violin.png"))
            plot_box(self.metrics_dict, out=os.path.join(temp_dir, "all_box.png"))
            plot_heatmap_corr(self.metrics_dict, out=os.path.join(temp_dir, "all_heatmap.png"), annot=True)
            plot_clustermap(self.metrics_dict, out=os.path.join(temp_dir, "all_clustermap.png"))
            
            # Verify all files were created
            expected_files = ['all_violin.png', 'all_box.png', 'all_heatmap.png', 'all_clustermap.png']
            for filename in expected_files:
                self.assertTrue(os.path.exists(os.path.join(temp_dir, filename)))


if __name__ == '__main__':
    # Run with verbose output for better debugging
    unittest.main(verbosity=2)