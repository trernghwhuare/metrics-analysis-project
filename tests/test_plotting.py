import pytest
import sys
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the src directory to the path for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from network_metrics_package.plotting.compare_plots import (
        plot_violin, plot_box, plot_heatmap_corr, load_metrics
    )
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


@pytest.mark.skipif(not HAS_PLOTTING, reason="Plotting module not available")
class TestPlottingFunctions:
    """Test suite for network metrics plotting functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_metrics = {
            'degree_centrality': [0.1, 0.2, 0.3, 0.4, 0.5],
            'betweenness_centrality': [0.05, 0.15, 0.25, 0.35, 0.45],
            'clustering_coefficient': [0.8, 0.7, 0.6, 0.5, 0.4],
            'eigenvector_centrality': [0.2, 0.3, 0.4, 0.5, 0.6]
        }

    def test_plot_violin_basic(self):
        """Test basic violin plot generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, 'test_violin.png')
            
            # Test without saving (should work without errors)
            plot_violin(self.test_metrics)
            
            # Test with saving
            plot_violin(self.test_metrics, out=output_file)
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0

    def test_plot_box_basic(self):
        """Test basic box plot generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, 'test_box.png')
            
            # Test without saving
            plot_box(self.test_metrics)
            
            # Test with saving
            plot_box(self.test_metrics, out=output_file)
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0

    def test_plot_violin_empty_metrics(self):
        """Test violin plot with empty metrics."""
        empty_metrics = {}
        # Should not raise an exception
        plot_violin(empty_metrics)

    def test_plot_box_single_metric(self):
        """Test box plot with single metric."""
        single_metric = {'only_metric': [1, 2, 3, 4, 5]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, 'test_single_box.png')
            plot_box(single_metric, out=output_file)
            assert os.path.exists(output_file)

    def test_invalid_file_path(self):
        """Test error handling for invalid file paths."""
        invalid_path = '/invalid/path/plot.png'
        with pytest.raises((OSError, IOError)):
            plot_violin(self.test_metrics, out=invalid_path)


def test_load_metrics_function():
    """Test the load_metrics function with mock data."""
    # This would require creating a mock .npz file
    # For now, we'll just test that the function exists and is callable
    try:
        from network_metrics_package.plotting.compare_plots import load_metrics
        assert callable(load_metrics)
    except ImportError:
        pass
        
# Additional tests for functions that might be in the module
try:
    from network_metrics_package.plotting.compare_plots import plot_heatmap_corr
    
    @pytest.mark.skipif(not HAS_PLOTTING, reason="Plotting module not available")
    class TestHeatmapPlot:
        """Additional test class for heatmap functionality."""
        
        def setup_method(self):
            """Set up test fixtures before each test method."""
            self.test_metrics = {
                'degree_centrality': [0.1, 0.2, 0.3, 0.4, 0.5],
                'betweenness_centrality': [0.05, 0.15, 0.25, 0.35, 0.45],
                'clustering_coefficient': [0.8, 0.7, 0.6, 0.5, 0.4],
                'eigenvector_centrality': [0.2, 0.3, 0.4, 0.5, 0.6]
            }
        
        def test_plot_heatmap_corr_basic(self):
            """Test basic correlation heatmap generation."""
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_file = os.path.join(tmp_dir, 'test_heatmap.png')
                
                # Test without saving
                plot_heatmap_corr(self.test_metrics)
                
                # Test with saving
                plot_heatmap_corr(self.test_metrics, out=output_file)
                assert os.path.exists(output_file)
                assert os.path.getsize(output_file) > 0
                
except ImportError:
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])