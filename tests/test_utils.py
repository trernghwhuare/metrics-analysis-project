"""
Test suite for network metrics utility functions.
"""

import unittest
import sys
import os
import numpy as np

# Add the src directory to the path for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_metrics_package.metrics.utils import analyze_network_structure


class TestUtilsFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def test_analyze_network_structure_basic(self):
        """Test basic functionality of analyze_network_structure."""
        # Create mock metrics data
        np.random.seed(42)
        metrics_dict = {
            'pagerank': np.random.random(100),
            'betweenness': np.random.random(100),
            'closeness': np.random.random(100)
        }
        
        result = analyze_network_structure(metrics_dict)
        
        # Check that all expected keys are present
        self.assertIn('correlations', result)
        self.assertIn('summary_stats', result)
        self.assertIn('insights', result)
        
        # Check correlations structure
        self.assertEqual(set(result['correlations'].keys()), set(metrics_dict.keys()))
        for metric in metrics_dict.keys():
            self.assertEqual(set(result['correlations'][metric].keys()), set(metrics_dict.keys()))
        
        # Check summary stats structure
        self.assertEqual(set(result['summary_stats'].keys()), set(metrics_dict.keys()))
        for metric in metrics_dict.keys():
            stats = result['summary_stats'][metric]
            self.assertIn('mean', stats)
            self.assertIn('median', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)
            self.assertIn('count', stats)
            
        # Check insights
        self.assertIsInstance(result['insights'], list)
        self.assertGreater(len(result['insights']), 0)

    def test_analyze_network_structure_with_nan(self):
        """Test analyze_network_structure with NaN values."""
        metrics_dict = {
            'pagerank': np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            'betweenness': np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        }
        
        result = analyze_network_structure(metrics_dict)
        
        # Should handle NaN gracefully
        self.assertIn('correlations', result)
        self.assertIn('summary_stats', result)
        self.assertIn('insights', result)

    def test_analyze_network_structure_empty(self):
        """Test analyze_network_structure with empty data."""
        metrics_dict = {
            'empty_metric': np.array([])
        }
        
        result = analyze_network_structure(metrics_dict)
        
        # Should return empty results with appropriate insights
        self.assertIn('correlations', result)
        self.assertIn('summary_stats', result)
        self.assertIn('insights', result)
        self.assertEqual(len(result['correlations']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)