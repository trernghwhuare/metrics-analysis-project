"""
Comprehensive test suite for the Network Metrics Generator.

This test suite validates the core functionality of the network metrics analysis framework,
ensuring robustness, accuracy, and reliability of computed metrics.
"""

import unittest
import sys
import os
import numpy as np

# Add the src directory to the path for cross-package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from network_metrics_package.metrics.generator import _metric_per_component_mapped
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False


@unittest.skipUnless(HAS_GENERATOR, "Network metrics generator not available")
class TestMetricsGenerator(unittest.TestCase):
    """Test suite for network metrics generator functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock data for testing
        self.mock_components = [np.array([1, 2, 3]), np.array([4, 5])]
        self.mock_metric_func = lambda x: np.mean(x)

    def test_metric_per_component_mapped_basic(self):
        """Test basic functionality of _metric_per_component_mapped."""
        result = _metric_per_component_mapped(
            self.mock_components, 
            self.mock_metric_func
        )
        
        expected = [2.0, 4.5]  # mean of [1,2,3] and [4,5]
        np.testing.assert_array_almost_equal(result, expected)

    def test_metric_per_component_mapped_empty_component(self):
        """Test handling of empty components."""
        components_with_empty = [np.array([1, 2]), np.array([]), np.array([3])]
        result = _metric_per_component_mapped(
            components_with_empty,
            self.mock_metric_func
        )
        
        # Empty array should be handled gracefully (mean of empty is NaN)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertTrue(np.isnan(result[1]))
        self.assertAlmostEqual(result[2], 3.0)

    def test_metric_per_component_mapped_single_element(self):
        """Test components with single elements."""
        single_elements = [np.array([1]), np.array([2]), np.array([3])]
        result = _metric_per_component_mapped(
            single_elements,
            self.mock_metric_func
        )
        
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_metric_per_component_mapped_custom_metric(self):
        """Test with custom metric function."""
        custom_metric = lambda x: np.sum(x) if len(x) > 0 else 0
        result = _metric_per_component_mapped(
            self.mock_components,
            custom_metric
        )
        
        expected = [6.0, 9.0]  # sum of [1,2,3] and [4,5]
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    # Run with verbose output for better debugging
    unittest.main(verbosity=2)