import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_metrics_package.metrics.generator import _metric_per_component_mapped
import numpy as np

class TestMetricsGenerator(unittest.TestCase):

    def test_metric_per_component_mapped(self):
        # Test the _metric_per_component_mapped function
        # Note: This is a simplified test - in practice, you'd want to create a mock graph
        pass

if __name__ == '__main__':
    unittest.main()