import unittest
from src.metrics.generator import generate_metrics, align_metrics
import numpy as np

class TestMetricsGenerator(unittest.TestCase):

    def test_generate_metrics(self):
        # Test the generation of metrics
        metrics = generate_metrics(num_samples=100)
        self.assertEqual(len(metrics), 100)
        self.assertTrue(all(isinstance(m, (int, float)) for m in metrics))

    def test_align_metrics(self):
        # Test the alignment of metrics
        metrics_a = np.array([1, 2, 3, 4, 5])
        metrics_b = np.array([2, 3, 4])
        aligned_a, aligned_b = align_metrics(metrics_a, metrics_b)
        self.assertEqual(len(aligned_a), len(aligned_b))
        self.assertTrue(np.array_equal(aligned_a, np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(aligned_b, np.array([1, 2, 3])))

if __name__ == '__main__':
    unittest.main()