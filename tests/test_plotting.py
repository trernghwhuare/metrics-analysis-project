import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network_metrics_package.plotting.compare_plots import plot_violin, plot_box

def test_plot_violin():
    # Example data for testing
    metrics_dict = {
        'metric_a': [1, 2, 3, 4, 5],
        'metric_b': [2, 3, 4, 5, 6]
    }
    
    # Call the plotting function
    plot_violin(metrics_dict)
    
    # Check if the plot was created (you may need to adjust this based on your implementation)
    assert True  # Replace with actual checks, e.g., checking if a file was created

def test_plot_box():
    # Example data for testing
    metrics_dict = {
        'metric_a': [1, 2, 3, 4, 5],
        'metric_b': [2, 3, 4, 5, 6]
    }
    filename = 'test_plot.png'
    
    # Call the plotting function
    plot_box(metrics_dict, out=filename)
    
    # Check if the file was created
    assert os.path.exists(filename)  # Ensure the file exists after saving
    
    # Clean up
    os.remove(filename)  # Remove the test file after the test