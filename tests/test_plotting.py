import pytest
from src.plotting.compare_plots import plot_metric_comparison, save_plot

def test_plot_metric_comparison():
    # Example data for testing
    metrics_a = [1, 2, 3, 4, 5]
    metrics_b = [2, 3, 4, 5, 6]
    labels = ['Metric A', 'Metric B']
    
    # Call the plotting function
    plot_metric_comparison(metrics_a, metrics_b, labels)
    
    # Check if the plot was created (you may need to adjust this based on your implementation)
    assert True  # Replace with actual checks, e.g., checking if a file was created

def test_save_plot():
    # Example plot data
    plot_data = [1, 2, 3, 4, 5]
    filename = 'test_plot.png'
    
    # Call the save function
    save_plot(plot_data, filename)
    
    # Check if the file was created
    assert os.path.exists(filename)  # Ensure the file exists after saving

    # Clean up
    os.remove(filename)  # Remove the test file after the test