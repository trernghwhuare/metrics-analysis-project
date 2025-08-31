#!/usr/bin/env python3

import sys
import os

# Add the metrics and plotting modules to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metrics'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plotting'))

def main():
    """Main entry point for the metrics analysis project"""
    print("Metrics Analysis Project")
    print("========================")
    print("Usage:")
    print("  For computing metrics: python src/metrics/main.py --graph <graph_file> [--out <output_dir>] [--prefix <prefix>]")
    print("  For plotting: python src/plotting/compare_plots.py")
    print("")
    print("Examples:")
    print("  python src/metrics/main.py --graph network.gt --out metrics_out --prefix my_network")
    print("  python src/plotting/compare_plots.py  # Will look for metrics in metrics_out/")

if __name__ == "__main__":
    main()