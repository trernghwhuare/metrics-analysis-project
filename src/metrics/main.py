import argparse
import logging
import graph_tool.all as gt
from .generator import compute_and_save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Compute graph-tool centrality metrics and save arrays.")
    p.add_argument("--graph", required=True, help="Path to graph-tool graph file (graphml/gt)")
    p.add_argument("--out", default=".", help="Output directory to save metric arrays")
    p.add_argument("--prefix", default="network", help="Prefix for output files")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable min-max normalization")
    p.add_argument("--threads", type=int, default=8, help="OpenMP threads for graph-tool")
    args = p.parse_args()

    try:
        G = gt.load_graph(args.graph)
    except Exception as e:
        logger.error("Failed to load graph %s: %s", args.graph, e)
        return

    compute_and_save_metrics(G, out_dir=args.out, prefix=args.prefix, normalize=args.normalize, nthreads=args.threads, save_files=True)

if __name__ == "__main__":
    main()