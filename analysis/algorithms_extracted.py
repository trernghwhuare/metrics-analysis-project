"""Simple, dependency-light re-implementations of the core algorithms
extracted from the gt_* scripts. These use networkx/numpy instead of
graph-tool so they can run in a typical environment.

Functions:
- price_network_equivalent: preferential-attachment graph generator
- run_sirs: SIRS epidemic model on a graph
- rewire_random: perform random rewiring iterations
- run_dynamic: rewiring-only experiment coupling with SIRS to record metrics
- run_combined: simultaneous rewiring + SIRS

This module is intentionally small and focuses on reproducing the
algorithmic behavior (not the visualization code) so the comparison
plots can be generated reliably.
"""
from typing import Tuple, List
import random
import numpy as np
import networkx as nx


def price_network_equivalent(n: int, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Return a Barabási–Albert graph as a lightweight equivalent to price_network.

    Args:
        n: number of nodes
        m: number of edges to attach from a new node to existing nodes
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return nx.barabasi_albert_graph(n, max(1, m), seed=seed)


def run_sirs(G: nx.Graph, steps: int, x: float, r: float, s: float, initial_infected: int = 5, seed: int | None = None) -> List[float]:
    """Run a simple asynchronous SIRS on G and return fraction infected per step.

    States: 0=S, 1=I, 2=R
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = G.number_of_nodes()
    state = [0] * n
    nodes = list(G.nodes())
    # seed initial infections
    for v in random.sample(nodes, min(initial_infected, n)):
        state[v] = 1

    infected_frac = []
    for t in range(steps):
        order = nodes.copy()
        random.shuffle(order)
        new_state = state.copy()
        for v in order:
            if state[v] == 1:
                if random.random() < r:
                    new_state[v] = 2
                else:
                    # try to infect a random neighbor
                    nbrs = list(G.neighbors(v))
                    if nbrs:
                        w = random.choice(nbrs)
                        if state[w] == 0 and random.random() < 0.5:
                            new_state[w] = 1
            elif state[v] == 0:
                if random.random() < x:
                    new_state[v] = 1
            elif state[v] == 2:
                if random.random() < s:
                    new_state[v] = 0
        state = new_state
        infected_frac.append(sum(1 for st in state if st == 1) / n)
    return infected_frac


def rewire_random(G: nx.Graph, iterations: int, rewire_prob: float = 0.01, seed: int | None = None) -> None:
    """Perform random edge rewiring: for a fraction of edges, reattach one end to a random node."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    nodes = list(G.nodes())
    edges = list(G.edges())
    m = len(edges)
    if m == 0:
        return
    # choose edges to rewire
    for _ in range(iterations):
        if random.random() > rewire_prob:
            continue
        u, v = random.choice(edges)
        # remove and add a new edge (u, w) where w random and not u
        w = random.choice(nodes)
        if w == u or G.has_edge(u, w):
            continue
        try:
            G.remove_edge(u, v)
            G.add_edge(u, w)
            # update edge list
            edges = list(G.edges())
        except Exception:
            continue


def run_dynamic(G: nx.Graph, steps: int, rewire_iterations_per_step: int = 100, rewire_prob: float = 0.02,
                sirs_params: Tuple[float, float, float] = (0.005, 0.5, 0.05), seed: int | None = None) -> List[float]:
    """Simulate network rewiring over time and run SIRS each step to measure infected fraction.

    Returns infected fraction per step.
    """
    x, r, s = sirs_params
    infected = []
    # run short SIRS epochs between rewiring to sample infection behaviour
    for t in range(steps):
        rewire_random(G, rewire_iterations_per_step, rewire_prob, seed)
        # run a short SIRS for 1 step to measure instantaneous infected fraction
        I = run_sirs(G, 1, x, r, s, initial_infected=5, seed=seed)
        infected.append(I[-1])
    return infected


def run_combined(G: nx.Graph, steps: int, rewire_iterations_per_step: int = 50, rewire_prob: float = 0.02,
                 sirs_params: Tuple[float, float, float] = (0.005, 0.5, 0.05), seed: int | None = None) -> List[float]:
    """Run rewiring and SIRS updates interleaved every step and record infected fraction."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    x, r, s = sirs_params
    n = G.number_of_nodes()
    state = [0] * n
    nodes = list(G.nodes())
    for v in random.sample(nodes, min(5, n)):
        state[v] = 1
    infected_frac = []
    for t in range(steps):
        # small rewiring
        rewire_random(G, rewire_iterations_per_step, rewire_prob, seed)
        # asynchronous SIRS update
        order = nodes.copy()
        random.shuffle(order)
        new_state = state.copy()
        for v in order:
            if state[v] == 1:
                if random.random() < r:
                    new_state[v] = 2
                else:
                    nbrs = list(G.neighbors(v))
                    if nbrs:
                        w = random.choice(nbrs)
                        if state[w] == 0 and random.random() < 0.5:
                            new_state[w] = 1
            elif state[v] == 0:
                if random.random() < x:
                    new_state[v] = 1
            elif state[v] == 2:
                if random.random() < s:
                    new_state[v] = 0
        state = new_state
        infected_frac.append(sum(1 for st in state if st == 1) / n)
    return infected_frac
