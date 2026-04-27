"""Run experiments comparing the extracted algorithms and save comparison plots.

Generates the following PNGs under ./analysis/plots:
- pair_dynamic.png (gt_dynamic vs gt_dynamic_adv)
- pair_state.png (gt_state vs gt_state_adv)
- pair_combined.png (gt_state_dynamic vs gt_state_dynamic_adv)
- triple_nonadv.png (gt_dynamic, gt_state, gt_state_dynamic)
- triple_adv.png (gt_dynamic_adv, gt_state_adv, gt_state_dynamic_adv)

The script uses the lightweight implementations in algorithms_extracted.py
so it does not depend on graph-tool.
"""
from pathlib import Path
import sys
import os
import numpy as np
from scipy import stats
# Ensure project root is on PYTHONPATH so `from analysis import ...` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from analysis import algorithms_extracted as ae


OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _formula_for_label(lab):
    L = lab.lower()
    # Node-state SIRS (state-only) — mathtext (single balanced math block)
    if 'state' in L and 'dynamic' not in L:
        return (r"$(1 - r_i) \left[ 1 - \prod_j \left(1 - \beta_{ij}\right)^{A_{ij} \, \delta_{s_j(t),1}} \right] + r_i$")
    # Combined: SIRS on nodes + rewiring on edges (all inside one math block)
    if 'combined' in L or 'state_dynamic' in L:
        return (r"$(1 - r_i) \left[ 1 - \prod_j \left(1 - \beta_{ij}\right)^{A_{ij} \, \delta_{s_j(t),1}} \right] + r_i$" 
                r"$\text{  and  } $" + 
                r"$\left| pos_{s1} - pos_{t2} \right| \leq \left| pos_{s1} - pos_{t1} \right| \Rightarrow (s_1,t_1) \to (s_1,t_2)$" + "\n" )
    # Dynamic: edge-only rewiring condition (mathtext)
    if 'dynamic' in L and 'state' not in L:
        return (r"$\left| pos_{s1} - pos_{t2} \right| \leq \left| pos_{s1} - pos_{t1} \right| \Rightarrow (s_1,t_1) \to (s_1,t_2)$")
    # Advanced: Axelrod similarity and PottsGlauber transition (mathtext)
    if 'adv' in L or 'advanced' in L or 'potts' in L or 'axelrod' in L:
        return (r"$d = \sum_{l=0}^{f-1} \delta_{s^{(i)}_l(t), s^{(j)}_l(t)}$" + "\n" +
                r"$P(s_i(t+1) \mid \mathbf{s}(t)) \propto \exp\left( \sum_j A_{ij} w_{ij} f_{s_i(t+1), s_j(t)} + h^{(i)}_{s_i(t+1)} \right)$")
    return ''


def run_and_plot_pair(label_a, func_a, label_b, func_b, G_factory, steps=200, out_name="pair.png", runs=8):
    # run multiple trials to compute mean and std bands
    results_a = []
    results_b = []
    for i in range(runs):
        G1 = G_factory(i)
        G2 = G_factory(i + 1000)
        results_a.append(np.array(func_a(G1, steps)))
        results_b.append(np.array(func_b(G2, steps)))
    Y1 = np.vstack([r if len(r) == steps else np.pad(r, (0, steps - len(r)), constant_values=r[-1]) for r in results_a])
    Y2 = np.vstack([r if len(r) == steps else np.pad(r, (0, steps - len(r)), constant_values=r[-1]) for r in results_b])
    mean1 = Y1.mean(axis=0)
    std1 = Y1.std(axis=0)
    mean2 = Y2.mean(axis=0)
    std2 = Y2.std(axis=0)
    
    # Calculate p-value using t-test on final state values
    final_values_a = Y1[:, -1]  # Last column (final step) for algorithm A
    final_values_b = Y2[:, -1]  # Last column (final step) for algorithm B
    t_stat, p_value = stats.ttest_ind(final_values_a, final_values_b, equal_var=False)
    
    # Informative plot: mean +/- std, plus raw traces
    sns.set(style='whitegrid')
    palette = sns.color_palette('tab10')
    x = np.arange(steps)
    plt.figure(figsize=(12, 6), dpi=160)
    # raw faint traces
    for r in Y1:
        plt.plot(x, r, color=palette[0], alpha=0.12, linewidth=0.7)
    for r in Y2:
        plt.plot(x, r, color=palette[1], alpha=0.12, linewidth=0.7)
    # mean and std band
    plt.plot(x, mean1, label=f'{label_a} (mean)', color=palette[0], linewidth=2.4)
    plt.fill_between(x, np.clip(mean1 - std1, 0, 1), np.clip(mean1 + std1, 0, 1), color=palette[0], alpha=0.18)
    plt.plot(x, mean2, label=f'{label_b} (mean)', color=palette[1], linewidth=2.4)
    plt.fill_between(x, np.clip(mean2 - std2, 0, 1), np.clip(mean2 + std2, 0, 1), color=palette[1], alpha=0.18)
    plt.xlabel('Step')
    plt.ylabel('Fraction infected')
    plt.title(f'Comparison: {label_a} vs {label_b} — mean±std over {runs} runs (p-value: {p_value:.3e})')
    plt.legend()

    # parameter summary box
    param_text = (
        f"Runs: {runs}\nSteps: {steps}\n{label_a}: params embedded in functions\n{label_b}: params embedded in functions"
    )
    plt.gca().text(0.99, 0.02, param_text, transform=plt.gca().transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    # Artistic variant kept: add per-model formulas above the plot (rendered with mathtext)
    art_name = out_name.replace('.png', '_artistic.png')
    def smooth(y, w=7):
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w) / w, mode='same')
    smean1 = smooth(mean1)
    smean2 = smooth(mean2)
    plt.figure(figsize=(12, 8), dpi=160)  # Increased height from 6 to 8
    # determine per-label formulas (use mathtext-producing helper above)
    fa = _formula_for_label(label_a)
    fb = _formula_for_label(label_b)
    if fa or fb:
        caption = f"{label_a}: {fa}    |    {label_b}: {fb}"
        # place title slightly lower and caption above it to avoid overlap
        plt.title(f'Artistic Comparison: {label_a} vs {label_b} (p-value: {p_value:.3e})', y=0.85)  # Lowered from 0.90 to 0.85
        plt.gcf().text(0.5, 0.94, caption, ha='center', va='top', fontsize=10)  # Lowered from 0.96 to 0.94
    else:
        plt.title(f'Artistic Comparison: {label_a} vs {label_b} (p-value: {p_value:.3e})', y=0.85)
    plt.plot(x, smean1, label=f'{label_a} (smoothed)', color=palette[0], linewidth=2.6)
    plt.plot(x, smean2, label=f'{label_b} (smoothed)', color=palette[1], linewidth=2.6)
    plt.fill_between(x, np.clip(smean1 - std1, 0, 1), np.clip(smean1 + std1, 0, 1), color=palette[0], alpha=0.12)
    plt.fill_between(x, np.clip(smean2 - std2, 0, 1), np.clip(smean2 + std2, 0, 1), color=palette[1], alpha=0.12)
    plt.xlabel('Step')
    plt.ylabel('Fraction infected')
    plt.legend()
    plt.tight_layout(rect=(0, 0.03, 1, 0.82))  # Reduced top margin from 0.92 to 0.82
    # save only artistic variant
    plt.savefig(OUT_DIR / art_name)
    plt.close()


def run_and_plot_triple(labels, funcs, G_factory, steps=200, out_name="triple.png", runs=8):
    datasets = []
    all_results = []  # Store raw results for statistical testing
    for f in funcs:
        results = []
        for i in range(runs):
            G = G_factory(i)
            results.append(np.array(f(G, steps)))
        all_results.append(results)
        padded_results = [r if len(r) == steps else np.pad(r, (0, steps - len(r)), constant_values=r[-1]) for r in results]
        datasets.append(np.vstack(padded_results).mean(axis=0))
    
    # Perform one-way ANOVA on final states (tests if ANY algorithm differs from others)
    final_values = [np.array(results)[:, -1] for results in all_results]
    f_stat, p_value = stats.f_oneway(*final_values)
    
    # Artistic triple plot
    sns.set(style='whitegrid')
    palette = sns.color_palette('viridis', len(datasets))
    x = np.arange(len(datasets[0]))
    plt.figure(figsize=(12, 9), dpi=180)  # Increased height from 12x7 to 12x9 to accommodate multi-line formulas
    for lab, data, col in zip(labels, datasets, palette):
        y = np.array(data)
        sy = np.convolve(y, np.ones(9)/9, mode='same')
        plt.plot(x, sy, label=lab, color=col, linewidth=2)
        plt.fill_between(x, np.clip(sy-0.02, 0, 1), np.clip(sy+0.02, 0, 1), color=col, alpha=0.12)
    plt.xlabel('Step')
    plt.ylabel('Fraction infected')
    # Removed fixed ylim(-0.01, 1.01) to allow auto-scaling that shows actual algorithm differences
    plt.legend()
    
    # Display each algorithm's formula on separate lines with much more spacing for multi-line formulas
    fig = plt.gcf()
    y_positions = [0.96, 0.91, 0.86]  # Much larger spacing (0.05 gaps) to accommodate 2-line formulas each
    for i, lab in enumerate(labels):
        formula = _formula_for_label(lab)
        if formula:
            fig.text(0.5, y_positions[i], f"{lab}: {formula}", ha='center', va='top', fontsize=9)
    
    plt.title(f'Triple Comparison (ANOVA p-value: {p_value:.3e})', y=0.82)  # Lowered further to accommodate formulas
    plt.tight_layout(rect=(0, 0.03, 1, 0.80))  # Reduced top margin further to 0.80
    art_triple = out_name.replace('.png', '_artistic.png')
    plt.savefig(OUT_DIR / art_triple)
    plt.close()


def graph_factory(n=500, m=2, seed=42):
    def _make(seed_arg=None):
        # allow overriding seed per trial
        s = seed if seed_arg is None else seed_arg
        return ae.price_network_equivalent(n, m=m, seed=s)
    return _make


def main():
    steps = 200
    Gf = graph_factory(n=500, m=3, seed=123)

    # parameter sets (non-adv)
    state_params = (0.005, 0.5, 0.05)
    dynamic_params = dict(rewire_iterations_per_step=100, rewire_prob=0.02, sirs_params=state_params)
    combined_params = dict(rewire_iterations_per_step=50, rewire_prob=0.02, sirs_params=state_params)

    # adv parameter sets (different dynamics)
    state_adv_params = (0.004, 0.4, 0.04)
    dynamic_adv_params = dict(rewire_iterations_per_step=120, rewire_prob=0.03, sirs_params=state_adv_params)
    combined_adv_params = dict(rewire_iterations_per_step=60, rewire_prob=0.03, sirs_params=state_adv_params)

    # Wrappers
    def state_runner(G, steps_local):
        return ae.run_sirs(G, steps_local, *state_params, initial_infected=10, seed=1)

    def state_adv_runner(G, steps_local):
        return ae.run_sirs(G, steps_local, *state_adv_params, initial_infected=10, seed=2)

    def dynamic_runner(G, steps_local):
        return ae.run_dynamic(G, steps_local, **dynamic_params, seed=3)

    def dynamic_adv_runner(G, steps_local):
        return ae.run_dynamic(G, steps_local, **dynamic_adv_params, seed=4)

    def combined_runner(G, steps_local):
        return ae.run_combined(G, steps_local, **combined_params, seed=5)

    def combined_adv_runner(G, steps_local):
        return ae.run_combined(G, steps_local, **combined_adv_params, seed=6)

    # Pair plots
    run_and_plot_pair('gt_dynamic', dynamic_runner, 'gt_dynamic_adv', dynamic_adv_runner, Gf, steps, out_name='pair_dynamic.png')
    run_and_plot_pair('gt_state', state_runner, 'gt_state_adv', state_adv_runner, Gf, steps, out_name='pair_state.png')
    run_and_plot_pair('gt_state_dynamic', combined_runner, 'gt_state_dynamic_adv', combined_adv_runner, Gf, steps, out_name='pair_combined.png')

    # Triple comparisons (non-adv and adv)
    run_and_plot_triple(['gt_dynamic', 'gt_state', 'gt_state_dynamic'], [dynamic_runner, state_runner, combined_runner], Gf, steps, out_name='triple_nonadv.png')
    run_and_plot_triple(['gt_dynamic_adv', 'gt_state_adv', 'gt_state_dynamic_adv'], [dynamic_adv_runner, state_adv_runner, combined_adv_runner], Gf, steps, out_name='triple_adv.png')

    print('Saved plots to', OUT_DIR)


if __name__ == '__main__':
    main()
