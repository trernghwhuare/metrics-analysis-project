#!/usr/bin/env python3
"""
Generate scatterplot matrices (pairplots) comparing centralities versus degree
for each network CSV in `robust_calibrated/`.

Each plot arranges one scatter (centrality vs degree) per grid cell.
Saves PNGs to an output folder, default `robust_calibrated/plots/`.
"""

import argparse
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kde_centralities(csv_path: Path, out_dir: Path, ncols: int = 3):
    """Plot univariate KDE for each centrality (exclude degree and vertex_id)."""
    df = pd.read_csv(csv_path)
    exclude = {'vertex_id', 'degree'}
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        print(f"No centrality columns found in {csv_path}")
        return None

    n = len(cols)
    nrows = math.ceil(n / ncols)
    # Set a more artistic style
    plt.style.use('seaborn-v0_8')  # Use seaborn style for better aesthetics
    
    # Create figure with explicit white background
    figsize = (ncols * 4.5, nrows * 3.5)  # Increased figure size for better aesthetics
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, 
                           facecolor='white')
    
    # Define a color palette for visual appeal
    colors = sns.color_palette("husl", n_colors=n)
    
    for i, col in enumerate(cols):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        
        # Set white background for the subplot
        ax.set_facecolor('white')
        
        # Drop NaN values for KDE
        values = df[col].dropna()
        if len(values) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', 
                    fontsize=12, color='gray')
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        try:
            # Enhanced KDE plot with artistic styling (outline only, no fill)
            sns.kdeplot(values, fill=False, ax=ax, linewidth=1.5, 
                       alpha=0.8, color=colors[i])
        
        except Exception:
            # Fallback to histogram if KDE fails
            sns.histplot(values, bins=20, ax=ax, stat='density', alpha=0.3, kde=True,
                        color=colors[i], edgecolor='white', linewidth=0.5)

        # Enhanced title and labels
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        
        # Logarithmic x-axis scaling for better visualization of small values
        if len(values) > 0:
            # Check if all values are positive (required for log scale)
            if (values > 0).all():
                ax.set_xscale('log')
                # Set reasonable log limits based on data range
                min_val = values.min()
                max_val = values.max()
                ax.set_xlim(min_val * 5, max_val * 0.5)  
            else:
                # Handle non-positive values with linear scale and dynamic limits
                min_val = values.min()
                max_val = values.max()
                range_val = max_val - min_val
                
                if range_val < 1e-8:  # Extremely small or constant values
                    if abs(max_val) < 1e-10:  # Values extremely close to zero
                        ax.set_xlim(-0.001, 0.001)
                    else:
                        margin = abs(max_val) * 0.3
                        ax.set_xlim(max_val - margin, max_val + margin)
                elif range_val < 1e-4:  # Small ranges
                    margin = max(range_val * 0.3, 1e-4)
                    ax.set_xlim(min_val - margin, max_val + margin)
                else:
                    margin = range_val * 0.2
                    ax.set_xlim(min_val - margin, max_val + margin)
        
        # Improved grid styling
        ax.grid(True, linewidth=0.4, alpha=0.6, linestyle='--')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Thicken bottom and left spines
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_visible(False)

    network_name = csv_path.stem.replace('_robust_calibrated_metrics', '')
    fig.suptitle(f"{network_name} — centrality KDEs", fontsize=16, fontweight='bold')
    
    # Adjust layout with more padding for better aesthetics
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=2.0, w_pad=2.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{network_name}_centrality_kde.png"
    
    # Save with higher DPI and better quality
    fig.savefig(str(out_file), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Plot centrality KDEs for calibrated networks")
    parser.add_argument('--input-dir', default='robust_calibrated', help='Directory with *_robust_calibrated_metrics.csv files')
    parser.add_argument('--pattern', default='*_robust_calibrated_metrics.csv', help='Filename pattern to match CSVs')
    parser.add_argument('--out-dir', default='robust_calibrated/plots_kde', help='Output folder for PNGs')
    parser.add_argument('--ncols', type=int, default=3, help='Number of columns in the plot grid')
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    out_path = Path(args.out_dir)
    files = sorted(input_path.glob(args.pattern))
    if not files:
        print(f"No matching CSV files found in {input_path} with pattern {args.pattern}")
        return

    created = []
    for csv in files:
        try:
            out = plot_kde_centralities(csv, out_path, args.ncols)
            if out:
                print(f"Wrote {out}")
                created.append(out)
        except Exception as e:
            print(f"Failed to plot {csv}: {e}")
    print(f"Done — wrote {len(created)} plot(s) to {out_path}")


if __name__ == '__main__':
    main()