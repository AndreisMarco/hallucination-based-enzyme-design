"""
Plot CA-RMSD vs diffusion sampling steps for Protenix 2025,
one line per target, colored by difficulty tier.

Usage:
    python RMSD_benchmark/plot_diffusion_steps.py \
        --csv path/to/results.csv \
        --out path/to/output.png \
        [--targets PDB1 PDB2 ...]
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TIER_COLORS = {"Short": "#2d7d9a", "Medium": "#d4883a", "Long": "#c0392b"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Restrict to these PDB IDs (default: all)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]

    if args.targets:
        df = df[df["pdb"].isin(args.targets)]

    lengths = df.groupby("pdb")["sequence_length"].first()
    tiers = {}
    for name, l in lengths.items():
        if l < 100:
            tiers[name] = "Short"
        elif l < 170:
            tiers[name] = "Medium"
        else:
            tiers[name] = "Long"

    steps = sorted(df.diffusion_steps.unique())
    targets_sorted = lengths.sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(7, 5))

    for target in targets_sorted:
        tier = tiers[target]
        tdf = df[df.pdb == target].groupby("diffusion_steps")["ca_rmsd"].mean().reindex(steps)
        ax.plot(steps, tdf.values, "o-", color=TIER_COLORS[tier],
                linewidth=1.0, markersize=3, alpha=0.7)

    ax.set_xlabel("Diffusion sampling steps")
    ax.set_ylabel(r"Mean C$\alpha$-RMSD (Å)")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(bottom=0)

    n = {t: sum(1 for v in tiers.values() if v == t) for t in ["Short", "Medium", "Long"]}
    legend_elements = [
        Line2D([0], [0], color=TIER_COLORS["Short"], marker="o", markersize=4,
               linewidth=1.2, label=f"Short (< 100 res, n={n['Short']})"),
        Line2D([0], [0], color=TIER_COLORS["Medium"], marker="o", markersize=4,
               linewidth=1.2, label=f"Medium (100–170 res, n={n['Medium']})"),
        Line2D([0], [0], color=TIER_COLORS["Long"], marker="o", markersize=4,
               linewidth=1.2, label=f"Long (> 170 res, n={n['Long']})"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8,
              frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
