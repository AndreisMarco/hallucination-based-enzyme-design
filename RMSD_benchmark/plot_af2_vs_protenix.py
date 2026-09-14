"""
Plot per-target CA-RMSD comparison between Protenix 2025 (S=1) and AF2 monomer.

Usage:
    python RMSD_benchmark/plot_af2_vs_protenix.py \
        --protenix-csv path/to/protenix_results.csv \
        --af2-csv path/to/af2_results.csv \
        --out path/to/output.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protenix-csv", type=Path, required=True)
    parser.add_argument("--af2-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    p25 = pd.read_csv(args.protenix_csv)
    p25 = p25[p25["error"].isna() | (p25["error"] == "")]
    p25 = p25[p25.diffusion_steps == 1]

    af2 = pd.read_csv(args.af2_csv)
    af2 = af2[af2["error"].isna() | (af2["error"] == "")]

    p25_s = p25.groupby("pdb").agg(
        rmsd=("ca_rmsd", "mean"), std=("ca_rmsd", "std"),
        length=("sequence_length", "first"))
    af2_s = af2.groupby("pdb").agg(
        rmsd=("ca_rmsd", "mean"), std=("ca_rmsd", "std"),
        length=("sequence_length", "first"))

    common = sorted(p25_s.index.intersection(af2_s.index),
                    key=lambda n: af2_s.loc[n, "length"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(common))
    w = 0.35

    ax.bar(x - w / 2,
           [p25_s.loc[n, "rmsd"] for n in common],
           yerr=[p25_s.loc[n, "std"] for n in common],
           width=w, label="Protenix 2025", color="#c0392b",
           capsize=2, edgecolor="none")
    ax.bar(x + w / 2,
           [af2_s.loc[n, "rmsd"] for n in common],
           yerr=[af2_s.loc[n, "std"] for n in common],
           width=w, label="AF2 Monomer", color="#2d7d9a",
           capsize=2, edgecolor="none")

    labels = [f'{n}\n({int(af2_s.loc[n, "length"])})' for n in common]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel(r"Mean C$\alpha$-RMSD (Å)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=9, frameon=True)
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
