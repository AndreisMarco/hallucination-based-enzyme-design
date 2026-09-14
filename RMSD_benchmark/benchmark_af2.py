"""
Benchmark: CA-RMSD for AF2 monomer predictions on the post-cutoff dataset.

Single-sequence, no MSA, no templates. Runs 5 replicates per target
(different model indices). Saves per-replicate PDBs, a summary CSV,
and a RMSD-vs-target plot.

Usage:
    python RMSD_benchmark/benchmark_af2.py RMSD_benchmark/config_post_cutoff_af2.yaml
"""

import sys
import os
import csv
import time
import shutil
import argparse
from glob import glob
from pathlib import Path

import yaml
import numpy as np
import jax
import jax.numpy as jnp
import gemmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mosaic.models.af2 import AlphaFold2
from mosaic.structure_prediction import TargetChain, StructurePrediction
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.util import calculate_rmsd


def load_chain(chain: gemmi.Chain):
    coords = np.zeros((len(chain), 4, 3))
    for idx in range(len(chain)):
        for atom_idx, atom_name in enumerate(["N", "CA", "C", "O"]):
            try:
                atom = chain[idx].sole_atom(atom_name)
                coords[idx, atom_idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
            except Exception:
                coords[idx, atom_idx] = np.nan
    return gemmi.one_letter_code([r.name for r in chain]), coords

_log_file = None

def log(message: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[AF2 benchmark][{now}] {message}"
    print(line)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def make_summary_plot(csv_path, output_path):
    targets = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["error"]:
                continue
            name = row["pdb"]
            rmsd = float(row["ca_rmsd"])
            targets.setdefault(name, []).append(rmsd)

    names = sorted(targets.keys(), key=lambda n: np.mean(targets[n]))
    means = [np.mean(targets[n]) for n in names]
    stds = [np.std(targets[n]) for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
    x = range(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color="#2d7d9a", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("CA-RMSD (Å)")
    ax.set_title("AF2 Monomer — Single-Sequence Prediction Quality")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_protein(af2, pdb_path, config, results_dir):
    pdb_path = Path(pdb_path)
    target_name = pdb_path.stem
    target_dir = results_dir / target_name

    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_ligands_and_waters()

    chain_id = config.get("chain_id")
    if chain_id is not None:
        chain = st[0][chain_id]
    else:
        chain = st[0][0]

    sequence, gt_backbone = load_chain(chain)
    gt_ca = gt_backbone[:, 1, :]

    if np.any(np.isnan(gt_ca)):
        raise ValueError("Missing CA atoms in ground truth")

    log(f"  Sequence length: {len(sequence)}")

    log(f"  Computing features...")
    features, writer = af2.target_only_features([
        TargetChain(sequence=sequence, use_msa=False)
    ])
    log(f"  Features computed.")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdb_path, target_dir / pdb_path.name)

    recycling_steps = config.get("recycling_steps", 1)
    rows = []
    for replicate in range(config["n_replicates"]):
        key = jax.random.key(config["seed"] + replicate)

        log(f"    rep={replicate}  predicting...", )
        t0 = time.time()
        pred = af2.predict(
            PSSM=None,
            features=features,
            writer=writer,
            recycling_steps=recycling_steps,
            key=key,
        )
        elapsed = time.time() - t0

        pred_ca = jnp.array(pred.model_output.backbone_coordinates[:, 1, :])
        ca_rmsd = float(calculate_rmsd(pred_ca, jnp.array(gt_ca)))
        mean_plddt = float(jnp.mean(pred.plddt))

        pred.save_pdb(target_dir / f"sample_{replicate}.pdb")

        rows.append({
            "pdb": target_name,
            "sequence_length": len(sequence),
            "replicate": replicate,
            "ca_rmsd": f"{ca_rmsd:.4f}",
            "mean_plddt": f"{mean_plddt:.4f}",
            "time_seconds": f"{elapsed:.2f}",
            "error": "",
        })

        log(f"    rep={replicate}  CA-RMSD={ca_rmsd:.3f}  pLDDT={mean_plddt:.3f}  ({elapsed:.1f}s)")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark AF2 monomer single-sequence prediction")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_dir = Path(args.config).parent
    input_dir = config_dir / config["input_dir"]
    results_dir = config_dir / config["experiment_name"]

    csv_path = results_dir / "results.csv"
    plot_path = results_dir / "rmsd_bar.png"

    if results_dir.exists():
        log(f"Experiment directory {results_dir} already exists, regenerating summary plot only.")
        make_summary_plot(csv_path, plot_path)
        log(f"Summary plot saved to {plot_path}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    global _log_file
    _log_file = open(results_dir / "log.txt", "w")

    shutil.copy2(args.config, results_dir / "config.yaml")

    pdb_files = sorted(
        glob(str(input_dir / "*.pdb")) + glob(str(input_dir / "*.cif"))
    )
    max_proteins = config.get("max_proteins")
    if max_proteins is not None:
        pdb_files = pdb_files[:max_proteins]

    log(f"Found {len(pdb_files)} structures in {input_dir}")
    log(f"Replicates: {config['n_replicates']}")

    af2_fixes = config.get("af2_fixes", [])
    log(f"Loading AF2 monomer (fixes={af2_fixes})...")
    af2 = AlphaFold2(multimer=False, use_templates=False, af2_fixes=af2_fixes)
    log("Model loaded.")

    fieldnames = ["pdb", "sequence_length", "replicate",
                  "ca_rmsd", "mean_plddt", "time_seconds", "error"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pdb_path in enumerate(pdb_files):
            name = Path(pdb_path).stem
            log(f"[{i+1}/{len(pdb_files)}] Processing {name}:")

            try:
                rows = process_protein(af2, pdb_path, config, results_dir)
                writer.writerows(rows)
                csvfile.flush()
            except Exception as e:
                log(f"  FAILED: {e}")
                writer.writerow({
                    "pdb": name, "sequence_length": "", "replicate": "",
                    "ca_rmsd": "", "mean_plddt": "",
                    "time_seconds": "", "error": str(e),
                })
                csvfile.flush()

    log(f"Results written to {csv_path}")

    make_summary_plot(csv_path, plot_path)
    log(f"Summary plot saved to {plot_path}")


if __name__ == "__main__":
    main()
