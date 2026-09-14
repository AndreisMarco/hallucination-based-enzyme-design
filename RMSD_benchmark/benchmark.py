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
import equinox as eqx
import gemmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mosaic.models.protenix import Protenix2025, ProtenixMini, ProtenixTiny, ProtenixBase, ProtenixV2

MODELS = {
    "protenix_tiny": ProtenixTiny,
    "protenix_mini": ProtenixMini,
    "protenix_base": ProtenixBase,
    "protenix_2025": Protenix2025,
    "protenix_v2": ProtenixV2,
}
from mosaic.structure_prediction import TargetChain, StructurePrediction
from mosaic.losses.protenix import (
    get_trunk_state,
    protenix_forward_from_trunk,
    biotite_array_to_gemmi_struct,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.losses.protein_mpnn import load_chain
from mosaic.util import calculate_rmsd

_log_file = None

def log(message: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[diffusion steps benchmark][{now}] {message}"
    print(line)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


@eqx.filter_jit
def jit_get_trunk(model, features, recycling_steps, key):
    return get_trunk_state(
        model=model,
        features=features,
        initial_recycling_state=None,
        recycling_steps=recycling_steps,
        key=key,
    )


@eqx.filter_jit
def jit_sample_from_trunk(model, features, initial_embedding, trunk_state, sampling_steps, key):
    return protenix_forward_from_trunk(
        model=model,
        features=features,
        initial_embedding=initial_embedding,
        trunk_state=trunk_state,
        sampling_steps=sampling_steps,
        key=key,
    )


def save_pae_plot(pae, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pae, cmap="bwr", vmin=0, vmax=30)
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title("Predicted Aligned Error")
    fig.colorbar(im, ax=ax, label="PAE (Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_summary_plot(csv_path, output_path):
    targets = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["error"]:
                continue
            name = row["pdb"]
            steps = int(row["diffusion_steps"])
            rmsd = float(row["ca_rmsd"])
            targets.setdefault(name, {}).setdefault(steps, []).append(rmsd)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, steps_data in sorted(targets.items()):
        step_counts = sorted(steps_data.keys())
        means = [np.mean(steps_data[s]) for s in step_counts]
        stds = [np.std(steps_data[s]) for s in step_counts]
        ax.errorbar(step_counts, means, yerr=stds, marker="o", capsize=3, label=name)

    ax.set_xlabel("Diffusion Steps")
    ax.set_ylabel("CA-RMSD (Å)")
    ax.set_title("Structure Prediction Quality vs Diffusion Steps")
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_protein(protenix, pdb_path, config, results_dir):
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

    features, writer = protenix.target_only_features([
        TargetChain(sequence=sequence, use_msa=config.get("use_msa", False))
    ])

    key = jax.random.key(config["seed"])

    log("  Computing trunk embedding...")
    t0 = time.time()
    initial_embedding, trunk_state = jit_get_trunk(
        protenix.protenix, features, config["recycling_steps"], key
    )
    trunk_time = time.time() - t0
    log(f"  Trunk computed in {trunk_time:.1f}s")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdb_path, target_dir / pdb_path.name)

    rows = []
    for n_steps in config["step_counts"]:
        step_dir = target_dir / f"{n_steps}_steps"
        step_dir.mkdir(parents=True, exist_ok=True)

        for replicate in range(config["n_replicates"]):
            sample_key = jax.random.fold_in(jax.random.fold_in(key, n_steps), replicate)

            t0 = time.time()
            output = jit_sample_from_trunk(
                protenix.protenix, features,
                initial_embedding, trunk_state,
                n_steps, sample_key,
            )
            elapsed = time.time() - t0

            pred_ca = jnp.array(output.backbone_coordinates[:, 1, :])
            ca_rmsd = float(calculate_rmsd(pred_ca, jnp.array(gt_ca)))
            mean_plddt = float(jnp.mean(output.plddt))

            iptm = -IPTMLoss()(jnp.zeros((0, 20)), output, key=jax.random.key(0))[0]
            pred_st = biotite_array_to_gemmi_struct(
                writer, np.array(output.structure_coordinates[0])
            )
            prediction = StructurePrediction(
                st=pred_st, plddt=output.plddt, pae=output.pae,
                iptm=float(iptm), model_output=output,
            )
            prediction.save_pdb(step_dir / f"sample_{replicate}.pdb")

            save_pae_plot(
                np.array(output.pae),
                step_dir / f"sample_{replicate}_pae.png",
            )

            rows.append({
                "pdb": target_name,
                "sequence_length": len(sequence),
                "diffusion_steps": n_steps,
                "replicate": replicate,
                "ca_rmsd": f"{ca_rmsd:.4f}",
                "mean_plddt": f"{mean_plddt:.4f}",
                "time_seconds": f"{elapsed:.2f}",
                "error": "",
            })

            log(f"    steps={n_steps:3d}  rep={replicate}  CA-RMSD={ca_rmsd:.3f}  pLDDT={mean_plddt:.3f}  ({elapsed:.1f}s)")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark Protenix2025 diffusion steps")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_dir = Path(args.config).parent
    input_dir = config_dir / config["input_dir"]
    results_dir = config_dir / config["experiment_name"]

    csv_path = results_dir / "results.csv"
    plot_path = results_dir / "rmsd_vs_steps.png"

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
    log(f"Step counts: {config['step_counts']}")
    log(f"Replicates: {config['n_replicates']}")

    model_name = config.get("model", "protenix_2025")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}', choose from: {list(MODELS.keys())}")
    log(f"Loading model: {model_name}...")
    protenix = MODELS[model_name](bf16=config.get("bf16", False))
    log("Model loaded.")

    fieldnames = ["pdb", "sequence_length", "diffusion_steps", "replicate",
                  "ca_rmsd", "mean_plddt", "time_seconds", "error"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pdb_path in enumerate(pdb_files):
            name = Path(pdb_path).stem
            log(f"[{i+1}/{len(pdb_files)}] Processing {name}:")

            try:
                rows = process_protein(protenix, pdb_path, config, results_dir)
                writer.writerows(rows)
                csvfile.flush()
            except Exception as e:
                log(f"  FAILED: {e}")
                writer.writerow({
                    "pdb": name, "sequence_length": "", "diffusion_steps": "",
                    "replicate": "", "ca_rmsd": "", "mean_plddt": "",
                    "time_seconds": "", "error": str(e),
                })
                csvfile.flush()

    log(f"Results written to {csv_path}")

    make_summary_plot(csv_path, plot_path)
    log(f"Summary plot saved to {plot_path}")


if __name__ == "__main__":
    main()
