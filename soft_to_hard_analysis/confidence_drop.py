#!/usr/bin/env python
"""Soft→hard confidence-drop test (the `argmax(soft)` test).

For each design trajectory, take the soft optimum at the end of phase 1
(`logits_to_soft`, step 100 → index 99: soft=1, temp=1, hard=0) and measure how
much pLDDT drops when the soft PSSM is replaced by its hard `argmax` one-hot — with
*identical, deterministic* forward settings so the drop is input-driven, not
sampling/dropout noise.

Hypothesis: the drop is much larger for Protenix (diffusion structure module
over-rates soft blends) than for AF2 (single-pass structure module, soft↔hard
consistent). Runs entirely within mosaic (no `bench` dependency). From lib/mosaic/:

    # Several Protenix runs in one go — model loaded + predict compiled ONCE, reused
    # across all dirs (each must share the same backend + scaffold); one CSV per dir.
    .venv/bin/python soft_to_hard_analysis/confidence_drop.py \
        --backend protenix2025 --bf16 --num_keys 4 --batch 8 \
        --exp_dirs ../../outputs/moving_to_protenix/10_pairformer_dropout \
                   ../../outputs/moving_to_protenix/11_dropout_soft_hard_blend \
                   ../../outputs/moving_to_protenix/05b_increase_plddt_weight_5

    # AF2 reference (12_fix_dropout), average over the 5 models, dropout off
    .venv/bin/python soft_to_hard_analysis/confidence_drop.py \
        --backend af2_monomer --af2_fixes target_feat,msa_feat --batch 8 \
        --exp_dirs ../../outputs/chasing_colabdesign/12_fix_dropout

Notes
- The logged `trajectory["optim"]["pssm"]` is the *raw params* (clean_pssm), so the
  soft input the model saw is softmax(alpha * params / temp); the hard input is
  one_hot(argmax(params)). temp is read from the trajectory (=1.0 at the soft
  optimum); alpha is the config value (default 2.0).
- pLDDT here is mean over binder positions (matches PLDDTLoss). AF2 and Protenix
  pLDDT have different calibrations — compare the *drop* Δ = soft − hard, and read
  it qualitatively across models, not as identical absolute scales.
- Determinism: AF2 averages over all `num_models` with dropout off; Protenix
  averages over `--num_keys` diffusion seeds (trunk is deterministic). Soft and
  hard use the *same* keys/models, so Δ isolates the input change.
"""
import argparse
import csv
import glob
import os
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.indexed_scaffolding as is_

# mosaic ships its own copy of the scaffold structure
DEFAULT_PDB = str(Path(__file__).resolve().parents[1] / "structures" / "1LNS.pdb")


def load_backend(name, bf16=False, af2_fixes=(), use_templates=False):
    """Mosaic-native model construction (same idiom as RMSD_benchmark's MODELS dict).

    No `bench` dependency — `bench._load_backend` is just a thin wrapper over these
    same constructors.
    """
    if name == "protenix2025":
        from mosaic.models.protenix import Protenix2025
        return Protenix2025(bf16=bf16)
    if name in ("af2_monomer", "af2_multimer"):
        from mosaic.models.af2 import AlphaFold2
        return AlphaFold2(data_dir=str(Path.home() / ".alphafold"),
                          multimer=(name == "af2_multimer"),
                          use_templates=use_templates, af2_fixes=list(af2_fixes))
    raise ValueError(f"Unsupported backend: {name} (use protenix2025 | af2_monomer | af2_multimer)")


def load_step_pssm(exp_dir: str, step: int):
    """Return (design_ids, raw_params[D,N,20], temps[D]) at 0-based index step-1."""
    paths = sorted(glob.glob(os.path.join(exp_dir, "**", "mosaic_logs", "trajectory.pkl"),
                             recursive=True))
    if not paths:
        raise FileNotFoundError(f"no trajectory.pkl under {exp_dir}")
    idx = step - 1
    ids, params, temps = [], [], []
    for p in paths:
        traj = pickle.load(open(p, "rb"))
        pssm = np.asarray(traj["optim"]["pssm"])          # [n_steps, N, 20]
        temp = np.asarray(traj["optim"]["temp"])
        if idx >= pssm.shape[0]:
            print(f"  skip (only {pssm.shape[0]} steps): {p}")
            continue
        ids.append(Path(p).parent.parent.name)
        params.append(pssm[idx])
        temps.append(float(temp[idx]))
    return ids, np.stack(params), np.array(temps, dtype=np.float32)


def make_batched_predict(model, features, binder_len, backend, sampling_steps, num_keys):
    """Return eqx.filter_jit fn: (pssms[B,N,20], key) -> mean-binder-pLDDT[B]."""
    is_af2 = backend.startswith("af2")

    def plddt_one(pssm, key):
        if is_af2:
            # deterministic ensemble: average over all models, dropout off
            def per_model(m):
                smo = model.model_output(PSSM=pssm, features=features,
                                         recycling_steps=1, model_idx=m,
                                         use_dropout=False, key=key)
                return smo.plddt[:binder_len].mean()
            return jax.vmap(per_model)(jnp.arange(model.num_models)).mean()
        else:
            # average over num_keys diffusion seeds (deterministic trunk)
            def per_key(k):
                smo = model.model_output(PSSM=pssm, features=features,
                                         recycling_steps=1, sampling_steps=sampling_steps,
                                         key=k)
                return smo.plddt[:binder_len].mean()
            return jax.vmap(per_key)(jax.random.split(key, num_keys)).mean()

    @eqx.filter_jit
    def batched(pssms, key):
        keys = jax.random.split(key, pssms.shape[0])
        return jax.vmap(plddt_one)(pssms, keys)

    return batched


def run_in_batches(fn, pssms, batch, seed=0):
    """Apply fn over pssms[D,N,20] in fixed-size chunks (pad last to avoid recompile)."""
    D = pssms.shape[0]
    out = np.zeros(D, dtype=np.float32)
    key = jax.random.key(seed)
    for s in range(0, D, batch):
        chunk = pssms[s:s + batch]
        n = chunk.shape[0]
        if n < batch:  # pad to keep a single compiled shape
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - n, axis=0)], axis=0)
        key, sub = jax.random.split(key)
        vals = np.asarray(fn(jnp.asarray(chunk), sub))[:n]
        out[s:s + n] = vals
        print(f"  predicted {s + n}/{D}")
    return out


def process_exp_dir(exp_dir, predict, args):
    """Run the soft/hard test for one experiment dir; write its CSV; return summary."""
    ids, params, temps = load_step_pssm(exp_dir, args.step)
    if not ids:
        print(f"[{exp_dir}] no usable trajectories at step {args.step} — skipping")
        return None
    if params.shape[-2] != args.binder_length:
        raise ValueError(f"[{exp_dir}] PSSM length {params.shape[-2]} != binder_length "
                         f"{args.binder_length}; all --exp_dirs must share the scaffold.")
    print(f"[{exp_dir}] {len(ids)} designs, step {args.step} (idx {args.step - 1}); "
          f"temp[step]={temps.mean():.3f}")

    soft = jax.nn.softmax(args.alpha * jnp.asarray(params) / temps[:, None, None], axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(jnp.asarray(params), axis=-1), params.shape[-1])

    print("  predicting SOFT ...")
    plddt_soft = run_in_batches(predict, np.asarray(soft), args.batch, seed=args.seed)
    print("  predicting HARD (argmax) ...")
    plddt_hard = run_in_batches(predict, np.asarray(hard), args.batch, seed=args.seed)
    delta = plddt_soft - plddt_hard

    out = os.path.join(exp_dir, args.csv_name)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design_id", "plddt_soft", "plddt_hard", "delta"])
        for i, did in enumerate(ids):
            w.writerow([did, f"{plddt_soft[i]:.4f}", f"{plddt_hard[i]:.4f}", f"{delta[i]:.4f}"])
    print(f"  -> {out}")
    return {"exp_dir": exp_dir, "n": len(ids), "out": out,
            "soft": plddt_soft, "hard": plddt_hard, "delta": delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dirs", required=True, nargs="+",
                    help="one or more experiment output dirs (each contains .../mosaic_logs/"
                         "trajectory.pkl). All must share the same backend + scaffold so the "
                         "compiled predict fn is reused across them.")
    ap.add_argument("--backend", required=True, help="af2_monomer | af2_multimer | protenix2025 | ...")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--af2_fixes", default="", help="comma list, e.g. target_feat,msa_feat")
    ap.add_argument("--use_templates", action="store_true")
    # scaffold (defaults match the 1LNS motif-scaffolding runs)
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--keep_intervals", default="347-349,467-469,497-499")
    ap.add_argument("--loops", default="30,20,30,30")
    ap.add_argument("--order", default="0,1,2")
    ap.add_argument("--binder_length", type=int, default=120)
    ap.add_argument("--use_msa", action="store_true")
    ap.add_argument("--use_template", action="store_true", help="scaffold template (af2)")
    # test point + forward settings
    ap.add_argument("--step", type=int, default=100, help="optimizer step (1-based); soft optimum = end of phase 1")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--sampling_steps", type=int, default=2, help="protenix diffusion steps (default = model default)")
    ap.add_argument("--num_keys", type=int, default=4, help="protenix: diffusion seeds to average per design")
    ap.add_argument("--batch", type=int, default=8, help="designs per vmap batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv_name", default="soft_to_hard_confidence.csv",
                    help="CSV filename written into each experiment dir")
    args = ap.parse_args()

    # --- model + base features + compiled predict (built ONCE, reused across dirs) ---
    model = load_backend(args.backend, bf16=args.bf16,
                         af2_fixes=[f for f in args.af2_fixes.split(",") if f],
                         use_templates=args.use_templates)

    scaffold = is_.Scaffold(path_to_structure=args.pdb, length=args.binder_length,
                           keep_intervals=args.keep_intervals,
                           order=[int(x) for x in args.order.split(",")],
                           loops=[int(x) for x in args.loops.split(",")])
    chain = scaffold.build_chain(use_msa=args.use_msa, use_template=args.use_template)
    features, _ = model.binder_features(args.binder_length, [], binder_chain=chain)

    # Compiled once on the first batch; reused for every dir (same backend/scaffold/batch),
    # so we pay JAX compilation only once across all --exp_dirs.
    predict = make_batched_predict(model, features, args.binder_length,
                                   args.backend, args.sampling_steps, args.num_keys)

    summaries = []
    for exp_dir in args.exp_dirs:
        s = process_exp_dir(exp_dir, predict, args)
        if s is not None:
            summaries.append(s)

    print("\n================ soft→hard pLDDT drop ================")
    print(f"  backend: {args.backend}{' (bf16)' if args.bf16 else ''}   step: {args.step}")
    print(f"  {'experiment':<48} {'n':>4} {'soft':>8} {'hard':>8} {'Δ':>8} {'medianΔ':>9}")
    for s in summaries:
        name = os.path.basename(os.path.normpath(s["exp_dir"]))
        print(f"  {name:<48} {s['n']:>4} {s['soft'].mean():>8.4f} {s['hard'].mean():>8.4f} "
              f"{s['delta'].mean():>8.4f} {np.median(s['delta']):>9.4f}")
    print("======================================================")

if __name__ == "__main__":
    main()
