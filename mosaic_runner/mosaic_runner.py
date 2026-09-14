#!/usr/bin/env python3
"""Mosaic binder design runner — self-contained subprocess entry point.

Built on mosaic_contexts: uses DesignSpace for scaffold/free-design,
registry-based loss building, multi-target contexts.

Layers:
  1. Backend loading
  2. Model loss building & context finalization
  3. PSSM initialization
  4. Stage scheduling (weight + model param tree surgery)
  5. Optimization stages
  6. Reprediction & metrics
  7. run_design orchestrator

Runs inside Mosaic's JAX venv. NO pipeline imports. Only Mosaic + stdlib + gemmi.

Usage: python mosaic_runner.py <config.json>
"""
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
from jax import Array

from mosaic.common import LossTerm, LinearCombination, TOKENS

Loss = LossTerm | LinearCombination
from mosaic.logger import TrajectoryLogger
from mosaic.structure_prediction import StructurePredictionModel, TargetChain

from mosaic_utils import log, sum_losses, schedule_value
from mosaic_contexts import DesignSpace, ContextSpec, build_contexts
from custom_optimizers import simplex_APGM, annealed_design, unindexed_annealed

OPTIMIZERS = {
    "simplex_APGM": simplex_APGM,
    "annealed_design": annealed_design,
    "unindexed_annealed": unindexed_annealed,
}

def _load_backend(cfg: dict) -> StructurePredictionModel:
    base = Path.home()
    models_dir = os.environ.get("MODELS_DIR")
    if models_dir:
        base = Path(models_dir)
        base.mkdir(parents=True, exist_ok=True)

    name = cfg.get("name", "protenix_tiny")
    log(f"Loading backend: {name}")

    if name == "boltz1":
        from mosaic.models.boltz1 import Boltz1
        return Boltz1(cache_path=base / ".boltz" / "boltz1_conf.ckpt")

    elif name == "boltz2":
        from mosaic.models.boltz2 import Boltz2
        return Boltz2(cache_path=base / ".boltz" / "boltz2_conf.ckpt")

    elif name in ("protenix_tiny", "protenix_mini", "protenix_base", "protenix2025", "protenix_v2"):
        if models_dir:
            import protenix.backend as _ptx_backend
            _ptx_backend._CACHE_DIR = str(base / ".protenix")
        from mosaic.models.protenix import ProtenixTiny, ProtenixMini, ProtenixBase, Protenix2025, ProtenixV2
        protenix_models = {
            "protenix_tiny": ProtenixTiny, "protenix_mini": ProtenixMini,
            "protenix_base": ProtenixBase, "protenix2025": Protenix2025,
            "protenix_v2": ProtenixV2,
        }
        return protenix_models[name](bf16=cfg.get("bf16", False))

    elif name in ("af2_multimer", "af2_monomer"):
        from mosaic.models.af2 import AlphaFold2
        af2_fixes = cfg.get("af2_fixes", [])
        use_templates = cfg.get("use_templates", True)
        return AlphaFold2(
            data_dir=str(base / ".alphafold"),
            multimer=(name == "af2_multimer"),
            use_templates=use_templates,
            af2_fixes=af2_fixes,
        )

    else:
        raise ValueError(
            f"Unknown backend: {name}. Available: boltz1, boltz2, protenix_tiny, "
            "protenix_mini, protenix_base, protenix2025, protenix_v2, af2_monomer, af2_multimer"
        )

INIT_METHODS = ("gumbel", "normal", "scaled_normal", "soft_gumbel", "uniform", "esmc")
def _initialize_pssm(
    design_space: DesignSpace,
    key: jax.Array,
    method: str = "gumbel",
) -> tuple[Array, bool]:
    shape = (design_space.n_variable, design_space.n_aa)

    if method == "gumbel":
        scale_key, gumbel_key = jax.random.split(key)
        scale = jax.random.uniform(scale_key, minval=0.25, maxval=0.75)
        return scale * jax.random.gumbel(key=gumbel_key, shape=shape), True

    elif method == "normal":
        return jax.random.normal(key=key, shape=shape), True

    elif method == "scaled_normal":
        return 0.01 * jax.random.normal(key=key, shape=shape), True

    elif method == "soft_gumbel":
        key_normal, key_gumbel = jax.random.split(key)
        x = 0.01 * jax.random.normal(key=key_normal, shape=shape)
        g = jax.random.gumbel(key=key_gumbel, shape=shape)
        return jax.nn.softmax(x + g, axis=-1), False

    elif method == "uniform":
        return jnp.ones(shape) / design_space.n_aa, False

    elif method == "esmc":
        from esmc_initialization import initialize_esmc_pssm
        pssm = initialize_esmc_pssm(sequence=design_space.sequence, key=key)
        if not design_space.allow_cys:
            pssm = jnp.delete(pssm, 4, axis=1)
        return pssm, True

    raise ValueError(f"Unknown initialization: {method}. Available: {INIT_METHODS}")


def _initialize_assignment(design_space: DesignSpace, key: jax.Array) -> Array | None:
    if design_space.is_indexed or not design_space.has_scaffold:
        return None
    shape = (design_space.design_length, design_space.n_motif)
    return 0.01 * jax.random.normal(key=key, shape=shape)


@dataclass
class PredictionContext:
    spec: ContextSpec
    features: Any
    writer: Any
    name: str

def _build_stage_loss(
    contexts: List[PredictionContext],
    model: StructurePredictionModel,
    design_space: DesignSpace,
    model_loss_cfg: Dict[str, Any],
    stage_idx: int,
) -> tuple[Loss, List[Loss]]:
    build_fn = (
        model.build_multisample_loss
        if hasattr(model, "build_multisample_loss")
        else model.build_loss
    )
    stage_model_cfg = {k: schedule_value(v, stage_idx) for k, v in model_loss_cfg.items()}

    structure_losses = []
    model_losses = []
    for ctx in contexts:
        weights = jnp.array([
            schedule_value(ctx.spec.weight_schedules.get(t.name, 1.0), stage_idx)
            for t in ctx.spec.loss_terms
        ])
        structure_loss = LinearCombination(l=ctx.spec.loss_terms, weights=weights)
        structure_losses.append(structure_loss)
        stage_model_cfg["name"] = ctx.name
        model_losses.append(
            build_fn(loss=structure_loss, features=ctx.features, **stage_model_cfg)
        )

    return design_space.wrap_loss(sum_losses(*model_losses)), structure_losses


def build_model_contexts(
    model: StructurePredictionModel,
    specs: List[ContextSpec],
    design_length: int,
    backend_name: str,
    binder_sequence: Optional[str] = None,
) -> List[PredictionContext]:
    binder_kwargs = {}
    if binder_sequence is not None:
        sig = inspect.signature(model.binder_features)
        if "binder_chain" in sig.parameters:
            binder_kwargs["binder_chain"] = TargetChain(sequence=binder_sequence, use_msa=False)
        else:
            log(f"Warning: {type(model).__name__} does not support binder_chain — output PDBs will have UNK residue names")

    contexts = []
    for spec in specs:
        features, writer = model.binder_features(
            binder_length=design_length,
            chains=spec.target_chains,
            **binder_kwargs,
        )
        contexts.append(PredictionContext(
            spec=spec, features=features, writer=writer,
            name=f"{backend_name}_{spec.name}",
        ))
    return contexts


def _predict_and_evaluate(
    model: StructurePredictionModel,
    structure_loss: Loss,
    hard_pssm: Array,
    ctx: PredictionContext,
    design_length: int,
    key: jax.Array,
    predict_kwargs: Optional[Dict[str, Any]] = None,
    assignment: Array | None = None,
) -> tuple[Any, Dict[str, Any]]:
    if predict_kwargs is None:
        predict_kwargs = {}

    prediction = model.predict(
        PSSM=hard_pssm,
        features=ctx.features,
        key=key,
        writer=ctx.writer,
        **predict_kwargs,
    )

    loss_key = jax.random.fold_in(key, 1)
    loss_kwargs = {}
    if assignment is not None:
        loss_kwargs["assignment"] = assignment
    loss_value, loss_aux = structure_loss(hard_pssm, prediction.model_output, key=loss_key, **loss_kwargs)

    metrics = {
        "iptm": float(prediction.iptm) if ctx.spec.num_target_chains > 0 else 0.0,
        "mean_plddt": float(jnp.mean(prediction.plddt)),
        "mean_binder_plddt": float(jnp.mean(prediction.plddt[:design_length])),
        "loss": float(loss_value),
    }

    flat_aux = {}
    for d in loss_aux:
        flat_aux.update(d)
    metrics["loss_terms"] = {
        k: float(v) for k, v in flat_aux.items()
        if jnp.ndim(v) == 0
    }
    return prediction, metrics


def _optimize(
    contexts: List[PredictionContext],
    model: StructurePredictionModel,
    design_space: DesignSpace,
    model_loss_cfg: Dict[str, Any],
    optimization_cfg: Dict[str, Any],
    pssm: Array,
    is_logspace: bool,
    key: jax.Array,
    assignment: Array | None = None,
) -> tuple[Array, str, List[Loss], Optional[TrajectoryLogger], jax.Array, Array | None]:
    stages = optimization_cfg["stages"]
    n_stages = len(stages)

    for ctx in contexts:
        for name, value in ctx.spec.weight_schedules.items():
            if isinstance(value, list) and len(value) != n_stages:
                raise ValueError(f"Weight schedule '{name}' has {len(value)} values but there are {n_stages} stages")
    for name, value in model_loss_cfg.items():
        if isinstance(value, list) and len(value) != n_stages:
            raise ValueError(f"Model loss schedule '{name}' has {len(value)} values but there are {n_stages} stages")

    log_trajectory = bool(optimization_cfg.get("log_trajectory", True))
    logger = None

    for stage_idx, stage_cfg in enumerate(stages):
        stage_name = stage_cfg.get("name", f"stage_{stage_idx}")
        key, stage_key = jax.random.split(key)

        total_loss, _ = _build_stage_loss(
            contexts=contexts,
            model=model,
            design_space=design_space,
            model_loss_cfg=model_loss_cfg,
            stage_idx=stage_idx,
        )

        optimizer_name = stage_cfg.get("optimizer")
        if optimizer_name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {sorted(OPTIMIZERS)}")
        optimizer_fn = OPTIMIZERS[optimizer_name]

        needs_logspace = optimizer_name == "annealed_design" or stage_cfg.get("logspace", False)
        if not is_logspace and needs_logspace:
            pssm = jnp.log(pssm + 1e-5)

        log(f"Stage {stage_idx + 1}/{n_stages} '{stage_name}' optimizer={optimizer_name}")
        skip_keys = ("name", "optimizer", "use_best")
        kwargs = {k: v for k, v in stage_cfg.items() if k not in skip_keys}
        kwargs["loss_function"] = total_loss
        kwargs["x"] = pssm
        kwargs["key"] = stage_key
        kwargs["log_trajectory"] = log_trajectory
        if assignment is not None:
            kwargs["assignment"] = assignment
            kwargs["gt_distogram"] = design_space._scaffold.distogram()
            if stage_cfg.get("commit_assignment"):
                motif_pssm = design_space._scaffold.pssm()
                if not design_space.allow_cys:
                    cys_idx = TOKENS.index("C")
                    motif_pssm = jnp.concatenate([motif_pssm[:, :cys_idx], motif_pssm[:, cys_idx + 1:]], axis=-1)
                positions = jnp.argmax(assignment, axis=0)
                if is_logspace:
                    pssm = pssm.at[positions].set(10.0 * motif_pssm)
                else:
                    pssm = pssm.at[positions].set(motif_pssm)
                kwargs["x"] = pssm
                log(f"Set motif sequence at positions {positions.tolist()}")

        result = optimizer_fn(**kwargs)

        if assignment is not None and len(result) >= 4:
            final, best = result[0], result[1]
            final_assign, best_assign = result[2], result[3]
            stage_logger = result[4] if log_trajectory and len(result) > 4 else None
            use_best = bool(stage_cfg.get("use_best", False))
            assignment = best_assign if use_best else final_assign
        else:
            final, best = result[0], result[1]
            stage_logger = result[2] if log_trajectory and len(result) > 2 else None

        use_best = bool(stage_cfg.get("use_best", False))
        pssm = best if use_best else final

        is_logspace = needs_logspace

        if stage_logger is not None:
            logger = stage_logger if logger is None else logger + stage_logger

    total_loss, structure_losses = _build_stage_loss(
        contexts=contexts,
        model=model,
        design_space=design_space,
        model_loss_cfg=model_loss_cfg,
        stage_idx=n_stages - 1,
    )
    full_pssm = design_space.expand_pssm(pssm, total_loss)
    sequence_indices = jnp.argmax(full_pssm, axis=-1)
    hard_pssm = jax.nn.one_hot(sequence_indices, 20)
    binder_sequence = "".join([TOKENS[int(i)] for i in sequence_indices[:design_space.design_length]])
    log(f"Designed sequence: {binder_sequence}")

    return hard_pssm, binder_sequence, structure_losses, logger, key, assignment


def run_design(config: Dict) -> None:
    design_length = int(config.get("binder_length", 75))
    optimization_cfg = config.get("optimization", {})
    allow_cys = bool(optimization_cfg.get("allow_cys", True))

    design_space = DesignSpace(
        design_length=design_length,
        scaffold_cfg=config.get("scaffold"),
        allow_cys=allow_cys,
    )
    log(f"DesignSpace: design_length={design_space.design_length}, n_variable={design_space.n_variable}, has_scaffold={design_space.has_scaffold}")

    specs = build_contexts(
        target_cfgs=config.get("targets", []),
        design_space=design_space,
        shared_loss_terms=config.get("shared_loss_terms"),
    )

    backend_cfg = config.get("backend", {"name": "boltz2"})
    model = _load_backend(cfg=backend_cfg)
    backend_name = backend_cfg.get("name", "protenix_tiny")

    model_loss_cfg = config.get("model_loss", {})
    contexts = build_model_contexts(
        model=model,
        specs=specs,
        design_length=design_length,
        backend_name=backend_name)

    log(f"Built contexts, optimization will include {len(contexts)} context(s):")
    for ctx in contexts:
        log(f"{ctx.spec.name}: num_target_chains={ctx.spec.num_target_chains}, losses={[loss.name for loss in ctx.spec.loss_terms]}")

    key = jax.random.key(config.get("seed", 0))
    init_method = optimization_cfg["initialization"]
    key, pssm_key, assign_key = jax.random.split(key, 3)
    pssm, is_logspace = _initialize_pssm(design_space=design_space, key=pssm_key, method=init_method)
    assignment = _initialize_assignment(design_space=design_space, key=assign_key)
    log(f"Initialized PSSM: shape={pssm.shape}, method={init_method}, logspace={is_logspace}")
    if assignment is not None:
        log(f"Initialized assignment: shape={assignment.shape} (unindexed scaffolding)")

    hard_pssm, binder_sequence, structure_losses, logger, key, assignment = _optimize(
        contexts=contexts,
        model=model,
        design_space=design_space,
        model_loss_cfg=model_loss_cfg,
        optimization_cfg=optimization_cfg,
        pssm=pssm,
        is_logspace=is_logspace,
        key=key,
        assignment=assignment,
    )

    out_dir = Path(config.get("out_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    design_id = config.get("design_id", "design_0")

    # Rebuild contexts with designed sequence so PDBs have correct residue names
    contexts = build_model_contexts(
        model=model, specs=specs, design_length=design_length,
        backend_name=backend_name, binder_sequence=binder_sequence,
    )

    # Opt prediction — always performed, uses optimization model
    key, predict_key = jax.random.split(key)
    log("Running opt prediction (recycling_steps=1)")
    all_opt_metrics = {}
    for ctx, structure_loss in zip(contexts, structure_losses):
        prediction, metrics = _predict_and_evaluate(
            model=model,
            structure_loss=structure_loss,
            hard_pssm=hard_pssm,
            ctx=ctx,
            design_length=design_length,
            key=predict_key,
            predict_kwargs={"recycling_steps": 1},
            assignment=assignment,
        )
        opt_pdb_name = f"{design_id}_{ctx.spec.name}_opt.pdb"
        prediction.save_pdb(str(out_dir / opt_pdb_name))
        all_opt_metrics[ctx.spec.name] = metrics
        log(f"Opt '{ctx.spec.name}': iptm={metrics['iptm']:.3f}, plddt={metrics['mean_binder_plddt']:.3f}, loss={metrics['loss']:.3f}")

    structure_path = f"{design_id}_{contexts[0].spec.name}_opt.pdb"

    # Reprediction with specified backend
    repredict_cfg = config.get("repredict")
    if repredict_cfg is not None:
        repredict_backend_cfg = repredict_cfg["backend"]
        repredict_model_loss_cfg = repredict_cfg.get("model_loss", {})

        repredict_backend_name = repredict_backend_cfg.get("name", "unknown")
        log(f"Loading repredict backend: {repredict_backend_name}")
        del model
        import gc; gc.collect()
        repredict_model = _load_backend(cfg=repredict_backend_cfg)
        repredict_contexts = build_model_contexts(
            model=repredict_model,
            specs=specs,
            design_length=design_length,
            backend_name=repredict_backend_name,
            binder_sequence=binder_sequence,
        )

        _, repredict_structure_losses = _build_stage_loss(
            contexts=repredict_contexts,
            model=repredict_model,
            design_space=design_space,
            model_loss_cfg=repredict_model_loss_cfg,
            stage_idx=0,
        )

        key, predict_key = jax.random.split(key)
        all_repredict_metrics = {}
        for ctx, structure_loss in zip(repredict_contexts, repredict_structure_losses):
            log(f"Repredicting context '{ctx.spec.name}' with {repredict_model_loss_cfg}")
            prediction, metrics = _predict_and_evaluate(
                model=repredict_model,
                structure_loss=structure_loss,
                hard_pssm=hard_pssm,
                ctx=ctx,
                design_length=design_length,
                key=predict_key,
                predict_kwargs=repredict_model_loss_cfg,
                assignment=assignment,
            )
            repredict_pdb_name = f"{design_id}_{ctx.spec.name}.pdb"
            prediction.save_pdb(str(out_dir / repredict_pdb_name))
            all_repredict_metrics[ctx.spec.name] = metrics
            log(f"Repredict '{ctx.spec.name}': iptm={metrics['iptm']:.3f}, plddt={metrics['mean_binder_plddt']:.3f}, loss={metrics['loss']:.3f}")

        structure_path = f"{design_id}_{repredict_contexts[0].spec.name}.pdb"

    # Save trajectory
    if logger is not None:
        logger.save(out_dir / design_id)

    # Manifest
    import json
    design_entry = {
        "design_id": design_id,
        "structure_path": structure_path,
        "metadata": {
            "binder_sequence": binder_sequence,
            "binder_length": design_length,
            "motif_mapping": design_space.motif_metadata(assignment=assignment),
        },
        "reprediction_opt": {
            "backend": backend_name,
            "contexts": all_opt_metrics,
        },
    }
    if repredict_cfg is not None:
        design_entry["reprediction"] = {
            "backend": repredict_backend_name,
            "contexts": all_repredict_metrics,
        }

    manifest = {"designs": [design_entry]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"Done: {design_id}")


if __name__ == "__main__":
    import json

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run_design(config)
