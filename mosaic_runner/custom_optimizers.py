"""Custom optimizers for Mosaic v2.

Contains:
- simplex_APGM: Accelerated projected gradient on the simplex (with optional logspace mode).
- annealed_design: ColabDesign-style annealed optimizer with soft/hard/temp schedules.
- unindexed_annealed: Co-optimizes PSSM + assignment matrix for unindexed scaffolding.
"""
from __future__ import annotations

import time
from itertools import permutations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from mosaic.logger import TrajectoryLogger
from mosaic.optimizers import (
    _print_iter,
    clean_pssm,
    projection_simplex,
    standardize_aux,
)


# ============================================================================
# Shared helpers
# ============================================================================

def _normalize_gradient(g, max_gradient_norm=None):
    g = np.array(g, dtype=np.float32)
    g_norm = np.linalg.norm(g, axis=(-1, -2), keepdims=True)
    if max_gradient_norm is None:
        return g / (g_norm + 1e-7)
    if g_norm > max_gradient_norm:
        return g * (max_gradient_norm / (g_norm + 1e-7))
    return g


def _ste_argmax(x):
    hard = jax.nn.one_hot(jnp.argmax(x, axis=-1), x.shape[-1])
    return jax.lax.stop_gradient(hard - x) + x


# ============================================================================
# Simplex APGM (Mosaic-style)
# ============================================================================

@eqx.filter_jit
def _eval_simplex(loss, x, key, center_gradient=True):
    x = jnp.array(x, dtype=jnp.float32)
    (v, aux), g = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
    v = jnp.nan_to_num(v, nan=1e6)
    g = jnp.nan_to_num(g)
    if center_gradient:
        g = g - g.mean(axis=-1, keepdims=True)
    return (v, aux), g


def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key,
    normalize_gradient: bool = False,
    max_gradient_norm: float | None = None,
    scale: float = 1.0,
    e_scale: float | None = None,
    logspace: bool = False,
    alpha: float = 1.0,
    patience: int | None = None,
    log_trajectory: bool = False,
    **kwargs,
):
    n_variable = x.shape[0]
    stepsize = stepsize * np.sqrt(n_variable)

    if e_scale is None:
        e_scale = scale

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

    x_prev = x

    if log_trajectory:
        logger = TrajectoryLogger()

    iterations_since_improvement = 0
    for _iter in range(n_steps):
        start_time = time.time()

        t = (_iter + 1) / n_steps
        scale_i = scale + (e_scale - scale) * t

        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_simplex(
            loss_function,
            v if not logspace else jax.nn.softmax(v * alpha),
            key,
            center_gradient=not logspace,
        )

        g = np.array(g, dtype=np.float32)
        grad_norm_raw = float(np.linalg.norm(g))

        if normalize_gradient:
            g = _normalize_gradient(g, max_gradient_norm)

        grad_norm_effective = float(np.linalg.norm(g))

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale_i * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale_i * (v - stepsize * g))

        x_prev = x
        x = x_new

        if value < best_val and not np.isnan(value):
            best_val = value
            best_x = x
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        probs = x if not logspace else jax.nn.softmax(x * alpha)
        aux = standardize_aux(aux)
        average_nnz = float((probs > 0.01).sum(-1).mean())
        entropy = float(-(probs * jnp.log(probs + 1e-10)).sum(axis=-1).mean())
        aux.update({"optim": {
            "loss": value,
            "nnz": average_nnz,
            "entropy": entropy,
            "scale": float(scale_i),
            "stepsize": stepsize,
            "grad_norm": grad_norm_raw,
            "grad_norm_effective": grad_norm_effective,
            "time": time.time() - start_time,
            "pssm": clean_pssm(probs, loss_function),
        }})

        if log_trajectory:
            logger.update(aux)

        _print_iter(_iter, aux, value)

        if patience is not None and iterations_since_improvement >= patience:
            print(f"Early stopping after {_iter + 1} iterations "
                  f"(no improvement for {patience} iterations)")
            break

    if log_trajectory:
        logger.clean_trajectory()
        return x, best_x, logger

    return x, best_x


# ============================================================================
# Annealed design (ColabDesign-style)
# ============================================================================

@eqx.filter_jit
def _eval_annealed(loss, x, key, alpha, temp, soft_weight, hard_weight):
    def forward(x):
        soft = jax.nn.softmax(x * alpha / temp)
        hard = _ste_argmax(soft)
        pseudo = soft_weight * soft + (1 - soft_weight) * x
        pseudo = hard_weight * hard + (1 - hard_weight) * pseudo
        return loss(pseudo, key=key)

    (v, aux), g = eqx.filter_value_and_grad(forward, has_aux=True)(x)
    v = jnp.nan_to_num(v, nan=1e6)
    g = jnp.nan_to_num(g)
    return (v, aux), g


def annealed_design(
    *,
    loss_function,
    x: Float[Array, "N K"],
    n_steps: int,
    stepsize: float = 0.1,
    alpha: float = 2.0,
    temp: float = 1.0,
    e_temp: float = 0.01,
    soft: float = 0.0,
    e_soft: float = 1.0,
    hard: float = 0.0,
    e_hard: float = 1.0,
    normalize_gradient: bool = True,
    max_gradient_norm: float | None = None,
    key,
    log_trajectory: bool = False,
    **kwargs,
):
    n_variable = x.shape[0]

    stepsize = stepsize * np.sqrt(n_variable)

    params = np.array(x, dtype=np.float32)
    best_val = np.inf
    best_params = params.copy()

    if log_trajectory:
        logger = TrajectoryLogger()

    for i in range(n_steps):
        start_time = time.time()
        t = (i + 1) / n_steps

        temp_i = e_temp + (temp - e_temp) * (1 - t) ** 2   # quadratic
        soft_i = soft + (e_soft - soft) * t                  # linear
        hard_i = hard + (e_hard - hard) * t                  # linear

        step_scale = (1 - soft_i) + (soft_i * temp_i)
        stepsize_i = stepsize * step_scale

        (value, aux), g = _eval_annealed(
            loss_function, jax.device_put(params), key,
            alpha, jnp.float32(temp_i),
            jnp.float32(soft_i), jnp.float32(hard_i),
        )

        g = np.array(g, dtype=np.float32)
        grad_norm_raw = float(np.linalg.norm(g))

        if normalize_gradient:
            g = _normalize_gradient(g, max_gradient_norm)

        grad_norm_effective = float(np.linalg.norm(g))

        key = jax.random.fold_in(key, 0)

        params = np.array(params - stepsize_i * g, dtype=np.float32)

        value = float(value)
        if value < best_val and not np.isnan(value):
            best_val = value
            best_params = params.copy()

        probs = jax.nn.softmax(jnp.array(params) * alpha / temp_i)
        entropy = float(-(probs * jnp.log(probs + 1e-10)).sum(axis=-1).mean())
        average_nnz = float((probs > 0.01).sum(-1).mean())
        aux = standardize_aux(aux)
        aux.update({"optim": {
            "loss": value,
            "nnz": average_nnz,
            "entropy": entropy,
            "temp": temp_i,
            "soft": soft_i,
            "hard": hard_i,
            "stepsize": stepsize_i,
            "grad_norm": grad_norm_raw,
            "grad_norm_effective": grad_norm_effective,
            "time": time.time() - start_time,
            "pssm": clean_pssm(jnp.array(params), loss_function),
        }})

        if log_trajectory:
            logger.update(aux)

        _print_iter(i, aux, value)

    final = jnp.array(params)
    best = jnp.array(best_params)

    if log_trajectory:
        logger.clean_trajectory()
        return final, best, logger

    return final, best


# ============================================================================
# Unindexed annealed (co-optimizes PSSM + assignment)
# ============================================================================


def _extract_aux_field(aux, field_name):
    """Extract a named field from the aux tree, handling model-wrapped aux."""
    if isinstance(aux, dict):
        if field_name in aux:
            v = np.array(aux[field_name])
            return v[0] if v.ndim > 1 else v
        for val in aux.values():
            result = _extract_aux_field(val, field_name)
            if result is not None:
                return result
    elif isinstance(aux, (list, tuple)):
        for item in aux:
            result = _extract_aux_field(item, field_name)
            if result is not None:
                return result
    return None


@eqx.filter_jit
def _eval_unindexed(loss, pssm_params, assign_params, key,
                    alpha, temp, soft_weight, hard_weight, assign_temp,
                    commit_assignment=False):
    def forward(pssm_params, assign_params):
        soft = jax.nn.softmax(pssm_params * alpha / temp)
        hard = _ste_argmax(soft)
        pseudo = soft_weight * soft + (1 - soft_weight) * pssm_params
        pseudo = hard_weight * hard + (1 - hard_weight) * pseudo
        if commit_assignment:
            assignment = jax.lax.stop_gradient(
                jax.nn.one_hot(jnp.argmax(assign_params, axis=0), assign_params.shape[0]).T
            )
        else:
            assignment = jax.nn.softmax(assign_params / assign_temp, axis=0)
        return loss(pseudo, key=key, assignment=assignment)

    (v, aux), (g_pssm, g_assign) = jax.value_and_grad(
        forward, argnums=(0, 1), has_aux=True
    )(pssm_params, assign_params)
    v = jnp.nan_to_num(v, nan=1e6)
    g_pssm = jnp.nan_to_num(g_pssm)
    if commit_assignment:
        frozen = jnp.argmax(assign_params, axis=0)
        g_pssm = g_pssm.at[frozen].set(0.0)
    g_assign = jnp.nan_to_num(g_assign)
    return (v, aux), g_pssm, g_assign


def unindexed_annealed(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    assignment: Float[Array, "N M"],
    n_steps: int,
    stepsize: float = 0.1,
    assign_stepsize: float | None = None,
    alpha: float = 2.0,
    temp: float = 1.0,
    e_temp: float = 0.01,
    soft: float = 0.0,
    e_soft: float = 1.0,
    hard: float = 0.0,
    e_hard: float = 1.0,
    assign_temp: float = 1.0,
    e_assign_temp: float = 0.01,
    geo_stepsize: float = 0.0,
    e_geo_stepsize: float | None = None,
    commit_assignment: bool = False,
    normalize_gradient: bool = True,
    max_gradient_norm: float | None = None,
    key,
    patience: int | None = None,
    log_trajectory: bool = False,
    **kwargs,
):
    n_variable = x.shape[0]
    M = assignment.shape[1]

    if assign_stepsize is None:
        assign_stepsize = stepsize
    if e_geo_stepsize is None:
        e_geo_stepsize = geo_stepsize

    lr_pssm = stepsize * np.sqrt(n_variable)
    lr_assign = assign_stepsize * np.sqrt(n_variable)

    pssm_params = np.array(x, dtype=np.float32)
    assign_params = np.array(assignment, dtype=np.float32)

    best_val = np.inf
    best_pssm = pssm_params.copy()
    best_assign = assign_params.copy()

    if log_trajectory:
        logger = TrajectoryLogger()

    iterations_since_improvement = 0

    for i in range(n_steps):
        start_time = time.time()
        t = (i + 1) / n_steps

        temp_i = e_temp + (temp - e_temp) * (1 - t) ** 2
        soft_i = soft + (e_soft - soft) * t
        hard_i = hard + (e_hard - hard) * t
        assign_temp_i = e_assign_temp + (assign_temp - e_assign_temp) * (1 - t) ** 2
        geo_lr_i = geo_stepsize + (e_geo_stepsize - geo_stepsize) * t

        lr_scale = (1 - soft_i) + (soft_i * temp_i)
        lr_pssm_i = lr_pssm * lr_scale
        lr_assign_i = lr_assign * lr_scale

        (value, aux), g_pssm, g_assign = _eval_unindexed(
            loss_function,
            jax.device_put(pssm_params),
            jax.device_put(assign_params),
            key, alpha,
            jnp.float32(temp_i), jnp.float32(soft_i),
            jnp.float32(hard_i), jnp.float32(assign_temp_i),
            commit_assignment=commit_assignment,
        )

        g_pssm_raw = float(np.linalg.norm(np.array(g_pssm, dtype=np.float32)))
        g_assign_raw = float(np.linalg.norm(np.array(g_assign, dtype=np.float32)))

        if normalize_gradient:
            g_pssm = _normalize_gradient(g_pssm, max_gradient_norm)
            g_assign = _normalize_gradient(g_assign, max_gradient_norm)
        else:
            g_pssm = np.array(g_pssm, dtype=np.float32)
            g_assign = np.array(g_assign, dtype=np.float32)

        key = jax.random.fold_in(key, 0)

        pssm_params = np.array(pssm_params - lr_pssm_i * g_pssm, dtype=np.float32)

        if not commit_assignment:
            assign_params = np.array(assign_params - lr_assign_i * g_assign, dtype=np.float32)

            # Geometric vote: nudge assignment toward distogram-matched positions
            if geo_lr_i > 0:
                geo_idxs = _extract_aux_field(aux, "geo_motif_idxs")
                if geo_idxs is not None:
                    geo_idxs = np.array(geo_idxs, dtype=int)
                    assign_probs = np.array(jax.nn.softmax(
                        assign_params / max(assign_temp_i, 0.01), axis=0
                    ))
                    best_score = -1.0
                    best_perm_idxs = geo_idxs
                    for perm in permutations(range(M)):
                        perm_idxs = geo_idxs[list(perm)]
                        score = assign_probs[perm_idxs, np.arange(M)].sum()
                        if score > best_score:
                            best_score = score
                            best_perm_idxs = perm_idxs
                    geo_vote = np.zeros_like(assign_params)
                    geo_vote[best_perm_idxs, np.arange(M)] = 1.0
                    assign_params = assign_params + geo_lr_i * geo_vote

        value = float(value)
        if value < best_val and not np.isnan(value):
            best_val = value
            best_pssm = pssm_params.copy()
            best_assign = assign_params.copy()
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        pssm_probs = jax.nn.softmax(jnp.array(pssm_params) * alpha / temp_i)
        if commit_assignment:
            assign_probs = jnp.array(assign_params)
        else:
            assign_probs = jax.nn.softmax(jnp.array(assign_params) / assign_temp_i, axis=0)

        pssm_entropy = float(-(pssm_probs * jnp.log(pssm_probs + 1e-10)).sum(axis=-1).mean())
        assign_entropy = 0.0 if commit_assignment else float(-(assign_probs * jnp.log(assign_probs + 1e-10)).sum(axis=0).mean())
        average_nnz = float((pssm_probs > 0.01).sum(-1).mean())

        aux = standardize_aux(aux)
        aux.update({"optim": {
            "loss": value,
            "nnz": average_nnz,
            "entropy": pssm_entropy,
            "assign_entropy": assign_entropy if not commit_assignment else 0.0,
            "assign_temp": assign_temp_i,
            "temp": temp_i,
            "soft": soft_i,
            "hard": hard_i,
            "lr": lr_pssm_i,
            "grad_norm": g_pssm_raw,
            "grad_norm_assign": g_assign_raw,
            "committed": float(commit_assignment),
            "time": time.time() - start_time,
            "pssm": clean_pssm(pssm_probs, loss_function),
            "assign": np.array(assign_probs),
        }})

        if log_trajectory:
            logger.update(aux)

        _print_iter(i, aux, value)

        if patience is not None and iterations_since_improvement >= patience:
            print(f"Early stopping at step {i + 1} "
                  f"(no improvement for {patience} steps)")
            break

    final_pssm = jnp.array(pssm_params)
    final_assign = jnp.array(assign_params)
    best_pssm_out = jnp.array(best_pssm)
    best_assign_out = jnp.array(best_assign)

    if log_trajectory:
        logger.clean_trajectory()
        return final_pssm, best_pssm_out, final_assign, best_assign_out, logger

    return final_pssm, best_pssm_out, final_assign, best_assign_out
