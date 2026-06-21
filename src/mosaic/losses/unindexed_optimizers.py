"""Unindexed scaffolding optimizer.

Fork of colabdesign_optimizer that co-optimizes a soft assignment matrix [N, M]
alongside the PSSM [N, 20]. The two are kept as separate arrays — the PSSM
flows through the standard loss pipeline while the assignment is placed on
UnindexedRMSD via set_assignment before each forward pass.
"""
from __future__ import annotations

import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from mosaic.logger import TrajectoryLogger
from mosaic.losses.unindexed_scaffolding import set_assignment
from mosaic.optimizers import (
    _print_iter,
    clean_pssm,
    standardize_aux,
)


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


@eqx.filter_jit
def _eval_unindexed(loss, pssm_params, assign_params, key,
                    alpha, temp, soft_weight, hard_weight, assign_temp):
    """Forward pass with separate PSSM and assignment gradients."""
    def forward(pssm_params, assign_params):
        scaled = pssm_params * alpha
        soft = jax.nn.softmax(scaled / temp)
        hard = _ste_argmax(soft)
        pseudo = soft_weight * soft + (1 - soft_weight) * pssm_params
        pseudo = hard_weight * hard + (1 - hard_weight) * pseudo

        assignment = jax.nn.softmax(assign_params / assign_temp, axis=0)
        loss_with_assign = set_assignment(loss, assignment)
        return loss_with_assign(pseudo, key=key)

    (v, aux), (g_pssm, g_assign) = jax.value_and_grad(
        forward, argnums=(0, 1), has_aux=True
    )(pssm_params, assign_params)
    v = jnp.nan_to_num(v, nan=1e6)
    g_pssm = jnp.nan_to_num(g_pssm)
    g_assign = jnp.nan_to_num(g_assign)
    return (v, aux), g_pssm, g_assign


def unindexed_colabdesign_optimizer(
    *,
    loss_function,
    pssm: Float[Array, "N 20"],
    assign: Float[Array, "N M"],
    n_steps: int,
    learning_rate: float = 0.1,
    assign_learning_rate: float | None = None,
    alpha: float = 2.0,
    temp: float = 1.0,
    e_temp: float = 0.01,
    soft: float = 0.0,
    e_soft: float = 1.0,
    hard: float = 0.0,
    e_hard: float = 1.0,
    step: float = 1.0,
    e_step: float | None = None,
    assign_temp: float = 1.0,
    e_assign_temp: float = 0.01,
    normalize_gradient: bool = True,
    max_gradient_norm: float | None = None,
    key=None,
    patience: int | None = None,
    log_trajectory: bool = False,
    on_step: Callable | None = None,
):
    """ColabDesign optimizer with co-optimized assignment matrix.

    Manages PSSM [N, 20] and assignment [N, M] as separate arrays with
    independent schedules. The PSSM gets ColabDesign soft/hard/temp blending;
    the assignment gets temperature-annealed softmax over positions (axis=0).

    Args:
        loss_function: Mosaic loss (eqx.Module with UnindexedRMSD inside).
        pssm: Initial PSSM logits [N, 20].
        assign: Initial assignment logits [N, M].
        n_steps: Total optimization steps.
        learning_rate: Base LR for PSSM.
        assign_learning_rate: Base LR for assignment (defaults to learning_rate).
        alpha: Logit scaling factor for PSSM.
        temp / e_temp: PSSM temperature schedule (quadratic decay).
        soft / e_soft: Soft blending schedule (linear).
        hard / e_hard: Hard (STE) blending schedule (linear).
        step / e_step: Step multiplier schedule (linear).
        assign_temp / e_assign_temp: Assignment temperature schedule (quadratic decay).
        normalize_gradient: Normalize gradient to unit norm.
        max_gradient_norm: Gradient clipping threshold.
        key: JAX random key.
        patience: Early stopping patience.
        log_trajectory: Return TrajectoryLogger.
        on_step: Per-step callback(iter, aux).

    Returns:
        (final_pssm, best_pssm, final_assign, best_assign)
        or (final_pssm, best_pssm, final_assign, best_assign, logger).
    """
    n_variable = pssm.shape[0]

    if assign_learning_rate is None:
        assign_learning_rate = learning_rate
    if e_step is None:
        e_step = step
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    lr_pssm = learning_rate * np.sqrt(n_variable)
    lr_assign = assign_learning_rate * np.sqrt(n_variable)

    pssm_params = np.array(pssm, dtype=np.float32)
    assign_params = np.array(assign, dtype=np.float32)
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
        step_i = step + (e_step - step) * t
        assign_temp_i = e_assign_temp + (assign_temp - e_assign_temp) * (1 - t) ** 2

        lr_scale = step_i * ((1 - soft_i) + (soft_i * temp_i))
        lr_pssm_i = lr_pssm * lr_scale
        lr_assign_i = lr_assign * lr_scale

        (value, aux), g_pssm, g_assign = _eval_unindexed(
            loss_function,
            jax.device_put(pssm_params),
            jax.device_put(assign_params),
            key, alpha,
            jnp.float32(temp_i), jnp.float32(soft_i),
            jnp.float32(hard_i), jnp.float32(assign_temp_i),
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
        assign_params = np.array(assign_params - lr_assign_i * g_assign, dtype=np.float32)

        value = float(value)
        if value < best_val and not np.isnan(value):
            best_val = value
            best_pssm = pssm_params.copy()
            best_assign = assign_params.copy()
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        pssm_probs = jax.nn.softmax(jnp.array(pssm_params) * alpha / temp_i)
        assign_probs = jax.nn.softmax(jnp.array(assign_params) / assign_temp_i, axis=0)

        pssm_entropy = float(-(pssm_probs * jnp.log(pssm_probs + 1e-10)).sum(axis=-1).mean())
        assign_entropy = float(-(assign_probs * jnp.log(assign_probs + 1e-10)).sum(axis=0).mean())
        average_nnz = float((pssm_probs > 0.01).sum(-1).mean())

        aux = standardize_aux(aux)
        aux.update({"optim": {
            "loss": value,
            "nnz": average_nnz,
            "entropy": pssm_entropy,
            "assign_entropy": assign_entropy,
            "assign_temp": assign_temp_i,
            "temp": temp_i,
            "soft": soft_i,
            "hard": hard_i,
            "lr": lr_pssm_i,
            "grad_norm": g_pssm_raw,
            "grad_norm_assign": g_assign_raw,
            "time": time.time() - start_time,
            "pssm": clean_pssm(pssm_probs, loss_function),
            "assign": np.array(assign_probs),
        }})

        if log_trajectory:
            logger.update(aux)

        if on_step is not None:
            on_step(i, aux)

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
