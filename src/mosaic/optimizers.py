import equinox as eqx
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable
from mosaic.common import is_state_update, has_state_index, LossTerm, LinearCombination
import time
from mosaic.losses.transformations import NoCys, SetPositions

from mosaic.logger import TrajectoryLogger

AbstractLoss = LossTerm | LinearCombination

# ============================================================================
# Loss and gradient computation 
# ============================================================================

# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
def _eval_loss_and_grad(
    loss_function: AbstractLoss, x, key, *, serial_evaluation = False, sample_loss = False
):
    """
    Evaluates the loss function and its gradient.

    Args:
    - loss_function: ...
    - x: soft sequence (N x 20 array with each row in the simplex)
    - key: jax random key
    - serial_evaluation: if True, evaluate each loss function in the list sequentially, to save memory
    - sample_loss: if True *and* loss is a LinearCombination, randomly sample one of the loss functions to evaluate with probability proportional to its weight.
    
    Returns:
    - ((value, aux), g): value of the loss function and auxiliary information, and gradient of the loss with respect to x

    """
    assert not (serial_evaluation and sample_loss), "serial_evaluation and sample_loss cannot both be True"

    if sample_loss:
        assert isinstance(loss_function, LinearCombination), "sample_loss can only be used with LinearCombination loss functions"
        w_total = loss_function.weights.sum()
        idx = jax.random.choice(key, len(loss_function.l), p=loss_function.weights / w_total)
        key = jax.random.fold_in(key, 0)
        return _eval_loss_and_grad(loss_function.l[idx], x, key)


    if serial_evaluation:
        assert isinstance(loss_function, LinearCombination), "serial_evaluation can only be used with LinearCombination loss functions"
        results = [
            (w, _eval_loss_and_grad(l, x, jax.random.fold_in(key, idx)))
            for (idx, (w, l)) in enumerate(zip(loss_function.weights, loss_function.l))
        ]
        v = sum(w * r[0][0] for (w, r) in results)
        aux = [r[0][1] for (w, r) in results]
        g = sum(w * r[1] for (w, r) in results)
        return (v, aux), g
       
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan = 1000000.0), aux), jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))

# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


@eqx.filter_jit
def batched_eval(
    loss: AbstractLoss,
    xs: Float[Array, "B N K"],
    keys: jax.Array,
) -> tuple[Float[Array, "B"], PyTree, Float[Array, "B N K"]]:
    """Evaluate loss+grad for B sequences with B keys."""
    assert xs.ndim == 3, f"xs must be 3D [B, N, K], got {xs.ndim}D"

    def single(x: Float[Array, "N K"], key: jax.Array):
        (v, aux), g = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
        v = jnp.nan_to_num(v, nan=1e6)
        g = jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))
        return v, aux, g

    return jax.vmap(single)(xs, keys)


# this function is a mess, but it's used to update stateful loss functions. see comments in mosaic/common.py
def update_states(aux, loss):
    # Collect new_states and the id of their losses
    state_index_to_update = [(x[0].id, x[1])
                             for x in jax.tree.leaves(aux, is_leaf=is_state_update)
                             if is_state_update(x)]
    
    # for multisample losses, as standard we only keep the first new generated state
    state_index_to_update = {
        (int(k.squeeze()) if isinstance(k, np.ndarray) else int(k)): 
        (v[0] if isinstance(k, np.ndarray) else v)
        for k, v in state_index_to_update
        }

    # get loss terms to update
    def get_modules_to_update(loss):
        return tuple([x
                      for x in jax.tree.leaves(loss, is_leaf=has_state_index)
                      if has_state_index(x)])
    # PyTree surgery to update states
    def replace_fn(module):
        return module.update_state(state_index_to_update[int(module.state_index.id)])
    return eqx.tree_at(get_modules_to_update, loss, replace_fn=replace_fn)

# ============================================================================
# Helper functions 
# ============================================================================

def _print_iter(i, aux, minimal=True):
    def is_scalar_float(x):
        return isinstance(x, (float, jax.Array, np.ndarray)) and jnp.ndim(x) == 0
    metrics = {}
    for path, v in jax.tree_util.tree_leaves_with_path(aux):
        parts = [str(p.key) for p in path if hasattr(p, "key")]
        path_str = ".".join(parts) if parts else "value"
        if minimal and ("optim" not in path_str):
            continue
        if not is_scalar_float(v):
            continue
        if "state_index" in path_str:
            continue
        metrics[path_str] = float(v)
    print(i, " | ".join(f"{k:<5}: {v:>10.2f}" for k, v in metrics.items()))

def _is_model_aux(v):
    if not isinstance(v, dict):
        return False
    keys = v.keys()
    if len(keys) != 1:
        return False
    key = str(*keys)
    if isinstance(v[key], dict) and ("losses" in v[key] and "features" in v[key]):
        return True
    return False

OTHER_LOSSES_KEY = "other_losses"
def standardize_aux(aux):
    standardized = {}

    if isinstance(aux, dict):
        if _is_model_aux(aux):
            return aux
        else:
            return {OTHER_LOSSES_KEY: aux}
    
    elif isinstance(aux, list):
        for i in aux:
            if _is_model_aux(i):
                standardized.update(i)
            elif OTHER_LOSSES_KEY not in standardized:
                standardized[OTHER_LOSSES_KEY] = i
            else:
                standardized[OTHER_LOSSES_KEY].update(i)
    else:
        raise ValueError(f"Invalid aux format, must be either dict or list got {type(aux)} ")

    return standardized

def clean_pssm(PSSM, loss):       
    '''
    Unwraps loss transformations which modify the pssm returning a clean pssm 
    '''                                                                
    if isinstance(loss, NoCys):
        PSSM = NoCys.sequence(PSSM)
        loss = loss.loss                                
    if isinstance(loss, SetPositions):
        PSSM = loss.sequence(seq=PSSM)                                                            
    return PSSM    

# ============================================================================
# Optimizers
# ============================================================================

from scipy.special import softmax, log_softmax 
def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)


def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    alphabet_size: int = 20,
    key: None = None,
    detailed_balance: bool = False,
    fix_loss_key: bool = True,
    serial_evaluation: bool = False,
    log_trajectory: bool = False,
    on_step: Callable | None = None,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, alphabet_size), key=key_model, serial_evaluation=serial_evaluation
    )

    if log_trajectory:
        logger = TrajectoryLogger()

    for iter in range(steps):
        start_time = time.time()
        ### generate a proposal

        for i in range(50):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            path_length = jax.random.randint(
                key=jax.random.key(np.random.randint(10000)),
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
            for _ in range(path_length):
                p, log_p = _proposal(proposal, g_0, proposal_temp, alphabet_size=alphabet_size)
                mut_idx = jax.random.choice(
                    key=key,
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(),
                )
                key = jax.random.fold_in(key, 0)
                position, AA = np.unravel_index(mut_idx, p.shape)
                log_q_forward += log_p[position, AA]
                mutations += [(position, AA)]
                proposal = proposal.at[position].set(AA)
            # check if proposal is same as current sequence
            if np.all(proposal == sequence):
                print(f"\t {i}: proposal is the same as current sequence, skipping.")
                #_print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
                #continue
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa) in mutations])
        print(f"Proposed mutations: {muts}")
        
        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss, jax.nn.one_hot(proposal, alphabet_size), key=key_model if fix_loss_key else key, serial_evaluation=serial_evaluation
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(prop_backward, g_1, proposal_temp, alphabet_size=alphabet_size)
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        print(
            f"iter: {iter}, accept {np.exp(log_acceptance_probability): 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}"
        )

        
        print()
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1
        
        # add optimization info to aux 
        aux = standardize_aux(aux_0)
        aux.update({"optim": {
                "loss": v_0,
                "time": time.time() - start_time,
                "nnz": 1.0,
                "pssm": clean_pssm(jax.nn.one_hot(sequence, alphabet_size), loss),
            }})

        if log_trajectory: 
            logger.update(aux)

        if on_step is not None:
            on_step(iter, aux)

        _print_iter(
            iter,
            aux,
        )

        key = jax.random.fold_in(key, 0)

    if not log_trajectory:
        return sequence 
    else:
        logger.clean_trajectory()
        return sequence, logger


def projection_simplex(V, z=1):
    V = np.array(V, dtype=np.float64)
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key=None,
    max_gradient_norm: float | None = None,
    update_loss_state: bool = False,
    scale=1.0,
    logspace: bool = False,
    serial_evaluation: bool = False,
    sample_loss: bool = False,
    log_trajectory: bool = False,
    on_step: Callable | None = None,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
):
    """
    Accelerated projected gradient descent on the simplex.

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - stepsize: step size for gradient descent
    - momentum: momentum factor
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - update_loss_state: whether to update the loss function state
    - scale: proximal scaling factor for L2 regularization (or entropic regularization if logspace=True), set to > 1.0 to encourage sparsity
    - logspace: whether to optimize in log space, which corresponds to a bregman proximal algorithm.
    - log_trajectory: if True, record the optimization trajectory via the local TrajectoryLogger.
    - on_step: optional callback invoked as on_step(iter, aux) each step.
    - trajectory_fn: optional function (aux, x) -> any; if set, its return value is appended to a trajectory list each step. Independent of log_trajectory.

    returns:
    - (x, best_x) when neither log_trajectory nor trajectory_fn is set.
    - (x, best_x, logger) when log_trajectory is True.
    - (x, best_x, trajectory) when trajectory_fn is set (and log_trajectory is False).
    """
    assert not (log_trajectory and trajectory_fn is not None), \
        "log_trajectory and trajectory_fn are mutually exclusive; pick one"

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[0])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

    x_prev = x

    if log_trajectory:
        logger = TrajectoryLogger()
    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(
            x=v if not logspace else jax.nn.softmax(v),
            loss_function=loss_function,
            key=key,
            serial_evaluation=serial_evaluation,
            sample_loss=sample_loss,
        )

        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale * (v - stepsize * g))

        x_prev = x
        x = x_new

        if value < best_val and not np.isnan(value):
            best_val = value
            best_x = (
                x  # this isn't exactly right, because we evaluated loss at v, not x.
            )

        if update_loss_state:
            loss_function = update_states(aux, loss_function)
        
        # add optimization info to aux 
        aux = standardize_aux(aux)
        average_nnz = (
            (x > 0.01).sum(-1).mean()
            if not logspace
            else (jax.nn.softmax(x) > 0.01).sum(-1).mean()
        )
        aux.update({"optim": {
                "loss": value,
                "nnz": average_nnz,
                "time": time.time() - start_time,
                "pssm": clean_pssm(x, loss_function) if not logspace \
                        else clean_pssm(jax.nn.softmax(x), loss_function),
            }})
        
        if log_trajectory:
            logger.update(aux)

        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        if on_step is not None:
            on_step(_iter, aux)

        _print_iter(
            _iter,
            aux,
        )

    if logspace:
        x = jax.nn.softmax(x)
        best_x = jax.nn.softmax(best_x)

    if log_trajectory:
        logger.clean_trajectory()
        return x, best_x, logger
    if trajectory_fn is not None:
        return x, best_x, trajectory
    return x, best_x


def batched_simplex_APGM(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "B N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key: jax.Array | None = None,
    max_gradient_norm: float | None = None,
    scale: float = 1.0,
    logspace: bool = False,
) -> tuple[Float[Array, "B N 20"], Float[Array, "B N 20"]]:
    """
    Batched accelerated projected gradient descent on the simplex.
    Runs B copies of the optimization in parallel via vmap, where B = x.shape[0].

    Args:
    - loss_function: loss function (same for all designs)
    - x: initial soft sequences [B, N, 20]
    - n_steps: number of optimization steps
    - stepsize: step size (scalar or [B, 1, 1] array for per-design values)
    - momentum: momentum factor (scalar or [B, 1, 1] array)
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - scale: proximal scaling factor
    - logspace: whether to optimize in log space

    returns:
    - x: final soft sequences [B, N, 20]
    - best_x: best soft sequences found during optimization [B, N, 20]
    """
    assert x.ndim == 3, f"x must be 3D [B, N, 20], got {x.ndim}D"
    B = x.shape[0]

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[1])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if not logspace:
        flat = np.array(x).reshape(-1, x.shape[-1])
        x = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

    best_vals = jnp.full(B, jnp.inf)
    best_x = x
    x_prev = x

    for _iter in range(n_steps):
        start_time = time.time()
        v = jnp.array(x + momentum * (x - x_prev), dtype=jnp.float32)
        v_eval = jax.nn.softmax(v, axis=-1) if logspace else v

        values, auxs, grads = batched_eval(loss_function, v_eval, jax.random.split(key, B))

        norms = np.sqrt((grads**2).sum(axis=(-2, -1)))
        clip = np.where(norms > max_gradient_norm, max_gradient_norm / norms, 1.0)
        grads = grads * np.asarray(clip)[:, None, None]

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * grads)
        else:
            flat = np.array(scale * (v - stepsize * grads)).reshape(-1, x.shape[-1])
            x_new = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

        x_prev = x
        x = x_new

        better = (np.array(values) < np.array(best_vals)) & ~np.isnan(values)
        best_vals = jnp.where(jnp.array(better), values, best_vals)
        best_x = jnp.where(jnp.array(better)[:, None, None], x, best_x)

        for i in range(B):
            aux_i = jax.tree.map(lambda v: v[i], auxs)
            average_nnz = (
                (x[i] > 0.01).sum(-1).mean()
                if not logspace
                else (jax.nn.softmax(x[i]) > 0.01).sum(-1).mean()
            )
            # adapted to local _print_iter: metrics under "optim" so they print under minimal=True
            _print_iter(
                f"{_iter}[{i}]",
                {"optim": {"loss": values[i], "nnz": average_nnz, "time": time.time() - start_time}, "": aux_i},
            )

    if logspace:
        x = jax.nn.softmax(x, axis=-1)
        best_x = jax.nn.softmax(best_x, axis=-1)

    return x, best_x


def _topb_unseen_mutations(seq, g, seen, b):
    """Pick up to b 1-hop neighbours of `seq` ranked by first-order predicted delta.

    Returns (candidates, predicted_deltas) with shapes (m, N) and (m,), m <= b.
    Returns None if every 1-hop neighbour has already been seen.
    """
    N, K = g.shape
    a0 = seq.astype(np.int64)
    delta = g - g[np.arange(N), a0][:, None]
    delta[np.arange(N), a0] = np.inf  # mask no-ops

    order = np.argsort(delta.ravel(), kind="stable")

    cands = []
    deltas = []
    for idx in order:
        d = delta.ravel()[idx]
        if not np.isfinite(d):
            break
        pos, aa = divmod(int(idx), K)
        cand = seq.copy()
        cand[pos] = aa
        if cand.tobytes() in seen:
            continue
        cands.append(cand)
        deltas.append(float(d))
        if len(cands) == b:
            break

    if not cands:
        return None
    return np.stack(cands), np.asarray(deltas)


def batch_greedy_descent(
    loss: AbstractLoss,
    sequence: Int[Array, "N"],
    *,
    batch_size: int = 16,
    steps: int = 100,
    alphabet_size: int = 20,
    key: jax.Array | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy batch hillclimb on a discrete sequence.

    Each step: compute the gradient at the current sequence, rank all
    single-point mutations by predicted first-order delta, evaluate the
    top `batch_size` unseen candidates in parallel, and greedily accept
    the best if it improves. Stops early when the full 1-hop neighbourhood
    has been evaluated.

    Args:
    - loss: loss function (called as loss(x, key=...) returning (value, aux))
    - sequence: (N,) int starting sequence
    - batch_size: number of candidate mutations evaluated per step
    - steps: maximum number of steps
    - alphabet_size: token alphabet size
    - key: jax random key (fixed across all evals for deterministic comparison)

    Returns:
    - best_seq: best sequence found
    - best_val: loss at best sequence
    """
    sequence = np.asarray(sequence, dtype=np.int32).copy()
    assert sequence.ndim == 1, f"sequence must be 1D [N], got {sequence.ndim}D"
    B = int(batch_size)

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    # initial eval
    x0 = jax.nn.one_hot(jnp.asarray(sequence[None]), alphabet_size)
    vals, aux0, grads = batched_eval(loss, x0, jnp.broadcast_to(key, (x0.shape[0], *key.shape)))
    v = float(np.asarray(vals)[0])
    g = np.asarray(grads)[0]
    aux = jax.tree.map(lambda a: a[0], aux0)

    # adapted to local _print_iter: metrics under "optim"
    _print_iter("init", {"optim": {"loss": v}, "": aux})

    best_seq = sequence.copy()
    best_val = v
    seen: set[bytes] = {sequence.tobytes()}

    for it in range(steps):
        start_time = time.time()

        picked = _topb_unseen_mutations(sequence, g, seen, B)
        if picked is None:
            print(f"step {it}: neighbourhood exhausted, stopping")
            break
        cands, _ = picked
        m = cands.shape[0]

        xs = jax.nn.one_hot(jnp.asarray(cands), alphabet_size)
        vals, auxs, grads_batch = batched_eval(loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape)))
        vals_np = np.asarray(vals)

        for c in cands:
            seen.add(c.tobytes())

        best_in_batch = int(np.argmin(vals_np[:m]))
        v_best = float(vals_np[best_in_batch])

        if v_best < v:
            sequence = cands[best_in_batch].copy()
            v = v_best
            g = np.asarray(grads_batch)[best_in_batch]
            aux = jax.tree.map(lambda a: a[best_in_batch], auxs)

        if v < best_val:
            best_val = v
            best_seq = sequence.copy()

        # adapted to local _print_iter: metrics under "optim"
        _print_iter(
            it,
            {"optim": {"loss": v, "time": time.time() - start_time}, "": aux},
        )

    return best_seq, best_val
