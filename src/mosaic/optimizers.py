import equinox as eqx
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool
from typing import Callable, Tuple, List, Any
from mosaic.common import is_state_update, has_state_index, LossTerm, LinearCombination
from abc import ABC, abstractmethod
import wandb
import time
from dataclasses import dataclass

from mosaic.logger import TrajectoryLogger, aux_to_wandb

AbstractLoss = LossTerm | LinearCombination


def _print_iter(i, aux):
    """Print scalar metrics from aux to the terminal."""
    def is_scalar_float(x):
        return isinstance(x, (float, jax.Array, np.ndarray)) and jnp.ndim(x) == 0
    metrics = {
        jax.tree_util.keystr(k, simple=True, separator='.'): float(v)
        for k, v in jax.tree_util.tree_leaves_with_path(aux)
        if is_scalar_float(v)
        and "state_index" not in jax.tree_util.keystr(k, simple=True, separator=".")
    }
    print(i, " | ".join(f"{k:<5}: {v:>10.2f}" for k, v in metrics.items()))


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


class PSSMOptimizer(ABC):
    def __init__(self,
                loss_fn, 
                n_steps: int = 50,
                update_mask: Bool[Array, "N"] | None = None,
                max_gradient_norm: float | None = None,
                update_loss_state: bool = False,
                serial_evaluation: bool = False,
                sample_loss: bool = False,
                log_trajectory: bool = False,
                wandb_project: str | None = None
                ):

        self.loss_fn = loss_fn
        self.n_steps = n_steps
        self.update_mask = update_mask
        self.max_gradient_norm = max_gradient_norm
        self.update_loss_state = update_loss_state
        self.serial_evaluation = serial_evaluation
        self.sample_loss = sample_loss
        self.log_trajectory = log_trajectory
        self.wandb_project = wandb_project

    @abstractmethod
    def step(self, state, key):
        pass

    def run(self, 
            pssm_init: Float[Array, "N 20"],
            key,
            traj_logger: TrajectoryLogger | None = None,
            ):
        
        update_mask = self.update_mask if self.update_mask is not None \
                                       else jnp.ones(shape=(pssm_init.shape[0],), dtype=bool)
        if self.max_gradient_norm is None:
            self.max_gradient_norm = np.sqrt(pssm_init.shape[0])

        if self.log_trajectory:
            traj_logger = TrajectoryLogger()

        if self.wandb_project is not None and wandb.run is None:
            raise NotImplementedError(f"wandb support for single optimizer run is not directly supported, " \
                                      "the desired behavior can be achieved by wrapping the optimizer in MultiPhaseOptimization")

        state = {"x": pssm_init, "mask": update_mask,}
        best_loss = np.inf
        best_pssm = pssm_init
        for i in range(self.n_steps):
            start_time = time.time()

            state, loss, aux = self.step(state, key)
            key = jax.random.fold_in(key, i)

            if self.update_loss_state:
                self.loss_fn = update_states(aux, self.loss_fn)

            if loss < best_loss and not np.isnan(loss):
                best_loss = loss
                best_pssm = state["x"]

            aux.update({"optim": {
                "loss": loss,
                "nnz": (state["x"] > 0.01).sum(-1).mean(),
                "time": time.time() - start_time,
                "pssm": state["x"],
            }})

            if self.log_trajectory:
                traj_logger.update(aux=aux)

            if wandb.run is not None:
                wandb.log(aux_to_wandb(aux))

            _print_iter(i, aux)

        return state["x"], best_pssm, traj_logger

    def clip_gradient(self, g):
        n = np.sqrt((g**2).sum())
        if n > self.max_gradient_norm:
            g = g * (self.max_gradient_norm / n)
        return g
    
    def make_wandb_config(self):
        return {
            "opt_configs": {k: v for k, v in vars(self).items() if isinstance(v, (int, float, str, bool))},
            "losses": {str(l).strip('()'): float(w) \
                           for l, w in zip(self.loss_fn.loss.l, self.loss_fn.loss.weights)}
        }


class SimplexAPGM(PSSMOptimizer):
    def __init__(self, stepsize, scale=1.0, momentum=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stepsize = stepsize
        self.momentum = momentum
        self.scale = scale

    def step(self, state, key):
        if "x_prev" not in state:
            state["x"] = projection_simplex(state["x"]) # ensure input is on simplex
            state["x_prev"] = state["x"]

        x = state["x"]
        x_prev = state["x_prev"]
        v = jax.device_put(x + self.momentum * (x - x_prev))

        (loss, aux), g = _eval_loss_and_grad(
            loss_function=self.loss_fn,
            x=v, # v is already in simplex space
            key=key,
            serial_evaluation=self.serial_evaluation,
            sample_loss=self.sample_loss,
        )

        g = self.clip_gradient(g) * state["mask"][:, None]
        new_x = projection_simplex(self.scale * (v - self.stepsize * g)) # ensure new_x stays on simplex
        state["x"] = new_x
        state["x_prev"] = x

        return state, loss, aux


class LogitAPGM(PSSMOptimizer):
    def __init__(self, stepsize, scale=1.0, momentum=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stepsize = stepsize
        self.momentum = momentum
        self.scale = scale

    def step(self, state, key):
        if "x_prev_logit" not in state:
            state["x_logit"] = jax.nn.log_softmax(state["x"]) # convert to logit space
            state["x_prev_logit"] = state["x_logit"]

        x_logit = state["x_logit"]
        x_prev_logit = state["x_prev_logit"]
        v = jax.device_put(x_logit + self.momentum * (x_logit - x_prev_logit))

        (loss, aux), g = _eval_loss_and_grad(
            loss_function=self.loss_fn,
            x=jax.nn.softmax(v), # evaluation works in simplex space
            key=key,
            serial_evaluation=self.serial_evaluation,
            sample_loss=self.sample_loss,
        )

        g = self.clip_gradient(g) * state["mask"][:, None]
        new_x_logit = self.scale * (v - self.stepsize * g)
        state["x_logit"] = new_x_logit
        state["x_prev_logit"] = x_logit
        state["x"] = jax.nn.softmax(new_x_logit, axis=-1) # save new_x in simplex space

        return state, loss, aux

@dataclass
class Phase:
    optimizer: PSSMOptimizer
    name: str

class MultiPhaseOptimization:
    def __init__(self, 
                 phases: List[Phase],
                 log_trajectory: bool = False,
                 wandb_project: str | None = None):
        
        self.phases = phases
        self.log_trajectory = log_trajectory
        self.wandb_project = wandb_project

        if self.log_trajectory:
            for phase in self.phases:
                if phase.optimizer.log_trajectory is False:
                    raise RuntimeError(f"If log_trajectory=True all optimizers must have log_trajectory=True, got {phase.optimizer.log_trajectory} for phase'{phase.name}'")
                
        if self.wandb_project is not None:
            for phase in self.phases:
                if phase.optimizer.wandb_project != self.wandb_project:
                    raise RuntimeError(f"If using wandb, all optimizers must share the same wandb_project, got {phase.optimizer.wandb_project} for phase '{phase.name}'")

    def run(self,
            pssm_init: Float[Array, "N 20"], 
            key: Any | None = None):

        if key is None: 
            key = jax.random.key(np.random.randint(10000))

        if self.wandb_project is not None:
            wandb.init(
                project=self.wandb_project,
                config=self.make_wandb_config()
            )

        current_pssm = pssm_init
        traj_loggers = []
        all_final = np.zeros(shape=(len(self.phases), *pssm_init.shape))
        all_best = np.zeros_like(all_final)
        for i, phase in enumerate(self.phases):
            print(f"Starting phase {phase.name} - ({i+1}/{len(self.phases)})")
            current_pssm, best_pssm, logger  = phase.optimizer.run(
                pssm_init=current_pssm, 
                key=key)
            
            traj_loggers.append(logger)
            all_final[i] = current_pssm
            all_best[i] = best_pssm
            key = jax.random.fold_in(key, i)
        
        full_logger = None
        if self.log_trajectory:
            full_logger = traj_loggers[0]
            for logger in traj_loggers[1:]:
                full_logger += logger
            full_logger.clean_trajectory()

        if wandb.run is not None:
            wandb.finish()

        all = {"final": all_final, "best": all_best}
        return all["final"][-1], all, full_logger

    def make_wandb_config(self):
        config = {}
        for phase in self.phases:
            config[phase.name] = phase.optimizer.make_wandb_config()
        return config
    
# def _proposal(sequence, g, temp, alphabet_size: int = 20):
#     input = jax.nn.one_hot(sequence, alphabet_size)
#     g_i_x_i = (g * input).sum(-1, keepdims=True)
#     logits = -((input * g).sum() - g_i_x_i + g) / temp
#     return jax.nn.softmax(logits), jax.nn.log_softmax(logits)

# rewrite in numpy to use float64
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
        
        _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
        

        key = jax.random.fold_in(key, 0)

    return sequence