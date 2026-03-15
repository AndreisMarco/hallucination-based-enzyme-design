import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.animation import FFMpegWriter

from mosaic.common import TOKENS

def plot_losses(loss: np.ndarray, additional_losses: dict[np.ndarray] | None = None):
    steps = range(len(loss))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, loss, color="black", linewidth=2.0, label="Total Loss")
    if additional_losses is not None:
        for name, values in additional_losses.items():
            values = np.mean(values, axis=-1)
            if values.mean() < 0:
                values = -values
                name = f"(neg) {name}"
            ax.plot(steps, values, alpha=0.35, linewidth=1.2, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_xlim(0, len(loss) - 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=6, fontsize=8)
    fig.tight_layout()
    return fig

def plot_pssm_heatmap(pssm, ax=None, return_wandb_image: bool = False):
    seq_len = pssm.shape[0]
    aa_labels = list(TOKENS)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 12))
    else:
        fig = ax.figure
        ax.clear()

    ax.imshow(pssm, aspect="auto", cmap='Greys', vmin=0, vmax=1)
    ax.set_title("PSSM")
    ax.set_xticks(np.arange(len(aa_labels)))
    ax.set_xticklabels(aa_labels)
    ax.set_yticks(np.arange(0, seq_len, 10))
    fig.tight_layout()

    if return_wandb_image:
        wandb_image = wandb.Image(fig)
        plt.close(fig)
        return wandb_image
    return fig

def make_pssm_video(pssm_trajectory, output_path: str = "pssm_trajectory.mp4", fps: int = 10):
    n_steps = pssm_trajectory.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(4, 12))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, dpi=100):
        for i in range(n_steps):
            plot_pssm_heatmap(pssm_trajectory[i], ax=ax)
            ax.set_title(f"PSSM — step {i} / {n_steps - 1}")
            writer.grab_frame()
    plt.close(fig)
    return output_path

def _default_is_leaf(x):
    """A list of non-dict items is treated as a trajectory leaf (not traversed further)."""
    return isinstance(x, list) and len(x) > 0 and not isinstance(x[0], dict)

class TrajectoryLogger:
    def __init__(self, is_leaf=None):
        self.trajectory_list = None
        self.trajectory = None
        self.is_leaf = is_leaf if is_leaf is not None else _default_is_leaf

    def _to_cpu(self, x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return np.array(x)
        return x

    @classmethod
    def load(cls, path: str):
        import pickle
        logger = cls()
        with open(path, "rb") as f:
            logger.trajectory = pickle.load(f)
        return logger

    def update(self, aux):
        # Initialize
        if self.trajectory_list is None:
            self.trajectory_list = jax.tree.map(lambda x: [self._to_cpu(x)], aux)
        # Update
        else:
            self.trajectory_list = jax.tree.map(
                lambda traj, new_value: traj + [self._to_cpu(new_value)],
                self.trajectory_list,
                aux,
                is_leaf=self.is_leaf,
            )

    def __add__(self, other):
        merged = TrajectoryLogger(is_leaf=self.is_leaf)
        # Merge based on trajectory_list if available (faster)
        if self.trajectory_list is not None and other.trajectory_list is not None:
            merged.trajectory_list = jax.tree.map(
                lambda a, b: a + b,
                self.trajectory_list,
                other.trajectory_list,
                is_leaf=self.is_leaf,
            )
        # Fall back to trajectory (slower), useful for loaded trajectories
        elif self.trajectory is not None and other.trajectory is not None:
            merged.trajectory = jax.tree.map(
                lambda a, b: np.concatenate([a, b], axis=0) if isinstance(a, np.ndarray) else a,
                self.trajectory,
                other.trajectory,
            )
        else:
            raise RuntimeError(
                    "Both loggers must have trajectory_list or trajectory to be added."
                    "Call clean_trajectory() on each first, or ensure trajectory_list is available."
                )
        return merged

    def clean_trajectory(self, keep_trajectory_list=True):
        if self.trajectory_list is None:
            raise RuntimeError(
                "Logger does not have trajectory_list to be cleaned."
                "If logger was loaded from file, it already has a trajectory."
            )
        # Stack arrays where shape is constant across steps
        def _stack(lst):
            try:
                return np.stack(lst)
            except (ValueError, TypeError):
                return lst
        stacked = jax.tree_util.tree_map(_stack, self.trajectory_list, is_leaf=self.is_leaf)

        # Extract and save the model separately
        model_keys = [k for k in stacked.keys() if k != "optim"]
        if len(model_keys) != 1:
            raise ValueError(
                f"Expected exactly one model key besides 'optim', found: {model_keys}"
            )
        model_name = model_keys[0]
        model_dict = stacked[model_name]

        # merge single valued dictionary of losses into a single dictionary
        losses_list = model_dict.get("losses", [])
        losses = {k: v for d in losses_list for k, v in d.items()}

        self.trajectory = {
            "model":    model_name,
            "losses":   losses,
            "features": model_dict.get("features", {}),
            "optim":    stacked["optim"],
        }
        # Optionally clean memory
        if not keep_trajectory_list:
            self.trajectory_list = None

    def save(self, log_path: str, save_loss_plot: bool = True, save_pssm_video: bool = True):
        import pickle
        if self.trajectory is None:
            self.clean_trajectory()

        # Save trajectory
        with open(os.path.join(log_path, "trajectory.pkl"), "wb") as f:
            pickle.dump(self.trajectory, f)

        # Save plot of losses
        if save_loss_plot:
            fig = plot_losses(loss=self.trajectory["optim"]["loss"],
                                 additional_losses =self.trajectory["losses"])
            fig.savefig(os.path.join(log_path, "losses.png"))
            plt.close(fig)

        # Save pssm evolution video
        if save_pssm_video:
            make_pssm_video(
                self.trajectory["optim"]["pssm"],
                output_path=os.path.join(log_path, "pssm_evolution.mp4"),
            )
        print(f"Saved logs to: {log_path}")

def aux_to_wandb(aux):
    log = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(aux):
        parts = [str(p.key) for p in path if hasattr(p, "key")]
        path_str = ".".join(parts) if parts else "value"
        if "losses" in path_str and isinstance(leaf, (np.ndarray, jnp.ndarray)):
            log[path_str] = float(np.mean(leaf))
        elif "pssm" in path_str:
            log[path_str] = plot_pssm_heatmap(leaf, return_wandb_image=True)
        elif isinstance(leaf, (float, int)) or (hasattr(leaf, "ndim") and leaf.ndim == 0):
            log[path_str] = float(leaf)
    return log