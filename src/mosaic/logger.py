import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.animation import FFMpegWriter

from mosaic.common import TOKENS

def _default_is_leaf(x):
    """A list of non-dict items is treated as a trajectory leaf (not traversed further)."""
    return isinstance(x, list) and len(x) > 0 and not isinstance(x[0], dict)

def pssm_heatmap(pssm, return_wandb_image: bool = False):
    seq_len = pssm.shape[0]
    aa_labels = list(TOKENS)

    fig, ax = plt.subplots(1, 1, figsize=(4, 12))
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
    return fig  # never used ATM

def pssm_trajectory_video(pssm_trajectory, output_path: str = "pssm_trajectory.mp4", fps: int = 10):
    n_steps = pssm_trajectory.shape[0]
    aa_labels = list(TOKENS)
    seq_len = pssm_trajectory.shape[1]
    # Initialize first frame
    fig, ax = plt.subplots(1, 1, figsize=(4, 12))
    im = ax.imshow(pssm_trajectory[0], aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(aa_labels)))
    ax.set_xticklabels(aa_labels)
    ax.set_yticks(np.arange(0, seq_len, 10))
    title = ax.set_title(f"PSSM — step 0 / {n_steps - 1}")
    fig.tight_layout()
    # Iterate through steps 
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, dpi=100):
        for i in range(n_steps):
            im.set_data(pssm_trajectory[i])
            title.set_text(f"PSSM — step {i} / {n_steps - 1}")
            writer.grab_frame()

    plt.close(fig)
    return output_path

def aux_to_wandb(aux):
    log = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(aux):
        parts = [str(p.key) for p in path if hasattr(p, "key")]
        path_str = ".".join(parts) if parts else "value"
        if "losses" in path_str and isinstance(leaf, (np.ndarray, jnp.ndarray)):
            log[path_str] = float(np.mean(leaf))
        elif "pssm" in path_str:
            log[path_str] = pssm_heatmap(leaf, return_wandb_image=True)
        elif leaf.dim == 0 or isinstance(leaf, (float, int)):
            log[path_str] = float(leaf)
    return log


class TrajectoryLogger:
    def __init__(self, is_leaf=None):
        self.history = None
        self.clean_history = None
        self.is_leaf = is_leaf if is_leaf is not None else _default_is_leaf

    def _to_cpu(self, x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return np.array(x)
        return x

    def update(self, aux):
        # Initialize
        if self.history is None:
            self.history = jax.tree.map(lambda x: [self._to_cpu(x)], aux)
        # Update
        else:
            self.history = jax.tree.map(
                lambda traj, new_value: traj + [self._to_cpu(new_value)],
                self.history,
                aux,
                is_leaf=self.is_leaf,
            )

    def clean_trajectory(self):
        # Stack arrays where shape is constant across steps
        def _stack(lst):
            try:
                return np.stack(lst)
            except (ValueError, TypeError):
                return lst
        stacked = jax.tree_util.tree_map(_stack, self.history, is_leaf=self.is_leaf)

        # Extract and save the model separatly
        model_keys = [k for k in stacked.keys() if k != "optim"]
        if len(model_keys) != 1:
            raise ValueError(
                f"Expected exactly one model key besides 'optim', found: {model_keys}"
            )
        model_name = model_keys[0]
        model_dict = stacked[model_name]

        # merge single valued dicitonary of losses into a single dictionary
        losses_list = model_dict.get("losses", [])
        losses = {k: v for d in losses_list for k, v in d.items()}

        self.clean_history = {
            "model":    model_name,
            "losses":   losses,
            "features": model_dict.get("features", {}),
            "optim":    stacked["optim"],
        }

    def plot_losses(self):
        if self.clean_history is None:
            self.clean_trajectory()

        losses = self.clean_history["losses"]
        overall_loss = self.clean_history["optim"]["loss"]
        steps = np.arange(len(overall_loss))

        fig, ax = plt.subplots(figsize=(10, 4))
        for name, values in losses.items():
            if np.ndim(values) == 1:
                ax.plot(steps, values, alpha=0.35, linewidth=1.2, label=name)
        ax.plot(steps, overall_loss, color="black", linewidth=2.0,
                alpha=1.0, label="Total loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def save(self, log_path: str, plot_losses: bool = True, plot_pssm_video: bool = True):
        import pickle
        if self.clean_history is None:
            self.clean_trajectory()

        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.clean_history["model"]
        run_dir = os.path.join(log_path, f"{model_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Save clean_history
        with open(os.path.join(run_dir, "history.pkl"), "wb") as f:
            pickle.dump(self.clean_history, f)

        # Save plot of losses
        if plot_losses:
            fig = self.plot_losses()
            fig.savefig(os.path.join(run_dir, "losses.png"))
            plt.close(fig)

        # Save pssm evolution plot
        if plot_pssm_video:
            pssm_trajectory_video(
                self.clean_history["optim"]["pssm"],
                output_path=os.path.join(run_dir, "pssm_evolution.mp4"),
            )

        print(f"Saved run to: {run_dir}")
        return run_dir