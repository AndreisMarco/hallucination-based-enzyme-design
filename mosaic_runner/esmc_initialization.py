"""ESMC-based PSSM initialization for Mosaic optimization v2.

Single-pass masked language model prior: mask variable positions, run one
ESMC forward pass, convert to 20-AA logits, add scaled Gumbel noise.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from mosaic.losses.esmc import boltz_to_esmc_matrix
from mosaic_utils import log


def _load_esmc():
    from esmj import from_pretrained
    log("Loading ESMC model...")
    return from_pretrained("esmc_300m")


def _build_tokens(sequence: str, esmc) -> np.ndarray:
    """Build (L+2, 64) one-hot token array: CLS + sequence + EOS.

    Fixed residues get their AA token, 'X' positions get <mask>.
    """
    L = len(sequence)
    vocab = esmc.vocab
    token_indices = np.array([
        vocab["<mask>"] if aa == "X" else vocab[aa]
        for aa in sequence
    ])
    tokens = np.zeros((L + 2, 64), dtype=np.float32)
    tokens[0, vocab["<cls>"]] = 1.0
    tokens[1:L + 1] = np.eye(64, dtype=np.float32)[token_indices]
    tokens[L + 1, vocab["<eos>"]] = 1.0
    return tokens


def initialize_esmc_pssm(sequence: str, key: jax.Array) -> Array:
    """Generate an ESMC-based PSSM prior for variable positions.

    Args:
        sequence: design sequence where 'X' marks variable positions.
        key: JAX PRNG key for noise sampling.

    Returns:
        (n_variable, 20) logits array in logspace.
    """
    esmc = _load_esmc()
    conversion_matrix = np.array(boltz_to_esmc_matrix(esmc))  # (20, 64)
    tokens = _build_tokens(sequence, esmc)
    L = len(sequence)

    x = jnp.array(tokens) @ esmc.embed.embedding.weight
    x, _ = esmc.transformer(x[None])
    logits_64 = np.array(esmc.sequence_head(x)[0])  # (L+2, 64)
    del esmc

    logits_64 = logits_64[1:L + 1]  # strip CLS/EOS

    variable_mask = np.array([aa == "X" for aa in sequence])
    logits_64 = logits_64[variable_mask]  # (n_variable, 64)
    logits_20 = logits_64 @ conversion_matrix.T  # (n_variable, 20)

    scale_key, gumbel_key = jax.random.split(key)
    scale = jax.random.uniform(scale_key, minval=0.25, maxval=0.75)
    noise = jax.random.gumbel(key=gumbel_key, shape=logits_20.shape)

    return jnp.array(logits_20) + scale * noise
