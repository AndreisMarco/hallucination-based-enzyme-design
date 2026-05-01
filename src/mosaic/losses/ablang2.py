
# Common imports
import numpy as np
import jax
from jax import numpy as jnp

from mosaic.common import LossTerm, TOKENS
from ablang2.load_model import load_model

# Luis
from jaxtyping import Array, Float
from jablang2 import AbLang, from_torch

# source-mosaic
from typing import Any
from jablang import from_torch


########################
# from luis code
########################

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def boltz_to_ablang2_matrix() -> np.ndarray:
    """
    Build a (20, 26) matrix that maps the standard 20 amino-acid
    one-hot ordering (TOKENS) to the ablang2 vocabulary indices.
    """
    ablang2_aa = {
        "A": 14, "C": 11, "D": 5, "E": 6, "F": 17, "G": 12,
        "H": 3, "I": 16, "K": 4, "L": 20, "M": 1, "N": 9,
        "P": 13, "Q": 10, "R": 2, "S": 7, "T": 8, "V": 15,
        "W": 19, "Y": 18,
    }
    T = np.zeros((len(TOKENS), 26))
    for i, tok in enumerate(TOKENS):
        T[i, ablang2_aa[tok]] = 1
    return T


# Cache the matrix as a JAX array so we don't rebuild it on every call.
_BOLTZ_TO_ABLANG2 = jnp.array(boltz_to_ablang2_matrix())


def load_ablang2(model_name: str = "ablang2-paired"):
    """
    Load a pretrained ablang2 PyTorch model, convert it to JAX,
    and return the model plus hyperparameters.

    Weights are auto-downloaded on first call (~500 MB).
    """
    pt_model, pt_tokenizer, hparams = load_model(
        model_name, random_init=False, device="cpu"
    )
    pt_model.eval()
    return from_torch(pt_model), hparams


# ------------------------------------------------------------------
# Loss term
# ------------------------------------------------------------------

class AbLang2PseudoLikelihood(LossTerm):
    """
    Pseudo-log-likelihood loss for ablang2.

    Computes the average log-likelihood of each residue in the sequence
    by masking it one-at-a-time and reading off the model's predicted
    probability at that position.  This is done in parallel via vmap.

    Supports continuous (PSSM-style) token inputs via the same
    embedding-weight lookup trick used in the ablang1 loss.
    """

    model: AbLang
    stop_grad: bool = True
    chain: str = "heavy"  # "heavy" or "light"

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]

        # Convert from standard 20-token space to ablang2 26-token space
        ablang2_toks_unpadded = seq_standard_tokens @ _BOLTZ_TO_ABLANG2

        # Build the full token sequence with ablang2 special tokens.
        # Heavy-only (nanobody) format: <heavy>|
        #   0 = <  (start heavy)
        #  22 = >  (end heavy)
        #  25 = |  (chain separator)
        #  23 = *  (mask token)
        if self.chain == "heavy":
            toks = jnp.concatenate([
                jax.nn.one_hot([0], 26),    # <
                ablang2_toks_unpadded,      # heavy chain residues
                jax.nn.one_hot([22], 26),   # >
                jax.nn.one_hot([25], 26),   # |
            ])
            eval_indices = jnp.arange(start=1, stop=n + 1)
        elif self.chain == "light":
            toks = jnp.concatenate([
                jax.nn.one_hot([25], 26),   # |
                jax.nn.one_hot([0], 26),    # <
                ablang2_toks_unpadded,      # light chain residues
                jax.nn.one_hot([22], 26),   # >
            ])
            eval_indices = jnp.arange(start=2, stop=n + 2)
        else:
            raise ValueError(
                f"AbLang2PseudoLikelihood: chain must be 'heavy' or 'light', "
                f"got {self.chain!r}"
            )

        rep = self.model.rep
        head = self.model.head

        def single_ll(index: int):
            # Replace token at `index` with *
            masked_tokens = toks.at[index].set(
                jax.nn.one_hot(23, 26)
            )

            # Manual embedding lookup -- works for both discrete indices
            # and continuous (PSSM-style) token distributions.
            x = masked_tokens @ rep.aa_embed_layer.weight

            # Add batch dimension
            x = x[None]  # [1, seq_len, hidden]

            # Run through all encoder blocks
            for encoder_block in rep.encoder_blocks:
                x, _ = encoder_block(
                    x, attn_mask=None, return_attn_weights=False
                )

            # Final layer norm
            x = rep.layer_norm_after_encoder_blocks(x)

            # Decode head
            logits = head(x)[0]  # [seq_len, vocab_size]

            # Log-probabilities at the masked position
            return jax.nn.log_softmax(logits[index])

        # Evaluate only the residue positions (excluding special tokens)
        masked_log_likelihoods = jax.vmap(single_ll)(eval_indices)

        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(
                masked_log_likelihoods
            )

        pll = (masked_log_likelihoods * ablang2_toks_unpadded).sum(-1).mean()
        return -pll, {"ablang2_pll": pll}



########################
# from source-mosaic
########################
def boltz_to_ablang2_matrix_source(tokenizer):
    T = np.zeros((len(TOKENS), len(tokenizer.aa_to_token)))
    for i, tok in enumerate(TOKENS):
        idx = tokenizer.aa_to_token[tok]
        T[i, idx] = 1
    return T


def load_ablang2_source():
    model_pt, tokenizer, _hparams = load_model("ablang2-paired")
    model_pt.eval()
    return from_torch(model_pt), tokenizer


class Ablang2PseudoLikelihood_source(LossTerm):
    """Pseudo-likelihood loss using the AbLang2 paired model.

    Formats the concatenated binder sequence as ``<H>|<L>`` (or ``<H>|`` /
    ``|<L>`` for single-chain) to match ablang2's input convention, and masks
    special-token logits before log-softmax.

    ``heavy_len`` specifies how many leading residues belong to the heavy chain;
    the remainder are the light chain.  Use ``heavy_len=len(seq)`` for heavy-only
    and ``heavy_len=0`` for light-only.
    """

    model: Any
    tokenizer: Any
    heavy_len: int
    designable_positions: jax.Array | None
    token_mapping: jax.Array
    special_mask: jax.Array
    mask_onehot: jax.Array
    vocab_size: int
    stop_grad: bool = True
    aux_name: str = "ablang2_ppl"

    def __init__(
        self,
        model,
        tokenizer,
        heavy_len,
        designable_positions=None,
        stop_grad=True,
        aux_name="ablang2_ppl",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.heavy_len = heavy_len
        self.designable_positions = designable_positions
        self.stop_grad = stop_grad
        self.aux_name = aux_name
        self.token_mapping = jnp.array(boltz_to_ablang2_matrix(tokenizer))
        self.vocab_size = len(tokenizer.aa_to_token)
        special_indices = jnp.array(tokenizer.all_special_tokens, dtype=jnp.int32)
        self.special_mask = (
            jnp.zeros(self.vocab_size, dtype=bool).at[special_indices].set(True)
        )
        self.mask_onehot = jax.nn.one_hot(tokenizer.aa_to_token["*"], self.vocab_size)

    def __call__(self, seq_standard_tokens, *, key):
        del key
        n = seq_standard_tokens.shape[0]
        designable_positions = (
            self.designable_positions
            if self.designable_positions is not None
            else jnp.arange(n, dtype=jnp.int32)
        )

        ablang2_toks = seq_standard_tokens @ self.token_mapping
        at = self.tokenizer.aa_to_token

        def special(token):
            return jax.nn.one_hot(jnp.array([at[token]]), self.vocab_size)

        parts: list[jax.Array] = []
        sequence_token_indices = jnp.full(n, -1, dtype=jnp.int32)
        offset = 0

        if self.heavy_len > 0:
            parts += [special("<"), ablang2_toks[: self.heavy_len], special(">")]
            sequence_token_indices = sequence_token_indices.at[: self.heavy_len].set(
                jnp.arange(offset + 1, offset + 1 + self.heavy_len, dtype=jnp.int32)
            )
            offset += self.heavy_len + 2

        parts.append(special("|"))
        offset += 1

        if self.heavy_len < n:
            parts += [special("<"), ablang2_toks[self.heavy_len :], special(">")]
            sequence_token_indices = sequence_token_indices.at[self.heavy_len :].set(
                jnp.arange(offset + 1, offset + 1 + n - self.heavy_len, dtype=jnp.int32)
            )

        toks = jnp.concatenate(parts)
        residue_indices = sequence_token_indices[designable_positions]
        designable_toks = ablang2_toks[designable_positions]
        num_designable = designable_positions.shape[0]

        def single_ll(token_index):
            masked_tokens = toks.at[token_index].set(self.mask_onehot)
            x = masked_tokens @ self.model.rep.aa_embed_layer.weight
            x = self.model.rep.encoder_blocks(x[None])
            x = self.model.rep.layer_norm(x)
            logits = self.model.head(x)[0]
            logits = jnp.where(self.special_mask, -1e9, logits[token_index])
            return jax.nn.log_softmax(logits)

        masked_log_likelihoods = jax.vmap(single_ll)(residue_indices)
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        per_position_pll = (masked_log_likelihoods * designable_toks).sum(-1)
        pll = jnp.sum(per_position_pll) / jnp.maximum(
            jnp.array(num_designable, dtype=per_position_pll.dtype), 1.0
        )
        return -pll, {self.aux_name: jnp.exp(-pll)}
