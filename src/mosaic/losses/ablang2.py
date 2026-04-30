import numpy as np
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import LossTerm, TOKENS
from jablang2 import AbLang, from_torch
from ablang2.load_model import load_model


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
