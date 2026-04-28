"""Sapiens pseudo-log-likelihood loss.

This loss computes the pseudo-log-likelihood of an antibody sequence using
the Sapiens RoBERTa-based language model.  It supports continuous
(PSSM-style) token inputs via the embedding-weight matrix-multiplication
trick used in the ablang losses.
"""

import numpy as np
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import LossTerm, TOKENS
import jsapiens


# ------------------------------------------------------------------
# Token-space mapping
# ------------------------------------------------------------------

def boltz_to_sapiens_matrix() -> np.ndarray:
    """
    Build a (20, 25) matrix that maps the standard 20 amino-acid
    one-hot ordering (TOKENS) to the Sapiens vocabulary indices.

    Sapiens vocab (25 tokens):
        0 = <s>      1 = <pad>    2 = </s>     3 = <unk>
        4 = A        5 = C        6 = D        7 = E
        8 = F        9 = G       10 = H       11 = I
       12 = K       13 = L       14 = M       15 = N
       16 = P       17 = Q       18 = R       19 = S
       20 = T       21 = V       22 = W       23 = Y
       24 = <mask>
    """
    sapiens_aa = {
        "A": 4, "C": 5, "D": 6, "E": 7, "F": 8, "G": 9, "H": 10, "I": 11,
        "K": 12, "L": 13, "M": 14, "N": 15, "P": 16, "Q": 17, "R": 18,
        "S": 19, "T": 20, "V": 21, "W": 22, "Y": 23,
    }
    T = np.zeros((len(TOKENS), 25))
    for i, tok in enumerate(TOKENS):
        T[i, sapiens_aa[tok]] = 1
    return T


# Cache the matrix as a JAX array so we don't rebuild it on every call.
_BOLTZ_TO_SAPIENS = jnp.array(boltz_to_sapiens_matrix())


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_sapiens(checkpoint_path: str = "prihodad/biophi-sapiens1-vh"):
    """
    Load a pretrained Sapiens PyTorch checkpoint and convert it to JAX.

    Weights are auto-downloaded from HuggingFace on first call (~5 MB).
    """
    import transformers
    pt_model = transformers.RobertaForMaskedLM.from_pretrained(checkpoint_path)
    pt_model.eval()
    return jsapiens.from_torch(pt_model)


# ------------------------------------------------------------------
# Loss term
# ------------------------------------------------------------------

class SapiensPseudoLikelihood(LossTerm):
    """
    Pseudo-log-likelihood loss for Sapiens (RoBERTa-based antibody LM).

    Computes the average log-likelihood of each residue in the sequence
    by masking it one-at-a-time and reading off the model's predicted
    probability at that position.  This is done in parallel via ``vmap``.

    Supports continuous (PSSM-style) token inputs via the same
    embedding-weight lookup trick used in the ablang losses.
    """

    model: jsapiens.RobertaForMaskedLMEquinox
    stop_grad: bool = True

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]

        # Convert from standard 20-token space to Sapiens 25-token space
        sapiens_toks_unpadded = seq_standard_tokens @ _BOLTZ_TO_SAPIENS

        # Build the full token sequence with Sapiens special tokens.
        #   0 = <s>
        #   2 = </s>
        #  24 = <mask>
        toks = jnp.concatenate([
            jax.nn.one_hot([0], 25),    # <s>
            sapiens_toks_unpadded,      # residue tokens
            jax.nn.one_hot([2], 25),    # </s>
        ])

        # Extract sub-modules for manual forward pass (needed for the
        # continuous-input embedding trick and per-position masking).
        embeddings = self.model.roberta.embeddings
        encoder = self.model.roberta.encoder
        head = self.model.lm_head

        # RoBERTa position IDs start at padding_idx + 1 = 2
        position_ids = jnp.arange(len(toks)) + embeddings.padding_idx + 1
        max_pos = embeddings.position_embeddings.weight.shape[0]

        def single_ll(index: int):
            # Replace token at `index` with <mask>
            masked_toks = toks.at[index].set(jax.nn.one_hot(24, 25))

            # Manual embedding lookup -- works for both discrete indices
            # and continuous (PSSM-style) token distributions.
            word_emb = masked_toks @ embeddings.word_embeddings.weight

            # Position embeddings (RoBERTa-style)
            position_emb = (
                jax.nn.one_hot(position_ids, max_pos)
                @ embeddings.position_embeddings.weight
            )

            # Token type embeddings (all zeros, token type 0)
            token_type_emb = jnp.broadcast_to(
                embeddings.token_type_embeddings.weight[0],
                word_emb.shape,
            )

            x = embeddings.layer_norm(
                word_emb + position_emb + token_type_emb
            )

            # Add batch dimension
            x = x[None]  # [1, seq_len, hidden]

            # Run through encoder
            x = encoder(x)

            # Decode head
            logits = head(x)[0]  # [seq_len, vocab_size]

            # Log-probabilities at the masked position
            return jax.nn.log_softmax(logits[index])

        # Evaluate only the residue positions (excluding <s> and </s>)
        eval_indices = jnp.arange(start=1, stop=n + 1)
        masked_log_likelihoods = jax.vmap(single_ll)(eval_indices)

        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(
                masked_log_likelihoods
            )

        pll = (masked_log_likelihoods * sapiens_toks_unpadded).sum(-1).mean()
        return -pll, {"sapiens_pll": pll}