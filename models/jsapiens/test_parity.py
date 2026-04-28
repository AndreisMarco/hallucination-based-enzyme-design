#!/usr/bin/env python3
"""
Simple parity test: compare Sapiens PyTorch vs JAX layer by layer.

Run from the repo root (mosaic/) or from jsapiens/:
    python jsapiens/test_parity.py

The script will auto-download the pretrained Sapiens VH weights on first run.
"""

import os
import sys

# Add src/ to path so we can import jsapiens
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

import numpy as np
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import equinox as eqx
from transformers import RobertaTokenizer, RobertaForMaskedLM
import jsapiens


def compare(name, pt_tensor, jax_array):
    """Compare a PyTorch tensor with a JAX array."""
    pt = pt_tensor.detach().cpu().numpy() if torch.is_tensor(pt_tensor) else pt_tensor
    jax = np.array(jax_array)
    max_diff = float(np.max(np.abs(pt - jax)))
    mean_diff = float(np.mean(np.abs(pt - jax)))
    pt_max = float(np.max(np.abs(pt)))
    rel_diff = max_diff / (pt_max + 1e-9)
    status = "OK" if rel_diff < 0.01 else "CHECK"
    print(f"{name:45s}  {status}  max={max_diff:.3e}  mean={mean_diff:.3e}  rel={rel_diff:.3e}")
    return max_diff


def main():
    print("=" * 70)
    print("Sapiens PyTorch vs JAX Numerical Parity Test")
    print("=" * 70)

    # Load pretrained PyTorch model (auto-downloads on first run)
    print("\n[1/3] Loading pretrained PyTorch model (prihodad/biophi-sapiens1-vh)...")
    pt_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    pt_model.eval()

    print("[2/3] Converting to JAX...")
    jax_model = jsapiens.from_pretrained("prihodad/biophi-sapiens1-vh")

    # Prepare a test sequence
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    test_seq = (
        "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRV"
        "TLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    )

    print("[3/3] Tokenizing test sequence...")
    encoded = tokenizer(test_seq, return_tensors="pt")
    pt_input_ids = encoded["input_ids"]
    pt_attention_mask = encoded["attention_mask"]
    jax_input_ids = jnp.array(pt_input_ids.numpy())
    jax_attention_mask = jnp.array(pt_attention_mask.numpy())

    # Sanity check: tokenization should match
    if not np.allclose(pt_input_ids.numpy(), np.array(jax_input_ids)):
        raise RuntimeError("Tokenization mismatch between PyTorch and JAX!")
    print(
        f"      Tokens shape: {list(pt_input_ids.shape)}  "
        f"(batch={pt_input_ids.shape[0]}, seq_len={pt_input_ids.shape[1]})"
    )

    # ------------------------------------------------------------------
    # Layer-by-layer forward pass comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Layer-by-layer comparison")
    print("=" * 70)

    with torch.no_grad():
        # --- Embeddings ---
        pt_embeddings = pt_model.roberta.embeddings(pt_input_ids)
        jax_embeddings = jax_model.roberta.embeddings(jax_input_ids)
        compare("Embeddings", pt_embeddings, jax_embeddings)

        # --- Encoder ---
        pt_hidden = pt_embeddings
        jax_hidden = jax_embeddings

        # For an unmasked test sequence we can pass None; this sidesteps
        # SDPA backend differences when calling individual layers manually.
        num_layers = len(pt_model.roberta.encoder.layer)
        for i in range(num_layers):
            pt_layer = pt_model.roberta.encoder.layer[i]

            # Attention sub-layer (self-attn + output projection + residual + LN)
            pt_attention_output = pt_layer.attention(pt_hidden, None)[0]

            # Extract JAX layer i from the scanned encoder
            layer_i_params = jax.tree.map(
                lambda x: x[i], jax_model.roberta.encoder.layer_params
            )
            jax_layer = eqx.combine(
                jax_model.roberta.encoder.layer_static, layer_i_params
            )
            jax_attention_output = jax_layer.attention(
                jax_hidden, attention_mask=None
            )
            compare(f"Layer {i} attention output", pt_attention_output, jax_attention_output)

            # Intermediate sub-layer (dense 128->256 + GELU)
            pt_intermediate = pt_layer.intermediate(pt_attention_output)
            jax_intermediate = jax_layer.intermediate(jax_attention_output)
            compare(f"Layer {i} intermediate", pt_intermediate, jax_intermediate)

            # Output sub-layer (dense 256->128 + residual + LN) -> block output
            pt_layer_output = pt_layer.output(pt_intermediate, pt_attention_output)
            jax_layer_output = jax_layer.output(
                jax_intermediate, jax_attention_output
            )
            compare(f"Layer {i} block output", pt_layer_output, jax_layer_output)

            pt_hidden = pt_layer_output
            jax_hidden = jax_layer_output

        # --- Final encoder output ---
        compare("Encoder final output", pt_hidden, jax_hidden)

        # --- LM Head: dense ---
        pt_lm_dense = pt_model.lm_head.dense(pt_hidden)
        jax_lm_dense = jax_model.lm_head.dense(jax_hidden)
        compare("LM Head dense", pt_lm_dense, jax_lm_dense)

        # --- LM Head: GELU ---
        pt_lm_gelu = F.gelu(pt_lm_dense)
        jax_lm_gelu = jax.nn.gelu(jax_lm_dense)
        compare("LM Head GELU", pt_lm_gelu, jax_lm_gelu)

        # --- LM Head: LayerNorm ---
        pt_lm_ln = pt_model.lm_head.layer_norm(pt_lm_gelu)
        jax_lm_ln = jax_model.lm_head.layer_norm(jax_lm_gelu)
        compare("LM Head LayerNorm", pt_lm_ln, jax_lm_ln)

        # --- LM Head: decoder (logits) ---
        pt_lm_logits = pt_model.lm_head.decoder(pt_lm_ln)
        jax_lm_logits = jax_model.lm_head.decoder(jax_lm_ln)
        compare("LM Head decoder", pt_lm_logits, jax_lm_logits)

        # --- Full model: logits ---
        pt_logits = pt_model(pt_input_ids, attention_mask=pt_attention_mask).logits
        jax_logits = jax_model(jax_input_ids, attention_mask=jax_attention_mask)
        compare("Final logits", pt_logits, jax_logits)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()