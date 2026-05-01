#!/usr/bin/env python3
"""
Simple parity test: compare ablang2 PyTorch vs JAX layer by layer.

Run from the repo root (mosaic/) or from jablang2/:
    python jablang2/test_parity.py

The script will auto-download the pretrained ablang2-paired weights on first run.
"""

import os
import sys

# Add src/ to path so we can import jablang
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

import numpy as np
import torch
import jax.numpy as jnp
from ablang2.load_model import load_model
import jablang


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
    print("AbLang2 PyTorch vs JAX Numerical Parity Test")
    print("=" * 70)

    # Load pretrained PyTorch model (auto-downloads on first run)
    print("\n[1/3] Loading pretrained PyTorch model (ablang2-paired)...")
    print("      (This will download ~500MB weights on first run)")
    pt_model, pt_tokenizer, hparams = load_model(
        "ablang2-paired", random_init=False, device="cpu"
    )
    pt_model.eval()

    print("[2/3] Converting to JAX...")
    jax_model = jablang.from_torch(pt_model)

    # Prepare a test sequence: heavy + light chain
    test_seq = [
        (
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSL",
            "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK",
        )
    ]

    print("[3/3] Tokenizing test sequence...")
    pt_tokens = pt_tokenizer(test_seq, mode="encode", pad=True, w_extra_tkns=True, device="cpu")
    jax_tokens = jnp.array(pt_tokens.numpy(), dtype=jnp.int32)

    # Sanity check: tokenization should match
    if not np.allclose(pt_tokens.numpy(), np.array(jax_tokens)):
        raise RuntimeError("Tokenization mismatch between PyTorch and JAX tokenizers!")
    print(f"      Tokens shape: {list(pt_tokens.shape)}  (batch={pt_tokens.shape[0]}, seq_len={pt_tokens.shape[1]})")

    # ------------------------------------------------------------------
    # Layer-by-layer forward pass comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Layer-by-layer comparison")
    print("=" * 70)

    with torch.no_grad():
        # --- AbRep: Embeddings ---
        pt_hidden = pt_model.AbRep.aa_embed_layer(pt_tokens)
        jax_hidden = jax_model.rep.aa_embed_layer(jax_tokens)
        compare("AbRep Embedding", pt_hidden, jax_hidden)

        # --- Encoder blocks ---
        padding_mask_pt = pt_tokens.eq(hparams.pad_tkn)
        padding_mask_jax = jax_tokens == hparams.pad_tkn

        for i, (pt_enc, jax_enc) in enumerate(zip(pt_model.AbRep.encoder_blocks, jax_model.rep.encoder_blocks)):
            # Pre-attention LayerNorm
            pt_pre_ln = pt_enc.pre_attn_layer_norm(pt_hidden)
            jax_pre_ln = jax_enc.pre_attn_layer_norm(jax_hidden)
            compare(f"Enc{i} pre-attn LN", pt_pre_ln, jax_pre_ln)

            # Multi-head attention output + attention weights
            pt_mha, pt_attn_w = pt_enc.multihead_attention(
                pt_pre_ln, attn_mask=padding_mask_pt, return_attn_weights=True
            )
            jax_mha, jax_attn_w = jax_enc.multihead_attention(
                jax_pre_ln, attn_mask=padding_mask_jax, return_attn_weights=True
            )
            compare(f"Enc{i} MHA output", pt_mha, jax_mha)
            compare(f"Enc{i} attn weights", pt_attn_w, jax_attn_w)

            # Post-attention residual + LayerNorm
            pt_post_attn = pt_pre_ln + pt_mha
            jax_post_attn = jax_pre_ln + jax_mha
            pt_post_ln = pt_enc.final_layer_norm(pt_post_attn)
            jax_post_ln = jax_enc.final_layer_norm(jax_post_attn)
            compare(f"Enc{i} post-attn LN", pt_post_ln, jax_post_ln)

            # Intermediate (FFN)
            pt_int = pt_enc.intermediate_layer(pt_post_ln)
            jax_int = jax_enc.intermediate_layer(jax_post_ln)
            compare(f"Enc{i} intermediate", pt_int, jax_int)

            # Final residual for this block
            pt_hidden = pt_post_ln + pt_int
            jax_hidden = jax_post_ln + jax_int
            compare(f"Enc{i} block output", pt_hidden, jax_hidden)

        # --- AbRep: Final LayerNorm ---
        pt_final = pt_model.AbRep.layer_norm_after_encoder_blocks(pt_hidden)
        jax_final = jax_model.rep.layer_norm_after_encoder_blocks(jax_hidden)
        compare("AbRep final LN", pt_final, jax_final)

        # --- AbHead: FF (before weight-tied projection) ---
        pt_ff = pt_model.AbHead.ff(pt_final)
        jax_ff = jax_model.head.ff(jax_final)
        compare("AbHead FF output", pt_ff, jax_ff)

        # --- Full model: logits ---
        pt_logits = pt_model(pt_tokens)
        jax_logits = jax_model(jax_tokens)
        compare("Final logits", pt_logits, jax_logits)

    # ------------------------------------------------------------------
    # Alternative return modes
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Alternative return modes")
    print("=" * 70)

    with torch.no_grad():
        # Attention weights from full model
        pt_attn_all = pt_model(pt_tokens, return_attn_weights=True)
        jax_attn_all = jax_model(jax_tokens, return_attn_weights=True)
        for i, (pt_a, jax_a) in enumerate(zip(pt_attn_all, jax_attn_all)):
            compare(f"  Full-model attn layer {i}", pt_a, jax_a)

        # Intermediate representations
        return_layers = [0, 1, hparams.n_encoder_blocks]
        pt_reps = pt_model(pt_tokens, return_rep_layers=return_layers)
        jax_reps = jax_model(jax_tokens, return_rep_layers=return_layers)
        for k in sorted(pt_reps.keys()):
            compare(f"  Rep layer {k}", pt_reps[k], jax_reps[k])

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
