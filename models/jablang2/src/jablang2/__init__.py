from dataclasses import fields
from functools import singledispatch
from typing import NamedTuple, Optional

import einops
import equinox as eqx
import jax
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from functools import partial
import numpy as np

import ablang2.models.ablang2.ablang as ablang
import ablang2.models.ablang2.encoderblock as encoderblock

@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


# basic types
from_torch.register(torch.Tensor, lambda x: np.array(x.detach()))
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.GELU, lambda _: jax.nn.gelu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])


class AbstractFromTorch(eqx.Module):
    """
    Default implementation of `from_torch` for equinox modules.
    This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
    Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.

    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module):
        # assemble arguments to `cls` constructor from `model`

        field_to_type = {field.name: field.type for field in fields(cls)}
        kwargs = {
            child: from_torch(child_module)
            for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        }

        # add fields that are not child_modules or parameters
        for field_name, field_type in field_to_type.items():
            if not hasattr(model, field_name):
                if not isinstance(None, field_type):
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                    )
                else:
                    kwargs[field_name] = None
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        # check we're not passing any additional properties
        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
            )

        return cls(**kwargs)


def register_from_torch(torch_module_type):
    """Class decorator to register an equinox module for conversion from a torch module."""

    def decorator(cls):
        from_torch.register(torch_module_type, cls.from_torch)
        return cls

    return decorator


# this isn't very jax-y
def _vmap(f, tensor, *args):
    for _ in range(len(tensor.shape) - 1):
        f = jax.vmap(f)
    return f(tensor, *args)


def vmap_to_last_dimension(f):
    return partial(_vmap, f)


@register_from_torch(torch.nn.Linear)
class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics"""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(l: torch.nn.Linear):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))


@register_from_torch(torch.nn.LayerNorm)
class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics"""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        ln = eqx.nn.LayerNorm(
            shape=x.shape[-1],
            eps=self.eps,
            use_weight=self.weight is not None,
            use_bias=self.bias is not None,
        )
        ln = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            ln,
            (self.weight, self.bias),
            is_leaf=lambda x: x is None,
        )

        return vmap_to_last_dimension(ln)(x)

    @staticmethod
    def from_torch(l: torch.nn.LayerNorm):
        return LayerNorm(
            weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps
        )


@register_from_torch(torch.nn.Sequential)
class Sequential(eqx.Module):
    _modules: dict[
        str, AbstractFromTorch
    ]  # IMHO this is a fairly wild design choice, but this is really how pytorch works.

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))


@register_from_torch(torch.nn.modules.sparse.Embedding)
class SparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __call__(self, indices):
        ndims = len(indices.shape)

        def apply(index):
            return self.embedding(index)

        f = apply
        for _ in range(ndims):
            f = jax.vmap(f)

        return f(indices)

    @staticmethod
    def from_torch(m: torch.nn.modules.sparse.Embedding):
        return SparseEmbedding(embedding=eqx.nn.Embedding(weight=from_torch(m.weight)))


@register_from_torch(torch.nn.Embedding)
class Embedding(eqx.Module):
    weight: Float[Array, "Vocab Embedding"]
    padding_idx: int

    def __call__(self, indices):
        return jax.numpy.take(
            self.weight,
            indices,
            axis=0,
        )

    @staticmethod
    def from_torch(m: torch.nn.Embedding):
        return Embedding(
            weight=from_torch(m.weight),
            padding_idx=m.padding_idx,
        )


# --- Rotary Embeddings ---


def rotate_half(x: Float[Array, "... seq_len head_dim"]) -> Float[Array, "... seq_len head_dim"]:
    *leading, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    x = x.reshape(*leading, seq_len, head_dim // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.stack([-x2, x1], axis=-1).reshape(*leading, seq_len, head_dim)


def apply_rotary_emb(
    x: Float[Array, "... seq_len head_dim"], theta: float = 10000.0
) -> Float[Array, "... seq_len head_dim"]:
    *leading, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    freqs = theta ** (-jnp.arange(0, head_dim, 2) / head_dim)
    pos = jnp.arange(seq_len)
    angles = jnp.outer(pos, freqs)
    angles = jnp.repeat(angles, 2, axis=-1)

    x_rotated = rotate_half(x)
    return x * jnp.cos(angles) + x_rotated * jnp.sin(angles)


# --- Model Layers ---


@register_from_torch(encoderblock.SwiGLU)
class SwiGLU(eqx.Module):
    def __call__(self, x: Float[Array, "... 2D"]) -> Float[Array, "... D"]:
        x, gate = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * x

    @staticmethod
    def from_torch(m: encoderblock.SwiGLU):
        return SwiGLU()


@register_from_torch(encoderblock.MultiHeadAttention)
class MultiHeadAttention(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scaling: float = eqx.field(static=True)
    k_proj: Linear
    v_proj: Linear
    q_proj: Linear
    out_proj: Linear

    def __call__(
        self,
        x: Float[Array, "B S E"],
        attn_mask: Optional[Int[Array, "B S"]] = None,
        return_attn_weights: bool = False,
    ):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q = apply_rotary_emb(q)
        k = apply_rotary_emb(k)

        attn_weights = einops.einsum(q, k, "b h s1 d, b h s2 d -> b h s1 s2")
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_mask = einops.rearrange(attn_mask, "b s -> b 1 1 s")
            attn_weights = jnp.where(attn_mask, -jnp.inf, attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn = einops.einsum(attn_weights, v, "b h s1 s2, b h s2 d -> b h s1 d")

        attn = einops.rearrange(attn, "b h s d -> b s (h d)")
        attn = self.out_proj(attn)

        if return_attn_weights:
            return attn, attn_weights
        else:
            return attn, None

    @staticmethod
    def from_torch(m: encoderblock.MultiHeadAttention):
        return MultiHeadAttention(
            embed_dim=m.embed_dim,
            num_heads=m.num_heads,
            head_dim=m.head_dim,
            scaling=m.scaling,
            k_proj=from_torch(m.k_proj),
            v_proj=from_torch(m.v_proj),
            q_proj=from_torch(m.q_proj),
            out_proj=from_torch(m.out_proj),
        )


@register_from_torch(ablang.TransformerEncoder)
class TransformerEncoder(eqx.Module):
    multihead_attention: MultiHeadAttention
    intermediate_layer: Sequential
    pre_attn_layer_norm: LayerNorm
    final_layer_norm: LayerNorm

    def __call__(
        self,
        hidden_embed: Float[Array, "B S H"],
        attn_mask=None,
        return_attn_weights: bool = False,
    ):
        residual = hidden_embed
        hidden_embed = self.pre_attn_layer_norm(hidden_embed)
        hidden_embed, attn_weights = self.multihead_attention(
            hidden_embed,
            attn_mask=attn_mask,
            return_attn_weights=return_attn_weights,
        )
        hidden_embed = residual + hidden_embed

        residual = hidden_embed
        hidden_embed = self.final_layer_norm(hidden_embed)
        hidden_embed = self.intermediate_layer(hidden_embed)
        hidden_embed = residual + hidden_embed
        return hidden_embed, attn_weights

    @staticmethod
    def from_torch(m: ablang.TransformerEncoder):
        return TransformerEncoder(
            multihead_attention=from_torch(m.multihead_attention),
            intermediate_layer=from_torch(m.intermediate_layer),
            pre_attn_layer_norm=from_torch(m.pre_attn_layer_norm),
            final_layer_norm=from_torch(m.final_layer_norm),
        )


class DataAbRep(NamedTuple):
    last_hidden_states: Float[Array, "B S H"]
    many_hidden_states: Optional[dict[int, Float[Array, "B S H"]]] = None
    attention_weights: Optional[list[Float[Array, "B H S S"]]] = None


@register_from_torch(ablang.AbRep)
class AbRep(eqx.Module):
    padding_tkn: int = eqx.field(static=True)
    mask_tkn: int = eqx.field(static=True)
    aa_embed_layer: Embedding
    encoder_blocks: list[TransformerEncoder]
    layer_norm_after_encoder_blocks: LayerNorm

    def __call__(
        self,
        tokens: Int[Array, "B N"],
        return_attn_weights: bool = False,
        return_rep_layers=[],
    ):
        assert tokens.ndim == 2
        padding_mask = tokens == self.padding_tkn

        hidden_embed = self.aa_embed_layer(tokens)

        return_rep_layers = set(return_rep_layers)
        rep_layers = {}
        if 0 in return_rep_layers:
            rep_layers[0] = hidden_embed

        all_attn_weights = []

        for n_layer, encoder_block in enumerate(self.encoder_blocks):
            hidden_embed, attn_weights = encoder_block(
                hidden_embed, padding_mask, return_attn_weights
            )
            if (n_layer + 1) in return_rep_layers:
                rep_layers[n_layer + 1] = hidden_embed
            if return_attn_weights:
                all_attn_weights.append(attn_weights)

        hidden_embed = self.layer_norm_after_encoder_blocks(hidden_embed)

        return DataAbRep(
            last_hidden_states=hidden_embed,
            many_hidden_states=rep_layers,
            attention_weights=all_attn_weights,
        )

    @staticmethod
    def from_torch(m: ablang.AbRep):
        return AbRep(
            padding_tkn=m.padding_tkn,
            mask_tkn=m.mask_tkn,
            aa_embed_layer=from_torch(m.aa_embed_layer),
            encoder_blocks=from_torch(m.encoder_blocks),
            layer_norm_after_encoder_blocks=from_torch(m.layer_norm_after_encoder_blocks),
        )


@register_from_torch(ablang.AbHead)
class AbHead(eqx.Module):
    ff: Sequential
    weights: Float[Array, "Vocab Hidden"]
    bias: Float[Array, "Vocab"]

    def __call__(self, hidden_embed: Float[Array, "... Hidden"]) -> Float[Array, "... Vocab"]:
        hidden_embed = self.ff(hidden_embed)
        logits = einops.einsum(
            hidden_embed, self.weights, "... Hidden, Vocab Hidden -> ... Vocab"
        )
        logits = logits + self.bias
        return logits

    @staticmethod
    def from_torch(m: ablang.AbHead):
        return AbHead(
            ff=from_torch(m.ff),
            weights=from_torch(m.weights),
            bias=from_torch(m.bias),
        )


@register_from_torch(ablang.AbLang)
class AbLang(eqx.Module):
    rep: AbRep
    head: AbHead

    def __call__(
        self,
        tokens: Int[Array, "B N"],
        return_attn_weights: bool = False,
        return_rep_layers=[],
    ):
        representations = self.rep(tokens, return_attn_weights, return_rep_layers)

        if return_attn_weights:
            return representations.attention_weights
        elif return_rep_layers != []:
            return representations.many_hidden_states
        else:
            likelihoods = self.head(representations.last_hidden_states)
            return likelihoods

    @staticmethod
    def from_torch(m: ablang.AbLang):
        return AbLang(
            rep=from_torch(m.AbRep),
            head=from_torch(m.AbHead),
        )


from .tokenizers import ABtokenizer
