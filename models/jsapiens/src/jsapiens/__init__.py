"""JAX/Equinox translation of Sapiens (RoBERTa-based antibody language model)."""

from dataclasses import fields
from functools import singledispatch, partial
from typing import Optional

import einops
import equinox as eqx
import jax
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np
import transformers
import transformers.models.roberta.modeling_roberta as roberta_module


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
from_torch.register(torch.nn.Sequential, lambda x: Sequential(_modules=from_torch(x._modules)))


class AbstractFromTorch(eqx.Module):
    """
    Default implementation of `from_torch` for equinox modules.
    This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
    Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.
    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module):
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
    _modules: dict[str, AbstractFromTorch]

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))


@register_from_torch(torch.nn.Embedding)
class Embedding(eqx.Module):
    weight: Float[Array, "Vocab Embedding"]
    padding_idx: int | None

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


@register_from_torch(roberta_module.RobertaEmbeddings)
class RobertaEmbeddings(eqx.Module):
    word_embeddings: Embedding
    position_embeddings: Embedding
    token_type_embeddings: Embedding
    layer_norm: LayerNorm
    padding_idx: int

    def __call__(
        self,
        input_ids: Int[Array, "B N"],
        token_type_ids: Optional[Int[Array, "B N"]] = None,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_ids = self._compute_position_ids(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        return self.layer_norm(embeddings)

    def _compute_position_ids(self, input_ids: Int[Array, "B N"]) -> Int[Array, "B N"]:
        mask = input_ids != self.padding_idx
        incremental_indices = jnp.cumsum(mask, axis=-1) * mask
        return incremental_indices + self.padding_idx

    @staticmethod
    def from_torch(m: roberta_module.RobertaEmbeddings):
        assert not m.dropout.training
        return RobertaEmbeddings(
            word_embeddings=from_torch(m.word_embeddings),
            position_embeddings=from_torch(m.position_embeddings),
            token_type_embeddings=from_torch(m.token_type_embeddings),
            layer_norm=from_torch(m.LayerNorm),
            padding_idx=m.padding_idx,
        )


@register_from_torch(roberta_module.RobertaAttention)
class RobertaAttention(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    output_layer_norm: LayerNorm

    def __call__(
        self,
        hidden_states: Float[Array, "B N D"],
        attention_mask: Optional[Int[Array, "B N"]] = None,
    ):
        if attention_mask is not None:
            # Equinox: False = ignore, True = attend
            mask = attention_mask.astype(bool)
            seq_len = hidden_states.shape[1]
            mask = jnp.broadcast_to(
                mask[:, None, :],
                (hidden_states.shape[0], seq_len, seq_len),
            )
        else:
            mask = None

        attn_output = jax.vmap(self.attention)(
            hidden_states, hidden_states, hidden_states, mask=mask
        )
        return self.output_layer_norm(attn_output + hidden_states)

    @staticmethod
    def from_torch(m: roberta_module.RobertaAttention):
        self_attn = m.self
        self_output = m.output

        eqx_attn = eqx.nn.MultiheadAttention(
            num_heads=self_attn.num_attention_heads,
            query_size=self_attn.query.in_features,
            key_size=self_attn.key.in_features,
            value_size=self_attn.value.in_features,
            output_size=self_output.dense.out_features,
            qk_size=self_attn.attention_head_size,
            vo_size=self_attn.attention_head_size,
            use_query_bias=self_attn.query.bias is not None,
            use_key_bias=self_attn.key.bias is not None,
            use_value_bias=self_attn.value.bias is not None,
            use_output_bias=self_output.dense.bias is not None,
            dropout_p=0.0,
            inference=True,
            key=jax.random.key(0),
        )

        eqx_attn = eqx.tree_at(
            lambda attn: (attn.query_proj, attn.key_proj, attn.value_proj, attn.output_proj),
            eqx_attn,
            from_torch((self_attn.query, self_attn.key, self_attn.value, self_output.dense)),
        )

        return RobertaAttention(
            attention=eqx_attn,
            output_layer_norm=from_torch(self_output.LayerNorm),
        )


@register_from_torch(roberta_module.RobertaIntermediate)
class RobertaIntermediate(eqx.Module):
    dense: Linear

    def __call__(self, hidden_states: Float[Array, "... D"]) -> Float[Array, "... 256"]:
        return jax.nn.gelu(self.dense(hidden_states))

    @staticmethod
    def from_torch(m: roberta_module.RobertaIntermediate):
        assert m.intermediate_act_fn.__class__.__name__ in ("GELUActivation",)
        return RobertaIntermediate(dense=from_torch(m.dense))


@register_from_torch(roberta_module.RobertaOutput)
class RobertaOutput(eqx.Module):
    dense: Linear
    layer_norm: LayerNorm

    def __call__(
        self,
        hidden_states: Float[Array, "... D"],
        input_tensor: Float[Array, "... D"],
    ) -> Float[Array, "... D"]:
        hidden_states = self.dense(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)

    @staticmethod
    def from_torch(m: roberta_module.RobertaOutput):
        return RobertaOutput(
            dense=from_torch(m.dense),
            layer_norm=from_torch(m.LayerNorm),
        )


@register_from_torch(roberta_module.RobertaLayer)
class RobertaLayer(eqx.Module):
    attention: RobertaAttention
    intermediate: RobertaIntermediate
    output: RobertaOutput

    def __call__(
        self,
        hidden_states: Float[Array, "B N D"],
        attention_mask: Optional[Int[Array, "B N"]] = None,
    ):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    @staticmethod
    def from_torch(m: roberta_module.RobertaLayer):
        return RobertaLayer(
            attention=from_torch(m.attention),
            intermediate=from_torch(m.intermediate),
            output=from_torch(m.output),
        )


@register_from_torch(roberta_module.RobertaEncoder)
class RobertaEncoder(eqx.Module):
    layer_params: RobertaLayer
    layer_static: RobertaLayer

    def __call__(
        self,
        hidden_states: Float[Array, "B N D"],
        attention_mask: Optional[Int[Array, "B N"]] = None,
    ):
        def body(hidden_states, params):
            layer = eqx.combine(self.layer_static, params)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            return hidden_states, None

        final_state, _ = jax.lax.scan(body, hidden_states, self.layer_params)
        return final_state

    @staticmethod
    def from_torch(m: roberta_module.RobertaEncoder):
        layers = [from_torch(layer) for layer in m.layer]
        layer_params = jax.tree.map(
            lambda *v: jnp.stack(v),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        layer_static = eqx.partition(layers[0], eqx.is_inexact_array)[1]
        return RobertaEncoder(
            layer_params=layer_params,
            layer_static=layer_static,
        )


@register_from_torch(roberta_module.RobertaModel)
class RobertaModel(eqx.Module):
    embeddings: RobertaEmbeddings
    encoder: RobertaEncoder

    def __call__(
        self,
        input_ids: Int[Array, "B N"],
        attention_mask: Optional[Int[Array, "B N"]] = None,
        token_type_ids: Optional[Int[Array, "B N"]] = None,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
        return self.encoder(embedding_output, attention_mask=attention_mask)

    @staticmethod
    def from_torch(m: roberta_module.RobertaModel):
        return RobertaModel(
            embeddings=from_torch(m.embeddings),
            encoder=from_torch(m.encoder),
        )


@register_from_torch(roberta_module.RobertaLMHead)
class RobertaLMHead(eqx.Module):
    dense: Linear
    layer_norm: LayerNorm
    decoder: Linear

    def __call__(self, features: Float[Array, "... D"]) -> Float[Array, "... Vocab"]:
        x = jax.nn.gelu(self.dense(features))
        x = self.layer_norm(x)
        return self.decoder(x)

    @staticmethod
    def from_torch(m: roberta_module.RobertaLMHead):
        return RobertaLMHead(
            dense=from_torch(m.dense),
            layer_norm=from_torch(m.layer_norm),
            decoder=from_torch(m.decoder),
        )


@register_from_torch(roberta_module.RobertaForMaskedLM)
class RobertaForMaskedLMEquinox(eqx.Module):
    roberta: RobertaModel
    lm_head: RobertaLMHead

    def __call__(
        self,
        input_ids: Int[Array, "B N"],
        attention_mask: Optional[Int[Array, "B N"]] = None,
        token_type_ids: Optional[Int[Array, "B N"]] = None,
    ):
        sequence_output = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return self.lm_head(sequence_output)

    @staticmethod
    def from_torch(m: roberta_module.RobertaForMaskedLM):
        return RobertaForMaskedLMEquinox(
            roberta=from_torch(m.roberta),
            lm_head=from_torch(m.lm_head),
        )


def from_pretrained(checkpoint_path: str) -> RobertaForMaskedLMEquinox:
    """Load a Sapiens checkpoint and convert it to JAX/Equinox."""
    model = transformers.RobertaForMaskedLM.from_pretrained(checkpoint_path)
    return from_torch(model)