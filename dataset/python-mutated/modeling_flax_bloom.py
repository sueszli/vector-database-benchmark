"""Flax BLOOM model."""
import math
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'bigscience/bloom'
_CONFIG_FOR_DOC = 'BloomConfig'
BLOOM_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a Flax Linen\n    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a\n    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
BLOOM_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`BloomTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):\n            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast\n            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

def build_alibi_tensor(attention_mask: jnp.ndarray, num_heads: int, dtype: Optional[jnp.dtype]=jnp.float32):
    if False:
        print('Hello World!')
    '\n    Flax implementation of the BLOOM Alibi tensor. BLOOM Alibi tensor is not causal as the original paper mentions, it\n    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value\n    `softmax(l+a) = softmax(l)`. Based on\n    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742\n    Link to paper: https://arxiv.org/abs/2108.12409\n\n    Args:\n        attention_mask (`jnp.ndarray`):\n            Token-wise attention mask, this should be of shape `(batch_size, max_seq_len)`.\n        num_heads (`int`):\n            Number of attention heads.\n        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):\n            The data type (dtype) of the output tensor.\n\n    Returns: Alibi tensor of shape `(batch_size * num_heads, 1, max_seq_len)`.\n    '
    (batch_size, seq_length) = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = jnp.array(2 ** (-2 ** (-(math.log2(closest_power_of_2) - 3))), dtype=jnp.float32)
    powers = jnp.arange(1, 1 + closest_power_of_2, dtype=jnp.float32)
    slopes = jax.lax.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = jnp.array(2 ** (-2 ** (-(math.log2(2 * closest_power_of_2) - 3))), dtype=jnp.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = jnp.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=jnp.float32)
        slopes = jnp.cat([slopes, jax.lax.pow(extra_base, extra_powers)], axis=0)
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    alibi = jnp.expand_dims(alibi, axis=2)
    return jnp.asarray(alibi, dtype)

class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        dense = partial(nn.Dense, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.query_key_value = dense(self.hidden_size * 3)
        self.dense = dense(self.hidden_size)
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def _split_heads(self, hidden_states):
        if False:
            i = 10
            return i + 15
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_heads, self.head_dim * 3))

    def _merge_heads(self, hidden_states):
        if False:
            i = 10
            return i + 15
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if False:
            i = 10
            return i + 15
        '\n        This function takes projected key, value states from a single input token and concatenates the states to cached\n        states from previous steps. This function is slighly adapted from the official Flax repository:\n        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252\n        '
        is_initialized = self.has_variable('cache', 'cached_key')
        cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable('cache', 'cached_value', jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable('cache', 'cache_index', lambda : jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            (*batch_dims, max_length, num_heads, depth_per_head) = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            pad_mask = jnp.broadcast_to(jnp.arange(max_length) < cur_index + num_updated_cache_vectors, tuple(batch_dims) + (1, num_updated_cache_vectors, max_length))
            attention_mask = combine_masks(pad_mask, attention_mask)
        return (key, value, attention_mask)

    def __call__(self, hidden_states, residual, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        if False:
            while True:
                i = 10
        (batch_size, seq_length) = hidden_states.shape[:2]
        fused_qkv = self.query_key_value(hidden_states)
        fused_qkv = self._split_heads(fused_qkv)
        (query, key, value) = jnp.split(fused_qkv, 3, axis=-1)
        causal_attention_mask = make_causal_mask(attention_mask, dtype='bool')
        causal_attention_mask_shift = self.variables['cache']['cache_index'] if self.has_variable('cache', 'cached_key') else 0
        if self.has_variable('cache', 'cached_key'):
            max_decoder_length = self.variables['cache']['cached_key'].shape[1]
            causal_attention_mask = jax.lax.dynamic_slice(causal_attention_mask, (0, 0, causal_attention_mask_shift, 0), (1, 1, seq_length, max_decoder_length))
        causal_attention_mask = jnp.broadcast_to(causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_attention_mask)
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        if self.has_variable('cache', 'cached_key') or init_cache:
            (key, value, attention_mask) = self._concatenate_to_cache(key, value, query, attention_mask)
        mask_value = jnp.finfo(self.dtype).min
        attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, mask_value).astype(self.dtype))
        attention_bias = attention_bias + alibi
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(query, key, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_dropout, deterministic=deterministic, dtype=attention_dtype)
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.dense(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        attn_output = attn_output + residual
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

class BloomGELU(nn.Module):

    def setup(self):
        if False:
            print('Hello World!')
        self.dtype = jnp.float32

    def __call__(self, x):
        if False:
            while True:
                i = 10
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        hidden_size = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.dense_h_to_4h = nn.Dense(4 * hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.dense_4h_to_h = nn.Dense(hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        self.act = BloomGELU()

    def __call__(self, hidden_states, residual, deterministic: bool=True):
        if False:
            print('Hello World!')
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        intermediate_output = self.dense_4h_to_h(hidden_states)
        intermediate_output = intermediate_output + residual
        hidden_states = self.hidden_dropout(intermediate_output, deterministic=deterministic)
        return hidden_states

class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)
        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(self, hidden_states, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        if False:
            for i in range(10):
                print('nop')
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(layernorm_output, residual=residual, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        post_layernorm = self.post_attention_layernorm(attention_output)
        if self.apply_residual_connection_post_layernorm:
            residual = post_layernorm
        else:
            residual = attention_output
        output = self.mlp(post_layernorm, residual, deterministic=deterministic)
        outputs = (output,) + outputs
        return outputs

class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BloomConfig
    base_model_prefix = 'transformer'
    module_class: nn.Module = None

    def __init__(self, config: BloomConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            i = 10
            return i + 15
        input_ids = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        (params_rng, dropout_rng) = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        if False:
            print('Hello World!')
        '\n        Args:\n            batch_size (`int`):\n                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.\n            max_length (`int`):\n                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized\n                cache.\n        '
        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=None, past_key_values: dict=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, sequence_length) = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False
        outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), not train, False, output_attentions, output_hidden_states, return_dict, rngs=rngs, mutable=mutable)
        if past_key_values is not None and return_dict:
            (outputs, past_key_values) = outputs
            outputs['past_key_values'] = unfreeze(past_key_values['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            (outputs, past_key_values) = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values['cache']),) + outputs[1:]
        return outputs

class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.layers = [FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype) for layer_number in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False):
        if False:
            print('Hello World!')
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer_number in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_number](hidden_states, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        return outputs

class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.embed_dim = self.config.hidden_size
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.embed_dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(self, input_ids=None, attention_mask=None, deterministic=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)
        outputs = self.h(hidden_states, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        if not return_dict:
            return tuple((v for v in [outputs[0], outputs[-1]] if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[-1])

@add_start_docstrings('The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.', BLOOM_START_DOCSTRING)
class FlaxBloomModel(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomModule
append_call_sample_docstring(FlaxBloomModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)

class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))

    def __call__(self, input_ids, attention_mask, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        outputs = self.transformer(input_ids, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables['params']['word_embeddings']['embedding'].T
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', BLOOM_START_DOCSTRING)
class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array]=None):
        if False:
            return 10
        (batch_size, seq_length) = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if attention_mask is not None:
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        return {'past_key_values': past_key_values, 'attention_mask': extended_attention_mask}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        if False:
            return 10
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        return model_kwargs
append_call_sample_docstring(FlaxBloomForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)