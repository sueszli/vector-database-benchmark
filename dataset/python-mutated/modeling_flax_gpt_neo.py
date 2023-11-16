from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gpt_neo import GPTNeoConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'GPTNeoConfig'
_CHECKPOINT_FOR_DOC = 'EleutherAI/gpt-neo-1.3B'
GPT_NEO_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a Flax Linen\n    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a\n    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`GPTNeoConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
GPT_NEO_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):\n            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast\n            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class FlaxGPTNeoSelfAttention(nn.Module):
    config: GPTNeoConfig
    attention_type: str
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        dense = partial(nn.Dense, self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        (self.q_proj, self.k_proj, self.v_proj) = (dense(use_bias=False), dense(use_bias=False), dense(use_bias=False))
        self.out_proj = dense()
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype='bool'), dtype='bool')
        if self.attention_type == 'local':
            self.causal_mask = self.causal_mask ^ jnp.tril(self.causal_mask, -config.window_size)

    def _split_heads(self, hidden_states):
        if False:
            return 10
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        if False:
            print('Hello World!')
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if False:
            print('Hello World!')
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

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        query = self.q_proj(hidden_states) * jnp.sqrt(self.head_dim).astype(self.dtype)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        (query_length, key_length) = (query.shape[1], key.shape[1])
        if self.has_variable('cache', 'cached_key'):
            mask_shift = self.variables['cache']['cache_index']
            max_decoder_length = self.variables['cache']['cached_key'].shape[1]
            causal_mask = lax.dynamic_slice(self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length))
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        if self.has_variable('cache', 'cached_key') or init_cache:
            (key, value, attention_mask) = self._concatenate_to_cache(key, value, query, attention_mask)
        attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        attn_weights = dot_product_attention_weights(query, key, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_dropout, deterministic=deterministic, dtype=self.dtype, precision=None)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

class FlaxGPTNeoAttention(nn.Module):
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        attention_type = self.config.attention_layers[self.layer_id]
        self.attention = FlaxGPTNeoSelfAttention(self.config, attention_type, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        if False:
            print('Hello World!')
        return self.attention(hidden_states, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)

class FlaxGPTNeoMLP(nn.Module):
    config: GPTNeoConfig
    intermediate_size: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        embed_dim = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.c_fc = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        self.c_proj = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)
        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(rate=self.config.resid_dropout)

    def __call__(self, hidden_states, deterministic: bool=True):
        if False:
            return 10
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class FlaxGPTNeoBlock(nn.Module):
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        hidden_size = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.attn = FlaxGPTNeoAttention(self.config, layer_id=self.layer_id, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.mlp = FlaxGPTNeoMLP(self.config, inner_dim, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        if False:
            while True:
                i = 10
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        outputs = self.attn(hidden_states, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attn_output = outputs[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = residual + feed_forward_hidden_states
        return (hidden_states,) + outputs[1:]

class FlaxGPTNeoPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoConfig
    base_model_prefix = 'transformer'
    module_class: nn.Module = None

    def __init__(self, config: GPTNeoConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            print('Hello World!')
        input_ids = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        (params_rng, dropout_rng) = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)['params']
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
            i = 10
            return i + 15
        '\n        Args:\n            batch_size (`int`):\n                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.\n            max_length (`int`):\n                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized\n                cache.\n        '
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=None, position_ids=None, params: dict=None, past_key_values: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        (batch_size, sequence_length) = input_ids.shape
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `position_ids` when passing `past_key_values`.')
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
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
        outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), jnp.array(position_ids, dtype='i4'), not train, False, output_attentions, output_hidden_states, return_dict, rngs=rngs, mutable=mutable)
        if past_key_values is not None and return_dict:
            (outputs, past_key_values) = outputs
            outputs['past_key_values'] = unfreeze(past_key_values['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            (outputs, past_key_values) = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values['cache']),) + outputs[1:]
        return outputs

class FlaxGPTNeoBlockCollection(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.blocks = [FlaxGPTNeoBlock(self.config, layer_id=i, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(hidden_states, attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        return outputs

class FlaxGPTNeoModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.embed_dim = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.wte = nn.Embed(self.config.vocab_size, self.embed_dim, embedding_init=embedding_init)
        self.wpe = nn.Embed(self.config.max_position_embeddings, self.embed_dim, embedding_init=embedding_init)
        self.dropout = nn.Dropout(rate=self.config.embed_dropout)
        self.h = FlaxGPTNeoBlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, deterministic=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        input_embeds = self.wte(input_ids.astype('i4'))
        position_embeds = self.wpe(position_ids.astype('i4'))
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        outputs = self.h(hidden_states, attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[-1])

@add_start_docstrings('The bare GPTNeo Model transformer outputting raw hidden-states without any specific head on top.', GPT_NEO_START_DOCSTRING)
class FlaxGPTNeoModel(FlaxGPTNeoPreTrainedModel):
    module_class = FlaxGPTNeoModule
append_call_sample_docstring(FlaxGPTNeoModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)

class FlaxGPTNeoForCausalLMModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.transformer = FlaxGPTNeoModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))

    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        outputs = self.transformer(input_ids, attention_mask, position_ids, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables['params']['wte']['embedding'].T
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    The GPTNeo Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', GPT_NEO_START_DOCSTRING)
class FlaxGPTNeoForCausalLM(FlaxGPTNeoPreTrainedModel):
    module_class = FlaxGPTNeoForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array]=None):
        if False:
            return 10
        (batch_size, seq_length) = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype='i4')[None, :], (batch_size, seq_length))
        return {'past_key_values': past_key_values, 'attention_mask': extended_attention_mask, 'position_ids': position_ids}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        if False:
            i = 10
            return i + 15
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['position_ids'] = model_kwargs['position_ids'][:, -1:] + 1
        return model_kwargs
append_call_sample_docstring(FlaxGPTNeoForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)