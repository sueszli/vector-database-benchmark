from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions, FlaxMaskedLMOutput, FlaxMultipleChoiceModelOutput, FlaxQuestionAnsweringModelOutput, FlaxSequenceClassifierOutput, FlaxTokenClassifierOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_electra import ElectraConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/electra-small-discriminator'
_CONFIG_FOR_DOC = 'ElectraConfig'
remat = nn_partitioning.remat

@flax.struct.dataclass
class FlaxElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ElectraForPreTraining`].

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
ELECTRA_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)\n\n    This model is also a Flax Linen\n    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a\n    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
ELECTRA_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n        head_mask (`numpy.ndarray` of shape `({0})`, `optional):\n            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n\n'

class FlaxElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.token_type_embeddings = nn.Embed(self.config.type_vocab_size, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool=True):
        if False:
            i = 10
            return i + 15
        inputs_embeds = self.word_embeddings(input_ids.astype('i4'))
        position_embeds = self.position_embeddings(position_ids.astype('i4'))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype('i4'))
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class FlaxElectraSelfAttention(nn.Module):
    config: ElectraConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError('`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`                    : {self.config.num_attention_heads}')
        self.query = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.key = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.value = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype='bool'), dtype='bool')

    def _split_heads(self, hidden_states):
        if False:
            return 10
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if False:
            return 10
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

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic=True, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]
        query_states = self.query(hidden_states)
        if is_cross_attention:
            key_states = self.key(key_value_states)
            value_states = self.value(key_value_states)
        else:
            key_states = self.key(hidden_states)
            value_states = self.value(hidden_states)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        if self.causal:
            (query_length, key_length) = (query_states.shape[1], key_states.shape[1])
            if self.has_variable('cache', 'cached_key'):
                mask_shift = self.variables['cache']['cache_index']
                max_decoder_length = self.variables['cache']['cached_key'].shape[1]
                causal_mask = lax.dynamic_slice(self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length))
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        if self.causal and (self.has_variable('cache', 'cached_key') or init_cache):
            (key_states, value_states, attention_mask) = self._concatenate_to_cache(key_states, value_states, query_states, attention_mask)
        if attention_mask is not None:
            attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_probs_dropout_prob, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype, precision=None)
        if layer_head_mask is not None:
            attn_weights = jnp.einsum('...hqk,h->...hqk', attn_weights, layer_head_mask)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

class FlaxElectraSelfOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool=True):
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FlaxElectraAttention(nn.Module):
    config: ElectraConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.self = FlaxElectraSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxElectraSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=None, init_cache=False, deterministic=True, output_attentions: bool=False):
        if False:
            return 10
        attn_outputs = self.self(hidden_states, attention_mask, layer_head_mask=layer_head_mask, key_value_states=key_value_states, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs

class FlaxElectraIntermediate(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class FlaxElectraOutput(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool=True):
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states

class FlaxElectraLayer(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.attention = FlaxElectraAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxElectraIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxElectraOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxElectraAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False):
        if False:
            print('Hello World!')
        attention_outputs = self.attention(hidden_states, attention_mask, layer_head_mask=layer_head_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask=encoder_attention_mask, layer_head_mask=layer_head_mask, key_value_states=encoder_hidden_states, deterministic=deterministic, output_attentions=output_attentions)
            attention_output = cross_attention_outputs[0]
        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs

class FlaxElectraLayerCollection(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gradient_checkpointing:
            FlaxElectraCheckpointLayer = remat(FlaxElectraLayer, static_argnums=(5, 6, 7))
            self.layers = [FlaxElectraCheckpointLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]
        else:
            self.layers = [FlaxElectraLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for                         {head_mask.shape[0]}.')
        for (i, layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, attention_mask, head_mask[i] if head_mask is not None else None, encoder_hidden_states, encoder_attention_mask, init_cache, deterministic, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

class FlaxElectraEncoder(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.layer = FlaxElectraLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        return self.layer(hidden_states, attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class FlaxElectraGeneratorPredictions(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)

    def __call__(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class FlaxElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.dense_prediction = nn.Dense(1, dtype=self.dtype)

    def __call__(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        hidden_states = self.dense_prediction(hidden_states).squeeze(-1)
        return hidden_states

class FlaxElectraPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ElectraConfig
    base_model_prefix = 'electra'
    module_class: nn.Module = None

    def __init__(self, config: ElectraConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, gradient_checkpointing: bool=False, **kwargs):
        if False:
            while True:
                i = 10
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        if False:
            return 10
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            i = 10
            return i + 15
        input_ids = jnp.zeros(input_shape, dtype='i4')
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        (params_rng, dropout_rng) = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states, encoder_attention_mask, return_dict=False)
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False)
        random_params = module_init_outputs['params']
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
            return 10
        '\n        Args:\n            batch_size (`int`):\n                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.\n            max_length (`int`):\n                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized\n                cache.\n        '
        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids, dtype='i4')
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, past_key_values: dict=None):
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if self.config.add_cross_attention:
            if past_key_values:
                inputs['cache'] = past_key_values
                mutable = ['cache']
            else:
                mutable = False
            outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), token_type_ids=jnp.array(token_type_ids, dtype='i4'), position_ids=jnp.array(position_ids, dtype='i4'), head_mask=jnp.array(head_mask, dtype='i4'), encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, deterministic=not train, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, rngs=rngs, mutable=mutable)
            if past_key_values is not None and return_dict:
                (outputs, past_key_values) = outputs
                outputs['past_key_values'] = unfreeze(past_key_values['cache'])
                return outputs
            elif past_key_values is not None and (not return_dict):
                (outputs, past_key_values) = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values['cache']),) + outputs[1:]
        else:
            outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), token_type_ids=jnp.array(token_type_ids, dtype='i4'), position_ids=jnp.array(position_ids, dtype='i4'), head_mask=jnp.array(head_mask, dtype='i4'), deterministic=not train, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, rngs=rngs)
        return outputs

class FlaxElectraModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.encoder = FlaxElectraEncoder(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask: Optional[np.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)
        return self.encoder(embeddings, attention_mask, head_mask=head_mask, deterministic=deterministic, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

@add_start_docstrings('The bare Electra Model transformer outputting raw hidden-states without any specific head on top.', ELECTRA_START_DOCSTRING)
class FlaxElectraModel(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraModule
append_call_sample_docstring(FlaxElectraModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)

class FlaxElectraTiedDense(nn.Module):
    embedding_size: int
    dtype: jnp.dtype = jnp.float32
    precision = None
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        if False:
            while True:
                i = 10
        self.bias = self.param('bias', self.bias_init, (self.embedding_size,))

    def __call__(self, x, kernel):
        if False:
            i = 10
            return i + 15
        x = jnp.asarray(x, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        bias = jnp.asarray(self.bias, self.dtype)
        return y + bias

class FlaxElectraForMaskedLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        prediction_scores = self.generator_predictions(hidden_states)
        if self.config.tie_word_embeddings:
            shared_embedding = self.electra.variables['params']['embeddings']['word_embeddings']['embedding']
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        else:
            prediction_scores = self.generator_lm_head(prediction_scores)
        if not return_dict:
            return (prediction_scores,) + outputs[1:]
        return FlaxMaskedLMOutput(logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('Electra Model with a `language modeling` head on top.', ELECTRA_START_DOCSTRING)
class FlaxElectraForMaskedLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForMaskedLMModule
append_call_sample_docstring(FlaxElectraForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

class FlaxElectraForPreTrainingModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.discriminator_predictions = FlaxElectraDiscriminatorPredictions(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.discriminator_predictions(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxElectraForPreTrainingOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.\n\n    It is recommended to load the discriminator checkpoint into that model.\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForPreTraining(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForPreTrainingModule
FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> from transformers import AutoTokenizer, FlaxElectraForPreTraining\n\n    >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")\n    >>> model = FlaxElectraForPreTraining.from_pretrained("google/electra-small-discriminator")\n\n    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")\n    >>> outputs = model(**inputs)\n\n    >>> prediction_logits = outputs.logits\n    ```\n'
overwrite_call_docstring(FlaxElectraForPreTraining, ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length') + FLAX_ELECTRA_FOR_PRETRAINING_DOCSTRING)
append_replace_return_docstrings(FlaxElectraForPreTraining, output_type=FlaxElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)

class FlaxElectraForTokenClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxTokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Electra model with a token classification head on top.\n\n    Both the discriminator and generator may be loaded into this model.\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForTokenClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForTokenClassificationModule
append_call_sample_docstring(FlaxElectraForTokenClassification, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC)

def identity(x, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return x

class FlaxElectraSequenceSummary(nn.Module):
    """
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.summary = identity
        if hasattr(self.config, 'summary_use_proj') and self.config.summary_use_proj:
            if hasattr(self.config, 'summary_proj_to_labels') and self.config.summary_proj_to_labels and (self.config.num_labels > 0):
                num_classes = self.config.num_labels
            else:
                num_classes = self.config.hidden_size
            self.summary = nn.Dense(num_classes, dtype=self.dtype)
        activation_string = getattr(self.config, 'summary_activation', None)
        self.activation = ACT2FN[activation_string] if activation_string else lambda x: x
        self.first_dropout = identity
        if hasattr(self.config, 'summary_first_dropout') and self.config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(self.config.summary_first_dropout)
        self.last_dropout = identity
        if hasattr(self.config, 'summary_last_dropout') and self.config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(self.config.summary_last_dropout)

    def __call__(self, hidden_states, cls_index=None, deterministic: bool=True):
        if False:
            print('Hello World!')
        '\n        Compute a single vector summary of a sequence hidden states.\n\n        Args:\n            hidden_states (`jnp.ndarray` of shape `[batch_size, seq_len, hidden_size]`):\n                The hidden states of the last layer.\n            cls_index (`jnp.ndarray` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):\n                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.\n\n        Returns:\n            `jnp.ndarray`: The summary of the sequence hidden states.\n        '
        output = hidden_states[:, 0]
        output = self.first_dropout(output, deterministic=deterministic)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output, deterministic=deterministic)
        return output

class FlaxElectraForMultipleChoiceModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.sequence_summary = FlaxElectraSequenceSummary(config=self.config, dtype=self.dtype)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled_output = self.sequence_summary(hidden_states, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)
        if not return_dict:
            return (reshaped_logits,) + outputs[1:]
        return FlaxMultipleChoiceModelOutput(logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForMultipleChoice(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForMultipleChoiceModule
overwrite_call_docstring(FlaxElectraForMultipleChoice, ELECTRA_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
append_call_sample_docstring(FlaxElectraForMultipleChoice, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC)

class FlaxElectraForQuestionAnsweringModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.qa_outputs(hidden_states)
        (start_logits, end_logits) = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]
        return FlaxQuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForQuestionAnswering(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForQuestionAnsweringModule
append_call_sample_docstring(FlaxElectraForQuestionAnswering, _CHECKPOINT_FOR_DOC, FlaxQuestionAnsweringModelOutput, _CONFIG_FOR_DOC)

class FlaxElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool=True):
        if False:
            return 10
        x = hidden_states[:, 0, :]
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN['gelu'](x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x

class FlaxElectraForSequenceClassificationModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            print('Hello World!')
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.classifier = FlaxElectraClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states, deterministic=deterministic)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Electra Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForSequenceClassification(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForSequenceClassificationModule
append_call_sample_docstring(FlaxElectraForSequenceClassification, _CHECKPOINT_FOR_DOC, FlaxSequenceClassifierOutput, _CONFIG_FOR_DOC)

class FlaxElectraForCausalLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask: Optional[jnp.ndarray]=None, token_type_ids: Optional[jnp.ndarray]=None, position_ids: Optional[jnp.ndarray]=None, head_mask: Optional[jnp.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        prediction_scores = self.generator_predictions(hidden_states)
        if self.config.tie_word_embeddings:
            shared_embedding = self.electra.variables['params']['embeddings']['word_embeddings']['embedding']
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        else:
            prediction_scores = self.generator_lm_head(prediction_scores)
        if not return_dict:
            return (prediction_scores,) + outputs[1:]
        return FlaxCausalLMOutputWithCrossAttentions(logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

@add_start_docstrings('\n    Electra Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for\n    autoregressive tasks.\n    ', ELECTRA_START_DOCSTRING)
class FlaxElectraForCausalLM(FlaxElectraPreTrainedModel):
    module_class = FlaxElectraForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array]=None):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['position_ids'] = model_kwargs['position_ids'][:, -1:] + 1
        return model_kwargs
append_call_sample_docstring(FlaxElectraForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC)