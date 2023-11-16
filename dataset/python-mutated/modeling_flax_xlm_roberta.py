"""Flax XLM-RoBERTa model."""
from typing import Callable, Optional, Tuple
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
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxBaseModelOutputWithPooling, FlaxBaseModelOutputWithPoolingAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions, FlaxMaskedLMOutput, FlaxMultipleChoiceModelOutput, FlaxQuestionAnsweringModelOutput, FlaxSequenceClassifierOutput, FlaxTokenClassifierOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xlm_roberta import XLMRobertaConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'xlm-roberta-base'
_CONFIG_FOR_DOC = 'XLMRobertaConfig'
remat = nn_partitioning.remat
FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ['xlm-roberta-base', 'xlm-roberta-large']

def create_position_ids_from_input_ids(input_ids, padding_idx):
    if False:
        return 10
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        input_ids: jnp.ndarray\n        padding_idx: int\n\n    Returns: jnp.ndarray\n    "
    mask = (input_ids != padding_idx).astype('i4')
    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype('i4') * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype('i4') * mask
    return incremental_indices.astype('i4') + padding_idx
XLM_ROBERTA_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)\n\n    This model is also a\n    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as\n    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and\n    behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`XLMRobertaConfig`]): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n'
XLM_ROBERTA_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n        head_mask (`numpy.ndarray` of shape `({0})`, `optional):\n            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class FlaxXLMRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.config.hidden_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.config.hidden_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.token_type_embeddings = nn.Embed(self.config.type_vocab_size, self.config.hidden_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
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

class FlaxXLMRobertaSelfAttention(nn.Module):
    config: XLMRobertaConfig
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
            for i in range(10):
                print('nop')
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if False:
            for i in range(10):
                print('nop')
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

class FlaxXLMRobertaSelfOutput(nn.Module):
    config: XLMRobertaConfig
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
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FlaxXLMRobertaAttention(nn.Module):
    config: XLMRobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.self = FlaxXLMRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxXLMRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=None, init_cache=False, deterministic=True, output_attentions: bool=False):
        if False:
            print('Hello World!')
        attn_outputs = self.self(hidden_states, attention_mask, layer_head_mask=layer_head_mask, key_value_states=key_value_states, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs

class FlaxXLMRobertaIntermediate(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class FlaxXLMRobertaOutput(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool=True):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states

class FlaxXLMRobertaLayer(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.attention = FlaxXLMRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxXLMRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxXLMRobertaOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxXLMRobertaAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False):
        if False:
            return 10
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

class FlaxXLMRobertaLayerCollection(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gradient_checkpointing:
            FlaxXLMRobertaCheckpointLayer = remat(FlaxXLMRobertaLayer, static_argnums=(5, 6, 7))
            self.layers = [FlaxXLMRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]
        else:
            self.layers = [FlaxXLMRobertaLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
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

class FlaxXLMRobertaEncoder(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.layer = FlaxXLMRobertaLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        return self.layer(hidden_states, attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class FlaxXLMRobertaPooler(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)

class FlaxXLMRobertaLMHead(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.bias = self.param('bias', self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN['gelu'](hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states

class FlaxXLMRobertaClassificationHead(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))

    def __call__(self, hidden_states, deterministic=True):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class FlaxXLMRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMRobertaConfig
    base_model_prefix = 'xlm-roberta'
    module_class: nn.Module = None

    def __init__(self, config: XLMRobertaConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, gradient_checkpointing: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        if False:
            i = 10
            return i + 15
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            return 10
        input_ids = jnp.zeros(input_shape, dtype='i4')
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
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
            for i in range(10):
                print('nop')
        '\n        Args:\n            batch_size (`int`):\n                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.\n            max_length (`int`):\n                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized\n                cache.\n        '
        input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        attention_mask = jnp.ones_like(input_ids, dtype='i4')
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, past_key_values: dict=None):
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
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

class FlaxXLMRobertaModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.embeddings = FlaxXLMRobertaEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxXLMRobertaEncoder(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.pooler = FlaxXLMRobertaPooler(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids: Optional[jnp.ndarray]=None, position_ids: Optional[jnp.ndarray]=None, head_mask: Optional[jnp.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)
        outputs = self.encoder(hidden_states, attention_mask, head_mask=head_mask, deterministic=deterministic, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states, pooler_output=pooled, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

@add_start_docstrings('The bare XLM RoBERTa Model transformer outputting raw hidden-states without any specific head on top.', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaModel(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaModule
append_call_sample_docstring(FlaxXLMRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)

class FlaxXLMRobertaForMaskedLMModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.roberta = FlaxXLMRobertaModule(config=self.config, add_pooling_layer=False, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = FlaxXLMRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxMaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('XLM RoBERTa Model with a `language modeling` head on top.', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForMaskedLM(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForMaskedLMModule
append_call_sample_docstring(FlaxXLMRobertaForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC, mask='<mask>')

class FlaxXLMRobertaForSequenceClassificationModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.roberta = FlaxXLMRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False, gradient_checkpointing=self.gradient_checkpointing)
        self.classifier = FlaxXLMRobertaClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLM Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForSequenceClassification(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForSequenceClassificationModule
append_call_sample_docstring(FlaxXLMRobertaForSequenceClassification, _CHECKPOINT_FOR_DOC, FlaxSequenceClassifierOutput, _CONFIG_FOR_DOC)

class FlaxXLMRobertaForMultipleChoiceModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.roberta = FlaxXLMRobertaModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]
        return FlaxMultipleChoiceModelOutput(logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for RocStories/SWAG tasks.\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForMultipleChoice(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForMultipleChoiceModule
overwrite_call_docstring(FlaxXLMRobertaForMultipleChoice, XLM_ROBERTA_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
append_call_sample_docstring(FlaxXLMRobertaForMultipleChoice, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC)

class FlaxXLMRobertaForTokenClassificationModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.roberta = FlaxXLMRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False, gradient_checkpointing=self.gradient_checkpointing)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxTokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLM Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.\n    for Named-Entity-Recognition (NER) tasks.\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForTokenClassification(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForTokenClassificationModule
append_call_sample_docstring(FlaxXLMRobertaForTokenClassification, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC)

class FlaxXLMRobertaForQuestionAnsweringModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.roberta = FlaxXLMRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False, gradient_checkpointing=self.gradient_checkpointing)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.qa_outputs(hidden_states)
        (start_logits, end_logits) = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]
        return FlaxQuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLM Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a\n    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForQuestionAnswering(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForQuestionAnsweringModule
append_call_sample_docstring(FlaxXLMRobertaForQuestionAnswering, _CHECKPOINT_FOR_DOC, FlaxQuestionAnsweringModelOutput, _CONFIG_FOR_DOC)

class FlaxXLMRobertaForCausalLMModule(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.roberta = FlaxXLMRobertaModule(config=self.config, add_pooling_layer=False, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = FlaxXLMRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, token_type_ids: Optional[jnp.ndarray]=None, head_mask: Optional[jnp.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxCausalLMOutputWithCrossAttentions(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

@add_start_docstrings('\n    XLM Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for\n    autoregressive tasks.\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForCausalLM(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForCausalLMModule

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
            for i in range(10):
                print('nop')
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['position_ids'] = model_kwargs['position_ids'][:, -1:] + 1
        return model_kwargs
append_call_sample_docstring(FlaxXLMRobertaForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC)