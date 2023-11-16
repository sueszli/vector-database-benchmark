from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxBaseModelOutputWithPooling, FlaxBaseModelOutputWithPoolingAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions, FlaxMaskedLMOutput, FlaxMultipleChoiceModelOutput, FlaxSequenceClassifierOutput, FlaxTokenClassifierOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/bigbird-roberta-base'
_CONFIG_FOR_DOC = 'BigBirdConfig'
remat = nn_partitioning.remat

@flax.struct.dataclass
class FlaxBigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].

    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
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
    prediction_logits: jnp.ndarray = None
    seq_relationship_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

@flax.struct.dataclass
class FlaxBigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooled_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            pooled_output returned by FlaxBigBirdModel.
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
    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    pooled_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
BIG_BIRD_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)\n\n    This model is also a\n    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as\n    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and\n    behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`BigBirdConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
BIG_BIRD_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n        head_mask (`numpy.ndarray` of shape `({0})`, `optional):\n            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n\n'

class FlaxBigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
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
        if self.config.rescale_embeddings:
            inputs_embeds *= self.config.hidden_size ** 0.5
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class FlaxBigBirdSelfAttention(nn.Module):
    config: BigBirdConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
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
            while True:
                i = 10
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

class FlaxBigBirdBlockSparseAttention(nn.Module):
    config: BigBirdConfig
    block_sparse_seed: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.query = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.key = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.value = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))

    @staticmethod
    def transpose_for_scores(x, n_heads, head_size):
        if False:
            while True:
                i = 10
        new_x_shape = x.shape[:-1] + (n_heads, head_size)
        x = x.reshape(*new_x_shape)
        return jnp.transpose(x, axes=(0, 2, 1, 3))

    def __call__(self, hidden_states, attention_mask, deterministic=True, output_attentions=False):
        if False:
            print('Hello World!')
        n_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // n_heads
        (blocked_encoder_mask, band_mask, from_mask, to_mask) = self.create_masks_for_block_sparse_attn(attention_mask, self.config.block_size)
        query_layer = self.transpose_for_scores(self.query(hidden_states), n_heads, head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states), n_heads, head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states), n_heads, head_size)
        indices_prng_key = None
        if not deterministic:
            indices_prng_key = self.make_rng('indices')
        (attn_output, attn_weights) = self.bigbird_block_sparse_attention(query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, blocked_encoder_mask, blocked_encoder_mask, n_heads, head_size, indices_prng_key=indices_prng_key, deterministic=deterministic, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=output_attentions)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size: int):
        if False:
            return 10
        (batch_size, seq_length) = attention_mask.shape
        if seq_length % block_size != 0:
            raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            if False:
                return 10
            '\n            Create 3D attention mask from a 2D tensor mask.\n\n            Args:\n                from_blocked_mask: 2D Tensor of shape [batch_size,\n                from_seq_length//from_block_size, from_block_size].\n                to_blocked_mask: int32 Tensor of shape [batch_size,\n                to_seq_length//to_block_size, to_block_size].\n\n            Returns:\n                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,\n                3*to_block_size].\n            '
            exp_blocked_to_pad = jnp.concatenate([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2)
            band_mask = jnp.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask = jnp.expand_dims(band_mask, 1)
            return band_mask
        blocked_encoder_mask = attention_mask.reshape(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        from_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)
        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, n_heads, head_size, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=None):
        if False:
            return 10
        (bsz, _, from_seq_len, _) = query_layer.shape
        to_seq_len = key_layer.shape[2]
        from_block_size = to_block_size = self.config.block_size
        if from_seq_len % from_block_size != 0:
            raise ValueError('Query sided sequence length must be multiple of block size')
        if to_seq_len % to_block_size != 0:
            raise ValueError('Key/Value sided sequence length must be multiple of block size')
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        n_rand_blocks = self.config.num_random_blocks
        rsqrt_d = 1 / jnp.sqrt(head_size)
        attn_mask_penalty = -10000.0
        if from_seq_len in [1024, 3072, 4096]:
            max_seqlen = self.config.max_position_embeddings
            rand_attn = [self._bigbird_block_rand_mask(max_seqlen, max_seqlen, from_block_size, to_block_size, n_rand_blocks, indices_prng_key=indices_prng_key, deterministic=deterministic, last_idx=1024)[:from_seq_len // from_block_size - 2] for _ in range(n_heads)]
        else:
            if plan_from_length is None:
                (plan_from_length, plan_num_rand_blocks) = self._get_rand_attn_plan(from_seq_len, from_block_size, n_rand_blocks)
            rand_attn = self._bigbird_block_rand_mask_with_head(from_seq_length=from_seq_len, to_seq_length=to_seq_len, from_block_size=from_block_size, to_block_size=to_block_size, num_heads=n_heads, plan_from_length=plan_from_length, plan_num_rand_blocks=plan_num_rand_blocks, indices_prng_key=indices_prng_key)
        rand_attn = jnp.stack(rand_attn, axis=0)
        rand_attn = jnp.broadcast_to(rand_attn, (bsz,) + rand_attn.shape)
        rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size)
        blocked_query_matrix = query_layer.reshape(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        shape = (bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        gathered_key = self.jax_gather(blocked_key_matrix, rand_attn, batch_dims=2).reshape(*shape)
        gathered_value = self.jax_gather(blocked_value_matrix, rand_attn, batch_dims=2).reshape(*shape)
        first_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, 0], key_layer)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = jax.nn.softmax(first_product, axis=-1)
        first_context_layer = jnp.einsum('bhqk,bhkd->bhqd', first_attn_weights, value_layer)
        first_context_layer = jnp.expand_dims(first_context_layer, 2)
        second_key_mat = jnp.concatenate([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1], blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1], gathered_key[:, :, 0]], axis=2)
        second_value_mat = jnp.concatenate([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1], blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1], gathered_value[:, :, 0]], axis=2)
        second_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, 1], second_key_mat)
        second_seq_pad = jnp.concatenate([to_mask[:, :, :, :3 * to_block_size], to_mask[:, :, :, -to_block_size:], jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype)], axis=3)
        second_rand_pad = jnp.concatenate([jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype), rand_mask[:, :, 0]], axis=3)
        second_product = second_product * rsqrt_d
        second_product += (1.0 - jnp.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = jax.nn.softmax(second_product, axis=-1)
        second_context_layer = jnp.einsum('bhqk,bhkd->bhqd', second_attn_weights, second_value_mat)
        second_context_layer = jnp.expand_dims(second_context_layer, 2)
        exp_blocked_key_matrix = jnp.concatenate([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], axis=3)
        exp_blocked_value_matrix = jnp.concatenate([blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]], axis=3)
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
        inner_band_product = jnp.einsum('bhlqd,bhlkd->bhlqk', middle_query_matrix, exp_blocked_key_matrix)
        inner_band_product = inner_band_product * rsqrt_d
        rand_band_product = jnp.einsum('bhlqd,bhlkd->bhlqk', middle_query_matrix, gathered_key[:, :, 1:-1])
        rand_band_product = rand_band_product * rsqrt_d
        first_band_product = jnp.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, 0])
        first_band_product = first_band_product * rsqrt_d
        last_band_product = jnp.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, -1])
        last_band_product = last_band_product * rsqrt_d
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, :to_block_size], 3)) * attn_mask_penalty
        last_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, -to_block_size:], 3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
        band_product = jnp.concatenate([first_band_product, inner_band_product, rand_band_product, last_band_product], axis=-1)
        attn_weights = jax.nn.softmax(band_product, axis=-1)
        context_layer = jnp.einsum('bhlqk,bhlkd->bhlqd', attn_weights[:, :, :, :, to_block_size:4 * to_block_size], exp_blocked_value_matrix)
        context_layer += jnp.einsum('bhlqk,bhlkd->bhlqd', attn_weights[:, :, :, :, 4 * to_block_size:-to_block_size], gathered_value[:, :, 1:-1])
        context_layer += jnp.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0])
        context_layer += jnp.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1])
        second_last_key_mat = jnp.concatenate([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3], blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1], gathered_key[:, :, -1]], axis=2)
        second_last_value_mat = jnp.concatenate([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3], blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1], gathered_value[:, :, -1]], axis=2)
        second_last_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, -2], second_last_key_mat)
        second_last_seq_pad = jnp.concatenate([to_mask[:, :, :, :to_block_size], to_mask[:, :, :, -3 * to_block_size:], jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype)], axis=3)
        second_last_rand_pad = jnp.concatenate([jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype), rand_mask[:, :, -1]], axis=3)
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - jnp.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = jax.nn.softmax(second_last_product, axis=-1)
        second_last_context_layer = jnp.einsum('bhqk,bhkd->bhqd', second_last_attn_weights, second_last_value_mat)
        second_last_context_layer = jnp.expand_dims(second_last_context_layer, 2)
        last_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, -1], key_layer)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = jax.nn.softmax(last_product, axis=-1)
        last_context_layer = jnp.einsum('bhqk,bhkd->bhqd', last_attn_weights, value_layer)
        last_context_layer = jnp.expand_dims(last_context_layer, 2)
        context_layer = jnp.concatenate([first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer], axis=2)
        context_layer = context_layer.reshape(bsz, n_heads, from_seq_len, -1) * from_mask
        context_layer = jnp.transpose(context_layer, axes=(0, 2, 1, 3)).reshape(bsz, from_seq_len, -1)
        attention_probs = None
        return (context_layer, attention_probs)

    @staticmethod
    def jax_gather(params, indices, batch_dims=2):
        if False:
            print('Hello World!')
        '\n        Gather the indices from params correctly (equivalent to tf.gather but with modifications)\n\n        Args:\n            params: (bsz, n_heads, num_blocks, block_size, head_dim)\n            indices: (<num_blocks, 1)\n        '

        def _jax_gather(params, indices):
            if False:
                print('Hello World!')
            return params[indices]
        for _ in range(batch_dims):
            _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))
        return _jax_gather(params, indices)

    def _create_rand_mask_from_inputs(self, from_blocked_mask, to_blocked_mask, broadcasted_rand_attn, num_attention_heads, num_random_blocks, batch_size, from_seq_length, from_block_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create 3D attention mask from a 2D tensor mask.\n\n        Args:\n            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].\n            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].\n            broadcasted_rand_attn:\n                [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]\n            num_attention_heads: int. Number of attention heads.\n            num_random_blocks: int. Number of random chunks per row.\n            batch_size: int. Batch size for computation.\n            from_seq_length: int. length of from sequence.\n            from_block_size: int. size of block in from sequence.\n\n        Returns:\n            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,\n            from_block_size, num_rand_blocks*to_block_size].\n        '
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = self.jax_gather(to_blocked_mask, broadcasted_rand_attn, batch_dims=1)
        rand_mask = rand_mask.reshape(batch_size, num_attention_heads, num_windows, num_random_blocks * from_block_size)
        rand_mask = jnp.einsum('blq,bhlk->bhlqk', from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        if False:
            i = 10
            return i + 15
        '\n        Gives the plan of where to put random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            from_block_size: int. size of block in from sequence.\n            num_rand_blocks: int. Number of random chunks per row.\n\n        Returns:\n            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for\n            each block\n        '
        plan_from_length = []
        plan_num_rand_blocks = []
        if 2 * num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - num_rand_blocks // 2)
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)
        return (plan_from_length, plan_num_rand_blocks)

    @staticmethod
    def _bigbird_block_rand_mask(from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, last_idx: Optional[int]=-1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create adjacency list of random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            to_seq_length: int. length of to sequence.\n            from_block_size: int. size of block in from sequence.\n            to_block_size: int. size of block in to sequence.\n            num_rand_blocks: int. Number of random chunks per row.\n            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations.\n            deterministic: bool. When False random attention will be used.\n            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,\n            if positive then num_rand_blocks blocks chosen only up to last_idx.\n\n        Returns:\n            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks\n        '
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rand_attn = jnp.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=jnp.int32)
        if deterministic:
            return rand_attn
        middle_seq = jnp.arange(1, to_seq_length // to_block_size - 1, dtype=jnp.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > 2 * to_block_size:
            last = last_idx // to_block_size - 1
        r = num_rand_blocks
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[2:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == 2:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[3:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == from_seq_length // from_block_size - 3:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == from_seq_length // from_block_size - 2:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif start > last:
                start = last
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:start])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif end + 1 == last:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:start])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            else:
                concat_values = jnp.concatenate((middle_seq[:start], middle_seq[end + 1:last]))
                seq_values = jax.random.permutation(indices_prng_key, concat_values)[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
        return rand_attn

    def _bigbird_block_rand_mask_with_head(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_heads, plan_from_length, plan_num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, window_block_left=1, window_block_right=1, global_block_top=1, global_block_bottom=1, global_block_left=1, global_block_right=1):
        if False:
            while True:
                i = 10
        '\n        Create adjacency list of random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            to_seq_length: int. length of to sequence.\n            from_block_size: int. size of block in from sequence.\n            to_block_size: int. size of block in to sequence.\n            num_heads: int. total number of heads.\n            plan_from_length: list. plan from length where num_random_blocks are choosen from.\n            plan_num_rand_blocks: list. number of rand blocks within the plan.\n            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations.\n            deterministic: bool. When False random attention will be used.\n            window_block_left: int. number of blocks of window to left of a block.\n            window_block_right: int. number of blocks of window to right of a block.\n            global_block_top: int. number of blocks at the top.\n            global_block_bottom: int. number of blocks at the bottom.\n            global_block_left: int. Number of blocks globally used to the left.\n            global_block_right: int. Number of blocks globally used to the right.\n\n        Returns:\n            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by\n            num_rand_blocks\n        '
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        if from_seq_length not in plan_from_length:
            raise ValueError('Error from sequence length not in plan!')
        num_blocks = from_seq_length // from_block_size
        plan_block_length = jnp.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [jnp.zeros((num_blocks, sum(plan_num_rand_blocks[:max_plan_idx + 1])), dtype=jnp.int32) for i in range(num_heads)]
        if deterministic:
            for nh in range(num_heads):
                rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
            return rand_attn
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=plan_block_length[plan_idx - 1], to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                            rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(sum(plan_num_rand_blocks[:pl_id + 1]))
                        for h in range(num_heads):
                            single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[pl_id], num_rand_blocks=plan_num_rand_blocks[pl_id], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                            rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                    rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1):
        if False:
            while True:
                i = 10
        '\n        For a single row block get random row attention.\n\n        Args:\n            block_id: int. block id of row.\n            to_start_block_id: int. random attention column start id.\n            to_end_block_id: int. random attention column end id.\n            num_rand_blocks: int. number of random blocks to be selected.\n            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations\n            window_block_left: int. number of blocks of window to left of a block.\n            window_block_right: int. number of blocks of window to right of a block.\n            global_block_left: int. Number of blocks globally used to the left.\n            global_block_right: int. Number of blocks globally used to the right.\n\n        Returns:\n            row containing the random attention vector of size num_rand_blocks.\n        '
        to_block_list = jnp.arange(to_start_block_id, to_end_block_id, dtype=jnp.int32)
        perm_block = jax.random.permutation(indices_prng_key, to_block_list)
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)
        selected_random_blocks = []
        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return jnp.array(selected_random_blocks, dtype=jnp.int32)

class FlaxBigBirdSelfOutput(nn.Module):
    config: BigBirdConfig
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
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FlaxBigBirdAttention(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        if self.config.attention_type == 'original_full':
            self.self = FlaxBigBirdSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        elif self.config.attention_type == 'block_sparse':
            self.self = FlaxBigBirdBlockSparseAttention(self.config, block_sparse_seed=self.layer_id, dtype=self.dtype)
        else:
            raise ValueError(f'Your `config.attention_type` is {self.config.attention_type} but it can either be `original_full` or `block_sparse`')
        self.output = FlaxBigBirdSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=None, init_cache=False, deterministic=True, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        if self.config.attention_type == 'original_full':
            attn_outputs = self.self(hidden_states, attention_mask, layer_head_mask=layer_head_mask, key_value_states=key_value_states, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        else:
            attn_outputs = self.self(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs

class FlaxBigBirdIntermediate(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class FlaxBigBirdOutput(nn.Module):
    config: BigBirdConfig
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

class FlaxBigBirdLayer(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.attention = FlaxBigBirdAttention(self.config, layer_id=self.layer_id, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxBigBirdAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False):
        if False:
            for i in range(10):
                print('nop')
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

class FlaxBigBirdLayerCollection(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            i = 10
            return i + 15
        if self.gradient_checkpointing:
            FlaxBigBirdCheckpointLayer = remat(FlaxBigBirdLayer, static_argnums=(5, 6, 7))
            self.layers = [FlaxBigBirdCheckpointLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]
        else:
            self.layers = [FlaxBigBirdLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

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

class FlaxBigBirdEncoder(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            print('Hello World!')
        self.layer = FlaxBigBirdLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        return self.layer(hidden_states, attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class FlaxBigBirdPredictionHeadTransform(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)

class FlaxBigBirdLMPredictionHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.transform = FlaxBigBirdPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param('bias', self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        if False:
            return 10
        hidden_states = self.transform(hidden_states)
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states

class FlaxBigBirdOnlyMLMHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        if False:
            while True:
                i = 10
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states

class FlaxBigBirdPreTrainingHeads(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        if False:
            return 10
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)

class FlaxBigBirdPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BigBirdConfig
    base_model_prefix = 'bert'
    module_class: nn.Module = None

    def __init__(self, config: BigBirdConfig, input_shape: Optional[tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, gradient_checkpointing: bool=False, **kwargs):
        if False:
            while True:
                i = 10
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        if config.attention_type == 'block_sparse' and input_shape is None:
            input_shape = (1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        if False:
            return 10
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            while True:
                i = 10
        input_ids = jnp.zeros(input_shape, dtype='i4')
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        (params_rng, dropout_rng, indices_rng) = jax.random.split(rng, num=3)
        rngs = {'params': params_rng, 'dropout': dropout_rng, 'indices': indices_rng}
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

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, params: dict=None, dropout_rng: Optional[jax.random.PRNGKey]=None, indices_rng: Optional[jax.random.PRNGKey]=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, past_key_values: dict=None):
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        rngs = {}
        if indices_rng is not None:
            rngs['indices'] = indices_rng
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

class FlaxBigBirdModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings = FlaxBigBirdEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBigBirdEncoder(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.pooler = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)
        outputs = self.encoder(hidden_states, attention_mask, head_mask=head_mask, deterministic=deterministic, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled = nn.tanh(self.pooler(hidden_states[:, 0, :])) if self.add_pooling_layer else None
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states, pooler_output=pooled, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

@add_start_docstrings('The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdModel(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdModule
append_call_sample_docstring(FlaxBigBirdModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)

class FlaxBigBirdForPreTrainingModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.cls = FlaxBigBirdPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            print('Hello World!')
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        hidden_states = outputs[0]
        pooled_output = outputs[1]
        (prediction_scores, seq_relationship_score) = self.cls(hidden_states, pooled_output, shared_embedding=shared_embedding)
        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]
        return FlaxBigBirdForPreTrainingOutput(prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next\n    sentence prediction (classification)` head.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForPreTraining(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForPreTrainingModule
FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> from transformers import AutoTokenizer, FlaxBigBirdForPreTraining\n\n    >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")\n    >>> model = FlaxBigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")\n\n    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")\n    >>> outputs = model(**inputs)\n\n    >>> prediction_logits = outputs.prediction_logits\n    >>> seq_relationship_logits = outputs.seq_relationship_logits\n    ```\n'
overwrite_call_docstring(FlaxBigBirdForPreTraining, BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length') + FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING)
append_replace_return_docstrings(FlaxBigBirdForPreTraining, output_type=FlaxBigBirdForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)

class FlaxBigBirdForMaskedLMModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.bert = FlaxBigBirdModule(config=self.config, add_pooling_layer=False, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            i = 10
            return i + 15
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxMaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('BigBird Model with a `language modeling` head on top.', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForMaskedLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMaskedLMModule
append_call_sample_docstring(FlaxBigBirdForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

class FlaxBigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, features, deterministic=True):
        if False:
            for i in range(10):
                print('nop')
        x = features[:, 0, :]
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x

class FlaxBigBirdForSequenceClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            while True:
                i = 10
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.classifier = FlaxBigBirdClassificationHead(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)
        if not return_dict:
            return (logits,) + outputs[2:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForSequenceClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForSequenceClassificationModule
append_call_sample_docstring(FlaxBigBirdForSequenceClassification, _CHECKPOINT_FOR_DOC, FlaxSequenceClassifierOutput, _CONFIG_FOR_DOC)

class FlaxBigBirdForMultipleChoiceModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
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
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]
        return FlaxMultipleChoiceModelOutput(logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMultipleChoiceModule

    def __init__(self, config: BigBirdConfig, input_shape: Optional[tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            return 10
        if config.attention_type == 'block_sparse' and input_shape is None:
            input_shape = (1, 1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
overwrite_call_docstring(FlaxBigBirdForMultipleChoice, BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
append_call_sample_docstring(FlaxBigBirdForMultipleChoice, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC)

class FlaxBigBirdForTokenClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, add_pooling_layer=False, gradient_checkpointing=self.gradient_checkpointing)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            print('Hello World!')
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxTokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForTokenClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForTokenClassificationModule
append_call_sample_docstring(FlaxBigBirdForTokenClassification, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput, _CONFIG_FOR_DOC)

class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, encoder_output, deterministic=True):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dropout(encoder_output, deterministic=deterministic)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states

class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = False
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.config.num_labels = 2
        self.bert = FlaxBigBirdModule(self.config, dtype=self.dtype, add_pooling_layer=self.add_pooling_layer, gradient_checkpointing=self.gradient_checkpointing)
        self.qa_classifier = FlaxBigBirdForQuestionAnsweringHead(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, logits_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled_output = outputs[1] if self.add_pooling_layer else None
        logits = self.qa_classifier(hidden_states, deterministic=deterministic)
        if logits_mask is not None:
            logits = logits - logits_mask * 1000000.0
        (start_logits, end_logits) = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]
        return FlaxBigBirdForQuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits, pooled_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForQuestionAnswering(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForQuestionAnsweringModule

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, question_lengths=None, params: dict=None, dropout_rng: Optional[jax.random.PRNGKey]=None, indices_rng: Optional[jax.random.PRNGKey]=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))
        if question_lengths is None and input_ids is not None:
            question_lengths = jnp.argmax((input_ids == self.config.sep_token_id).astype('i4'), axis=-1) + 1
            question_lengths = jnp.expand_dims(question_lengths, axis=1)
        seqlen = input_ids.shape[1]
        logits_mask = None
        if question_lengths is not None:
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = (~logits_mask).astype('i4')
            logits_mask = jnp.expand_dims(logits_mask, axis=2)
            logits_mask = logits_mask.at[:, 0].set(False)
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        if indices_rng is not None:
            rngs['indices'] = indices_rng
        return self.module.apply({'params': params or self.params}, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), token_type_ids, jnp.array(position_ids, dtype='i4'), jnp.array(head_mask, dtype='i4'), logits_mask, not train, output_attentions, output_hidden_states, return_dict, rngs=rngs)

    @staticmethod
    def prepare_question_mask(q_lengths, maxlen: int):
        if False:
            i = 10
            return i + 15
        mask = jnp.arange(0, maxlen)
        mask = jnp.expand_dims(mask, axis=0) < q_lengths
        return mask
append_call_sample_docstring(FlaxBigBirdForQuestionAnswering, _CHECKPOINT_FOR_DOC, FlaxBigBirdForQuestionAnsweringModelOutput, _CONFIG_FOR_DOC)

class FlaxBigBirdForCausalLMModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        if False:
            return 10
        self.bert = FlaxBigBirdModule(config=self.config, add_pooling_layer=False, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, token_type_ids: Optional[jnp.ndarray]=None, head_mask: Optional[jnp.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            while True:
                i = 10
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxCausalLMOutputWithCrossAttentions(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

@add_start_docstrings('\n    BigBird Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for\n    autoregressive tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForCausalLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array]=None):
        if False:
            print('Hello World!')
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
append_call_sample_docstring(FlaxBigBirdForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC)