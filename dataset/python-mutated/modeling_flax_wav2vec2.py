""" Flax Wav2Vec2 model."""
from functools import partial
from typing import Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config
logger = logging.get_logger(__name__)

@flax.struct.dataclass
class FlaxWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`FlaxWav2Vec2BaseModelOutput`], with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`jnp.ndarray` of shape `(batch_size, sequence_length, last_conv_dim)`):
            Sequence of extracted feature vectors of the last convolutional layer of the model with `last_conv_dim`
            being the dimension of the last convolutional layer.
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
    last_hidden_state: jnp.ndarray = None
    extract_features: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

@flax.struct.dataclass
class FlaxWav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FlaxWav2Vec2ForPreTrainingOutput`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when model is in train mode, `jnp.ndarray` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`jnp.ndarray` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`jnp.ndarray` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
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
    projected_states: jnp.ndarray = None
    projected_quantized_states: jnp.ndarray = None
    codevector_perplexity: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

def _compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int, attention_mask: Optional[np.ndarray]=None, min_masks: int=0) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for\n    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on\n    CPU as part of the preprocessing during training.\n\n    Args:\n        shape: the shape for which to compute masks.\n            should be of size 2 where first element is batch size and 2nd is timesteps\n        mask_prob:\n            probability for each token to be chosen as start of the span to be masked. this will be multiplied by\n            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.\n            however due to overlaps, the actual number will be smaller (unless no_overlap is True)\n        mask_length: size of the mask\n        min_masks: minimum number of masked spans\n\n    '
    (batch_size, sequence_length) = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    num_masked_spans = int(mask_prob * sequence_length / mask_length + np.random.rand(1).item())
    num_masked_spans = max(num_masked_spans, min_masks)
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = np.array([np.random.choice(np.arange(sequence_length - (mask_length - 1)), num_masked_spans, replace=False) for _ in range(batch_size)])
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, num_masked_spans, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, num_masked_spans * mask_length)
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, num_masked_spans, mask_length)).reshape(batch_size, num_masked_spans * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    if attention_mask is not None:
        spec_aug_mask = np.where(attention_mask, spec_aug_mask, False)
    return spec_aug_mask

def _sample_negative_indices(features_shape: Tuple, num_negatives: int, attention_mask: Optional[np.ndarray]=None):
    if False:
        print('Hello World!')
    '\n    Sample `num_negatives` vectors from feature vectors.\n    '
    (batch_size, sequence_length, hidden_size) = features_shape
    if sequence_length <= 1:
        raise ValueError(f'`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({(batch_size, sequence_length, hidden_size)}).')
    sampled_negative_indices = []
    for batch_idx in range(batch_size):
        high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
        sampled_indices_slice = np.random.randint(0, high, size=(num_negatives * sequence_length,))
        sampled_negative_indices.append(sampled_indices_slice)
    sampled_negative_indices = np.asarray(sampled_negative_indices, dtype=np.int32)
    feature_indices = np.broadcast_to(np.arange(sequence_length)[:, None], (sequence_length, num_negatives)).flatten()
    sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1
    for batch_idx in range(1, batch_size):
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length
    return sampled_negative_indices
WAV_2_VEC_2_START_DOCSTRING = '\n    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech\n    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael\n    Auli.\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a Flax Linen\n    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a\n    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
WAV_2_VEC_2_INPUTS_DOCSTRING = '\n    Args:\n        input_values (`jnp.ndarray` of shape `(batch_size, sequence_length)`):\n            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file\n            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install\n            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and\n            conversion into a tensor of type `jnp.ndarray`. See [`Wav2Vec2Processor.__call__`] for details.\n        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,\n            1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask) .. warning:: `attention_mask` should only be passed\n            if the corresponding processor has `config.return_attention_mask == True`. For all models whose processor\n            has `config.return_attention_mask == False`, such as\n            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), `attention_mask` should **not** be\n            passed to avoid degraded performance when doing batched inference. For such models `input_values` should\n            simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly\n            different results depending on whether `input_values` is padded or not.\n        mask_time_indices (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict\n            masked extracted features in *config.proj_codevector_dim* space.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        self.out_conv_dim = self.config.conv_dim[self.layer_id]
        self.conv = nn.Conv(features=self.config.conv_dim[self.layer_id], kernel_size=(self.config.conv_kernel[self.layer_id],), strides=(self.config.conv_stride[self.layer_id],), use_bias=self.config.conv_bias, kernel_init=jax.nn.initializers.he_normal(), padding='VALID', dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class FlaxConvWithWeightNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.conv = nn.Conv(features=self.config.hidden_size, kernel_size=(self.config.num_conv_pos_embeddings,), kernel_init=jax.nn.initializers.he_normal(), padding='VALID', feature_group_count=self.config.num_conv_pos_embedding_groups, dtype=self.dtype)
        weight_shape = (self.conv.features, self.conv.features // self.conv.feature_group_count, self.conv.kernel_size[0])
        self.weight_v = self.param('weight_v', jax.nn.initializers.he_normal(), weight_shape)
        self.weight_g = self.param('weight_g', lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :])
        self.bias = self.param('bias', jax.nn.initializers.zeros, (self.conv.features,))
        self.prev_padding = self.conv.kernel_size[0] // 2

    def _get_normed_weights(self):
        if False:
            for i in range(10):
                print('nop')
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        if False:
            while True:
                i = 10
        kernel = self._get_normed_weights()
        hidden_states = jnp.pad(hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0)))
        hidden_states = self.conv.apply({'params': {'kernel': kernel.T, 'bias': self.bias}}, hidden_states)
        return hidden_states

class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.conv = FlaxConvWithWeightNorm(self.config, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = hidden_states.transpose((0, 1, 2))
        hidden_states = self.conv(hidden_states)
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :-self.num_pad_remove, :]
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose((0, 1, 2))
        return hidden_states

class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        if self.config.feat_extract_norm == 'layer':
            self.layers = [FlaxWav2Vec2LayerNormConvLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype) for i in range(self.config.num_feat_extract_layers)]
        elif self.config.feat_extract_norm == 'group':
            raise NotImplementedError("At the moment only ``config.feat_extact_norm == 'layer'`` is supported")
        else:
            raise ValueError(f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group', 'layer']")

    def __call__(self, hidden_states):
        if False:
            return 10
        for (i, conv_layer) in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        return hidden_states

class FlaxWav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, input_values, freeze_feature_encoder=False):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = input_values[:, :, None]
        hidden_states = self.conv_layers(hidden_states)
        if freeze_feature_encoder:
            hidden_states = jax.lax.stop_gradient(hidden_states)
        return hidden_states

class FlaxWav2Vec2FeatureProjection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.projection = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.feat_proj_dropout)

    def __call__(self, hidden_states, deterministic=True):
        if False:
            for i in range(10):
                print('nop')
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return (hidden_states, norm_hidden_states)

class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if False:
            while True:
                i = 10
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        dense = partial(nn.Dense, self.embed_dim, use_bias=self.bias, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        (self.q_proj, self.k_proj, self.v_proj) = (dense(), dense(), dense())
        self.out_proj = dense()
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        if False:
            i = 10
            return i + 15
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray]=None, attention_mask: Optional[jnp.ndarray]=None, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        if False:
            print('Hello World!')
        'Input shape: Batch x Time x Channel'
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        if attention_mask is not None:
            attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.dropout, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype, precision=None)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights)

class FlaxWav2Vec2FeedForward(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.intermediate_dropout = nn.Dropout(rate=self.config.activation_dropout)
        self.intermediate_dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        if isinstance(self.config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_act_fn = self.config.hidden_act
        self.output_dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.output_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def __call__(self, hidden_states, deterministic=True):
        if False:
            print('Hello World!')
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.attention = FlaxWav2Vec2Attention(config=self.config, embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads, dropout=self.config.attention_dropout, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False):
        if False:
            while True:
                i = 10
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        (hidden_states, attn_weights) = self.attention(hidden_states, attention_mask=attention_mask, deterministic=deterministic)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states), deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.layers = [FlaxWav2Vec2EncoderLayerStableLayerNorm(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.pos_conv_embed = FlaxWav2Vec2PositionalConvEmbedding(self.config, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layers = FlaxWav2Vec2EncoderLayerStableLayerNormCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            while True:
                i = 10
        if attention_mask is not None:
            hidden_states = jnp.where(jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape), hidden_states, 0)
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        outputs = self.layers(hidden_states, attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = self.layer_norm(outputs[0])
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=outputs.attentions)

class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.num_groups = self.config.num_codevector_groups
        self.num_vars = self.config.num_codevectors_per_group
        if self.config.codevector_dim % self.num_groups != 0:
            raise ValueError(f'`config.codevector_dim {self.config.codevector_dim} must be divisible by `config.num_codevector_groups` {self.num_groups} for concatenation')
        self.codevectors = self.param('codevectors', jax.nn.initializers.uniform(), (1, self.num_groups * self.num_vars, self.config.codevector_dim // self.num_groups))
        self.weight_proj = nn.Dense(self.num_groups * self.num_vars, kernel_init=jax.nn.initializers.normal(1.0), dtype=self.dtype)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if False:
            print('Hello World!')
        if mask is not None:
            mask_extended = jnp.broadcast_to(mask.flatten()[:, None, None], probs.shape)
            probs = jnp.where(mask_extended, probs, jnp.zeros_like(probs))
            marginal_probs = probs.sum(axis=0) / mask.sum()
        else:
            marginal_probs = probs.mean(axis=0)
        perplexity = jnp.exp(-jnp.sum(marginal_probs * jnp.log(marginal_probs + 1e-07), axis=-1)).sum()
        return perplexity

    def __call__(self, hidden_states, mask_time_indices=None, deterministic=True, temperature=1):
        if False:
            print('Hello World!')
        (batch_size, sequence_length, hidden_size) = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size * sequence_length * self.num_groups, -1)
        if not deterministic:
            gumbel_rng = self.make_rng('gumbel')
            gumbels = jax.random.gumbel(gumbel_rng, hidden_states.shape)
            codevector_probs = nn.softmax((hidden_states + gumbels) / temperature)
            codevector_soft_dist = nn.softmax(hidden_states.reshape(batch_size * sequence_length, self.num_groups, -1), axis=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            codevector_idx = hidden_states.argmax(axis=-1)
            codevector_probs = jax.nn.one_hot(codevector_idx, hidden_states.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        codevector_probs = codevector_probs.reshape(batch_size * sequence_length, -1)
        codevectors_per_group = jnp.expand_dims(codevector_probs, axis=-1) * self.codevectors
        codevectors = codevectors_per_group.reshape(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).reshape(batch_size, sequence_length, -1)
        return (codevectors, perplexity)

class FlaxWav2Vec2Adapter(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        if self.config.output_hidden_size != self.config.hidden_size:
            self.proj = nn.Dense(self.config.output_hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
            self.proj_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        else:
            self.proj = self.proj_layer_norm = None
        self.layers = FlaxWav2Vec2AdapterLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        if False:
            return 10
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

class FlaxWav2Vec2AdapterLayer(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.conv = nn.Conv(features=2 * self.config.output_hidden_size, kernel_size=(self.config.adapter_kernel_size,), strides=(self.config.adapter_stride,), padding=((1, 1),), kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.glu(hidden_states, axis=2)
        return hidden_states

class FlaxWav2Vec2AdapterLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.layers = [FlaxWav2Vec2AdapterLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_adapter_layers)]

    def __call__(self, hidden_states):
        if False:
            i = 10
            return i + 15
        for conv_layer in self.layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states

class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix: str = 'wav2vec2'
    main_input_name = 'input_values'
    module_class: nn.Module = None

    def __init__(self, config: Wav2Vec2Config, input_shape: Tuple=(1, 1024), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            print('Hello World!')
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            i = 10
            return i + 15
        input_values = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_values)
        (params_rng, dropout_rng) = jax.random.split(rng, 2)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_values, attention_mask, return_dict=False)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, freeze_feature_encoder: bool=False, return_dict: Optional[bool]=None):
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        (batch_size, sequence_length) = input_values.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        return self.module.apply(inputs, jnp.array(input_values, dtype='f4'), jnp.array(attention_mask, dtype='i4'), mask_time_indices, not train, output_attentions, output_hidden_states, freeze_feature_encoder, return_dict, rngs=rngs)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        if False:
            while True:
                i = 10
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)

class FlaxWav2Vec2Module(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(self.config, dtype=self.dtype)
        self.feature_projection = FlaxWav2Vec2FeatureProjection(self.config, dtype=self.dtype)
        self.masked_spec_embed = self.param('masked_spec_embed', jax.nn.initializers.uniform(), (self.config.hidden_size,))
        if self.config.do_stable_layer_norm:
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(self.config, dtype=self.dtype)
        else:
            raise NotImplementedError('``config.do_stable_layer_norm is False`` is currently not supported.')
        self.adapter = FlaxWav2Vec2Adapter(self.config, dtype=self.dtype) if self.config.add_adapter else None

    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, deterministic=True, output_attentions=None, output_hidden_states=None, freeze_feature_encoder=False, return_dict=None):
        if False:
            print('Hello World!')
        extract_features = self.feature_extractor(input_values, freeze_feature_encoder=freeze_feature_encoder)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_adapter=False)
        (hidden_states, extract_features) = self.feature_projection(extract_features, deterministic=deterministic)
        if mask_time_indices is not None:
            hidden_states = jnp.where(jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape), jnp.broadcast_to(self.masked_spec_embed[None, None, :], hidden_states.shape), hidden_states)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return FlaxWav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the output length of the convolutional layers\n        '
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                while True:
                    i = 10
            return (input_length - kernel_size) // stride + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None):
        if False:
            print('Hello World!')
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        batch_size = attention_mask.shape[0]
        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype('bool')
        return attention_mask

@add_start_docstrings('The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.', WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2Module
FLAX_WAV2VEC2_MODEL_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> from transformers import AutoProcessor, FlaxWav2Vec2Model\n    >>> from datasets import load_dataset\n    >>> import soundfile as sf\n\n    >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")\n    >>> model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")\n\n\n    >>> def map_to_array(batch):\n    ...     speech, _ = sf.read(batch["file"])\n    ...     batch["speech"] = speech\n    ...     return batch\n\n\n    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n    >>> ds = ds.map(map_to_array)\n\n    >>> input_values = processor(\n    ...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"\n    ... ).input_values  # Batch size 1\n    >>> hidden_states = model(input_values).last_hidden_state\n    ```\n'
overwrite_call_docstring(FlaxWav2Vec2Model, WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxWav2Vec2Model, output_type=FlaxWav2Vec2BaseModelOutput, config_class=Wav2Vec2Config)

class FlaxWav2Vec2ForCTCModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.final_dropout)
        self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, deterministic=True, output_attentions=None, output_hidden_states=None, freeze_feature_encoder=False, return_dict=None):
        if False:
            while True:
                i = 10
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, mask_time_indices=mask_time_indices, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, freeze_feature_encoder=freeze_feature_encoder, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return (logits,) + outputs[2:]
        return FlaxCausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        if False:
            while True:
                i = 10
        '\n        Computes the output length of the convolutional layers\n        '
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                return 10
            return (input_length - kernel_size) // stride + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

@add_start_docstrings('Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2ForCTCModule
FLAX_WAV2VEC2_FOR_CTC_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> import jax.numpy as jnp\n    >>> from transformers import AutoProcessor, FlaxWav2Vec2ForCTC\n    >>> from datasets import load_dataset\n    >>> import soundfile as sf\n\n    >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60")\n    >>> model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")\n\n\n    >>> def map_to_array(batch):\n    ...     speech, _ = sf.read(batch["file"])\n    ...     batch["speech"] = speech\n    ...     return batch\n\n\n    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n    >>> ds = ds.map(map_to_array)\n\n    >>> input_values = processor(\n    ...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"\n    ... ).input_values  # Batch size 1\n    >>> logits = model(input_values).logits\n    >>> predicted_ids = jnp.argmax(logits, axis=-1)\n\n    >>> transcription = processor.decode(predicted_ids[0])\n    >>> # should give:  "A MAN SAID TO THE UNIVERSE SIR I EXIST"\n    ```\n'
overwrite_call_docstring(FlaxWav2Vec2ForCTC, WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_CTC_DOCSTRING)
append_replace_return_docstrings(FlaxWav2Vec2ForCTC, output_type=FlaxCausalLMOutput, config_class=Wav2Vec2Config)

class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        self.dropout_features = nn.Dropout(self.config.feat_quantizer_dropout)
        self.quantizer = FlaxWav2Vec2GumbelVectorQuantizer(self.config, dtype=self.dtype)
        self.project_q = nn.Dense(self.config.proj_codevector_dim, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.project_hid = nn.Dense(self.config.proj_codevector_dim, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, gumbel_temperature: int=1, deterministic: bool=True, output_attentions=None, output_hidden_states=None, freeze_feature_encoder=False, return_dict=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Example:\n\n        ```python\n\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, mask_time_indices=mask_time_indices, deterministic=deterministic, freeze_feature_encoder=freeze_feature_encoder, return_dict=return_dict)
        transformer_features = self.project_hid(outputs[0])
        extract_features = self.dropout_features(outputs[1], deterministic=deterministic)
        (quantized_features, codevector_perplexity) = self.quantizer(extract_features, mask_time_indices, deterministic=deterministic, temperature=gumbel_temperature)
        quantized_features = self.project_q(quantized_features)
        if not return_dict:
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return FlaxWav2Vec2ForPreTrainingOutput(projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        if False:
            return 10
        '\n        Computes the output length of the convolutional layers\n        '
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                print('Hello World!')
            return (input_length - kernel_size) // stride + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

@add_start_docstrings('Wav2Vec2 Model with a quantizer and `VQ` head on top.', WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForPreTraining(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2ForPreTrainingModule

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, gumbel_temperature: int=1, params: dict=None, dropout_rng: jax.random.PRNGKey=None, gumbel_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, freeze_feature_encoder: bool=False, return_dict: Optional[bool]=None):
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        (batch_size, sequence_length) = input_values.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        if gumbel_rng is not None:
            rngs['gumbel'] = gumbel_rng
        inputs = {'params': params or self.params}
        return self.module.apply(inputs, jnp.array(input_values, dtype='f4'), jnp.array(attention_mask, dtype='i4'), mask_time_indices, gumbel_temperature, not train, output_attentions, output_hidden_states, freeze_feature_encoder, return_dict, rngs=rngs)
FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> import optax\n    >>> import numpy as np\n    >>> import jax.numpy as jnp\n    >>> from transformers import AutoFeatureExtractor, FlaxWav2Vec2ForPreTraining\n    >>> from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices\n    >>> from datasets import load_dataset\n    >>> import soundfile as sf\n\n    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")\n    >>> model = FlaxWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")\n\n\n    >>> def map_to_array(batch):\n    ...     speech, _ = sf.read(batch["file"])\n    ...     batch["speech"] = speech\n    ...     return batch\n\n\n    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n    >>> ds = ds.map(map_to_array)\n\n    >>> input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1\n\n    >>> # compute masked indices\n    >>> batch_size, raw_sequence_length = input_values.shape\n    >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)\n    >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)\n\n    >>> outputs = model(input_values, mask_time_indices=mask_time_indices)\n\n    >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)\n    >>> cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)\n\n    >>> # show that cosine similarity is much higher than random\n    >>> assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5\n    ```\n'
overwrite_call_docstring(FlaxWav2Vec2ForPreTraining, WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING)
append_replace_return_docstrings(FlaxWav2Vec2ForPreTraining, output_type=FlaxWav2Vec2ForPreTrainingOutput, config_class=Wav2Vec2Config)