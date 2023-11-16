""" TensorFlow Wav2Vec2 model."""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_wav2vec2 import Wav2Vec2Config
logger = logging.get_logger(__name__)
_HIDDEN_STATES_START_POSITION = 2
_CHECKPOINT_FOR_DOC = 'facebook/wav2vec2-base-960h'
_CONFIG_FOR_DOC = 'Wav2Vec2Config'
TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/wav2vec2-base-960h', 'facebook/wav2vec2-large-960h', 'facebook/wav2vec2-large-960h-lv60', 'facebook/wav2vec2-large-960h-lv60-self']
LARGE_NEGATIVE = -100000000.0

@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`TFWav2Vec2BaseModelOutput`], with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: tf.Tensor = None
    extract_features: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

def _sample_without_replacement(distribution, num_samples):
    if False:
        return 10
    '\n    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see\n    https://github.com/tensorflow/tensorflow/issues/9260 for more info\n    '
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    (_, indices) = tf.nn.top_k(distribution + z, num_samples)
    return indices

def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    if False:
        print('Hello World!')
    '\n    Scatter function as in PyTorch with indices in format (batch_dim, indixes)\n    '
    indices_shape = shape_list(batch_indices)
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1])
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)

def _compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int, min_masks: int=0) -> tf.Tensor:
    if False:
        while True:
            i = 10
    "\n    Computes random mask spans for a given shape\n\n    Args:\n        shape: the shape for which to compute masks.\n            should be of size 2 where first element is batch size and 2nd is timesteps\n        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements\n        mask_prob:\n            probability for each token to be chosen as start of the span to be masked. this will be multiplied by\n            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.\n            however due to overlaps, the actual number will be smaller (unless no_overlap is True)\n        mask_length: size of the mask\n        min_masks: minimum number of masked spans\n\n    Adapted from [fairseq's\n    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).\n    "
    (batch_size, sequence_length) = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    tf.debugging.assert_less(mask_length, sequence_length, message=f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    num_masked_spans = tf.squeeze(num_masked_spans)
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))
    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    spec_aug_mask = _scatter_values_on_batch_indices(tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask))
    return spec_aug_mask

def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int]=None):
    if False:
        i = 10
        return i + 15
    '\n    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.\n    '
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    return (one_cst - expanded_mask) * LARGE_NEGATIVE

class TFWav2Vec2GroupNorm(tf.keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(self, groups: int=32, axis: int=-1, epsilon: float=0.001, center: bool=True, scale: bool=True, beta_initializer: tf.keras.initializers.Initializer='zeros', gamma_initializer: tf.keras.initializers.Initializer='ones', beta_regularizer: tf.keras.regularizers.Regularizer=None, gamma_regularizer: tf.keras.regularizers.Regularizer=None, beta_constraint: tf.keras.constraints.Constraint=None, gamma_constraint: tf.keras.constraints.Constraint=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        if False:
            print('Hello World!')
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        if False:
            while True:
                i = 10
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        (reshaped_inputs, group_shape) = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs
        return outputs

    def get_config(self):
        if False:
            print('Hello World!')
        config = {'groups': self.groups, 'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale, 'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer), 'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer), 'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer), 'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer), 'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint), 'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        if False:
            while True:
                i = 10
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        if False:
            return 10
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return (reshaped_inputs, group_shape)
        else:
            return (inputs, group_shape)

    def _apply_normalization(self, reshaped_inputs, input_shape):
        if False:
            print('Hello World!')
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        (mean, variance) = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)
        (gamma, beta) = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(reshaped_inputs, mean=mean, variance=variance, scale=gamma, offset=beta, variance_epsilon=self.epsilon)
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return (gamma, beta)

    def _check_if_input_shape_is_none(self, input_shape):
        if False:
            return 10
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of input tensor should have a defined dimension but the layer received an input with shape ' + str(input_shape) + '.')

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        dim = input_shape[self.axis]
        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be more than the number of channels (' + str(dim) + ').')
        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a multiple of the number of channels (' + str(dim) + ').')

    def _check_axis(self):
        if False:
            return 10
        if self.axis == 0:
            raise ValueError('You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead')

    def _create_input_spec(self, input_shape):
        if False:
            print('Hello World!')
        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):
        if False:
            i = 10
            return i + 15
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(shape=shape, name='gamma', initializer=self.gamma_initializer, regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.center:
            self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer, regularizer=self.beta_regularizer, constraint=self.beta_constraint)
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        if False:
            return 10
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape

class TFWav2Vec2WeightNormConv1D(tf.keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(filters=filters, kernel_size=kernel_size, groups=groups, padding='valid', use_bias=True, bias_initializer='he_normal', **kwargs)
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.initialized = False
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        if False:
            while True:
                i = 10
        'Set the norm of the weight vector.'
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        if False:
            return 10
        'Generate normalized weights.'
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if False:
            return 10
        if not self.built:
            input_shape = input_shape.as_list()
            if input_shape[-2] is not None:
                input_shape[-2] += self.explicit_padding * 2
            super().build(input_shape)
            self.kernel = tf.Variable(tf.transpose(self.kernel), name='weight_v', trainable=True)
            self.weight_v = self.kernel
            self.weight_g = self.add_weight(name='weight_g', shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1), initializer='ones', dtype=self.weight_v.dtype, trainable=True)
            self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        if not self.initialized:
            self._init_norm()
            self.initialized = True
        self._normalize_kernel()
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().call(padded_inputs)
        return output

class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, layer_id: int=0, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = tf.keras.layers.Conv1D(filters=self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], strides=config.conv_stride[layer_id], use_bias=config.conv_bias, name='conv')
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, layer_id: int=0, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = tf.keras.layers.Conv1D(filters=self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], strides=config.conv_stride[layer_id], use_bias=config.conv_bias, name='conv')
        self.layer_norm = tf.keras.layers.LayerNormalization(name='layer_norm', epsilon=config.layer_norm_eps)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, layer_id: int=0, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = tf.keras.layers.Conv1D(filters=self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], strides=config.conv_stride[layer_id], use_bias=config.conv_bias, name='conv')
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.layer_norm = TFWav2Vec2GroupNorm(groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name='layer_norm')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.conv = TFWav2Vec2WeightNormConv1D(filters=config.hidden_size, kernel_size=config.num_conv_pos_embeddings, groups=config.num_conv_pos_embedding_groups, explicit_padding=config.num_conv_pos_embeddings // 2, name='conv')
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):

    def __init__(self, num_conv_pos_embeddings, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if False:
            print('Hello World!')
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :-self.num_pad_remove, :]
        return hidden_states

class TFWav2Vec2FeatureEncoder(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        if config.feat_extract_norm == 'group':
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f'conv_layers.{0}')] + [TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f'conv_layers.{i + 1}') for i in range(config.num_feat_extract_layers - 1)]
        elif config.feat_extract_norm == 'layer':
            conv_layers = [TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f'conv_layers.{i}') for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']")
        self.conv_layers = conv_layers

    def call(self, input_values):
        if False:
            i = 10
            return i + 15
        hidden_states = tf.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states

class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config, **kwargs)
        warnings.warn(f'The class `{self.__class__.__name__}` has been depreciated and will be removed in Transformers v5. Use `{self.__class__.__bases__[0].__name__}` instead.', FutureWarning)

class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.projection = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='projection')
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return (hidden_states, norm_hidden_states)

class TFWav2Vec2Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='k_proj')
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='q_proj')
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='v_proj')
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='out_proj')

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        if False:
            while True:
                i = 10
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, key_value_states: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, attention_mask: tf.Tensor | None=None, layer_head_mask: tf.Tensor | None=None, training: Optional[bool]=False) -> Tuple[tf.Tensor, tf.Tensor | None]:
        if False:
            while True:
                i = 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, embed_dim) = shape_list(hidden_states)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)
        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
        tf.debugging.assert_equal(shape_list(attn_weights), [bsz * self.num_heads, tgt_len, src_len], message=f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {shape_list(attn_weights)}')
        if attention_mask is not None:
            tf.debugging.assert_equal(shape_list(attention_mask), [bsz, 1, tgt_len, src_len], message=f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}')
            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        attn_weights = stable_softmax(attn_weights, axis=-1)
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.num_heads], message=f'Head mask for a single layer should be of size {self.num_heads}, but is {shape_list(layer_head_mask)}')
            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)
        tf.debugging.assert_equal(shape_list(attn_output), [bsz * self.num_heads, tgt_len, self.head_dim], message=f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {shape_list(attn_output)}')
        attn_output = tf.transpose(tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
        return (attn_output, attn_weights, past_key_value)

class TFWav2Vec2FeedForward(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.intermediate_dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='intermediate_dense')
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        self.output_dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='output_dense')
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states

class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False, name='attention')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.feed_forward = TFWav2Vec2FeedForward(config, name='feed_forward')
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='final_layer_norm')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            while True:
                i = 10
        attn_residual = hidden_states
        (hidden_states, attn_weights, _) = self.attention(hidden_states, attention_mask=attention_mask, training=training)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False, name='attention')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.feed_forward = TFWav2Vec2FeedForward(config, name='feed_forward')
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='final_layer_norm')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        (hidden_states, attn_weights, _) = self.attention(hidden_states, attention_mask=attention_mask, training=training)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class TFWav2Vec2Encoder(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name='pos_conv_embed')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f'layers.{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: Optional[bool]=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            if training and dropout_probability < self.config.layerdrop:
                continue
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name='pos_conv_embed')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f'layers.{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: Optional[bool]=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, training=training)
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            if training and dropout_probability < self.config.layerdrop:
                continue
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

@keras_serializable
class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name='feature_extractor')
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name='feature_projection')
        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name='encoder')
        else:
            self.encoder = TFWav2Vec2Encoder(config, name='encoder')

    def build(self, input_shape: tf.TensorShape):
        if False:
            print('Hello World!')
        self.masked_spec_embed = self.add_weight(shape=(self.config.hidden_size,), initializer='uniform', trainable=True, name='masked_spec_embed')
        super().build(input_shape)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        if False:
            i = 10
            return i + 15
        '\n        Computes the output length of the convolutional layers\n        '

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                while True:
                    i = 10
            return (input_length - kernel_size) // stride + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None=None):
        if False:
            while True:
                i = 10
        '\n        Masks extracted features along time axis and/or along feature axis according to\n        [SpecAugment](https://arxiv.org/abs/1904.08779).\n        '
        (batch_size, sequence_length, hidden_size) = shape_list(hidden_states)
        if not getattr(self.config, 'apply_spec_augment', True):
            return hidden_states
        if mask_time_indices is not None:
            hidden_states = tf.where(tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool), self.masked_spec_embed[tf.newaxis, tf.newaxis, :], hidden_states)
        elif self.config.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, min_masks=2)
            hidden_states = tf.where(tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool), self.masked_spec_embed[tf.newaxis, tf.newaxis, :], hidden_states)
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length)
            hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)
        return hidden_states

    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False, **kwargs: Any):
        if False:
            while True:
                i = 10
        extract_features = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)
        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))
            attention_mask = tf.sequence_mask(output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype)
        (hidden_states, extract_features) = self.feature_projection(extract_features, training=training)
        mask_time_indices = kwargs.get('mask_time_indices', None)
        if training:
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return TFWav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = 'wav2vec2'
    main_input_name = 'input_values'

    @property
    def input_signature(self):
        if False:
            i = 10
            return i + 15
        return {'input_values': tf.TensorSpec((None, None), tf.float32, name='input_values'), 'attention_mask': tf.TensorSpec((None, None), tf.float32, name='attention_mask')}

    @property
    def dummy_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'input_values': tf.random.uniform(shape=(1, 500), dtype=tf.float32), 'attention_mask': tf.ones(shape=(1, 500), dtype=tf.float32)}

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        logger.warning(f'\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish to train/fine-tune this model, you need a GPU or a TPU')

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        if False:
            i = 10
            return i + 15
        '\n        Computes the output length of the convolutional layers\n        '
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                print('Hello World!')
            return tf.math.floordiv(input_length - kernel_size, stride) + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: tf.Tensor, add_adapter=None):
        if False:
            return 10
        non_padded_lengths = tf.math.cumsum(attention_mask, axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = tf.cast(output_lengths, tf.int32)
        batch_size = tf.shape(attention_mask)[0]
        attention_mask = tf.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, name='attention_mask')
        attention_mask = tf.tensor_scatter_nd_update(attention_mask, indices=tf.stack([tf.range(batch_size), output_lengths - 1], axis=1), updates=tf.ones([batch_size], dtype=attention_mask.dtype))
        attention_mask = tf.reverse(attention_mask, axis=[-1])
        attention_mask = tf.cumsum(attention_mask, axis=-1)
        attention_mask = tf.reverse(attention_mask, axis=[-1])
        attention_mask = tf.cast(attention_mask, tf.bool)
        return attention_mask
WAV_2_VEC_2_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_values` only and nothing else: `model(input_values)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_values": input_values, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
WAV_2_VEC_2_INPUTS_DOCSTRING = "\n    Args:\n        input_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_values` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert `input_values` indices into associated vectors\n            than the model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.', WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):

    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name='wav2vec2')

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoProcessor, TFWav2Vec2Model\n        >>> from datasets import load_dataset\n        >>> import soundfile as sf\n\n        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")\n        >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")\n\n\n        >>> def map_to_array(batch):\n        ...     speech, _ = sf.read(batch["file"])\n        ...     batch["speech"] = speech\n        ...     return batch\n\n\n        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n        >>> ds = ds.map(map_to_array)\n\n        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1\n        >>> hidden_states = model(input_values).last_hidden_state\n        ```'
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        return_dict = return_dict if return_dict else self.config.return_dict
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):

    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name='wav2vec2')
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name='lm_head')

    def freeze_feature_extractor(self):
        if False:
            i = 10
            return i + 15
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor.trainable = False

    @unpack_inputs
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, labels: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_values` docstring) Tokens with indices set to `-100` are ignored (masked),\n            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoProcessor, TFWav2Vec2ForCTC\n        >>> from datasets import load_dataset\n        >>> import soundfile as sf\n\n        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")\n        >>> model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")\n\n\n        >>> def map_to_array(batch):\n        ...     speech, _ = sf.read(batch["file"])\n        ...     batch["speech"] = speech\n        ...     return batch\n\n\n        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n        >>> ds = ds.map(map_to_array)\n\n        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1\n        >>> logits = model(input_values).logits\n        >>> predicted_ids = tf.argmax(logits, axis=-1)\n\n        >>> transcription = processor.decode(predicted_ids[0])\n\n        >>> # compute loss\n        >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"\n\n        >>> # Pass transcription as `text` to encode labels\n        >>> labels = processor(text=transcription, return_tensors="tf").input_ids\n\n        >>> loss = model(input_values, labels=labels).loss\n        ```'
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, training=training)
        logits = self.lm_head(hidden_states)
        if labels is not None:
            if tf.reduce_max(labels) >= self.config.vocab_size:
                raise ValueError(f'Label values must be <= vocab_size: {self.config.vocab_size}')
            attention_mask = attention_mask if attention_mask is not None else tf.ones_like(input_values, dtype=tf.float32)
            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, axis=-1))
            labels_mask = tf.cast(labels >= 0, tf.int32)
            target_lengths = tf.reduce_sum(labels_mask, axis=-1)
            loss = tf.nn.ctc_loss(logits=logits, labels=labels, logit_length=input_lengths, label_length=target_lengths, blank_index=self.config.pad_token_id, logits_time_major=False)
            if self.config.ctc_loss_reduction == 'sum':
                loss = tf.reduce_sum(loss)
            if self.config.ctc_loss_reduction == 'mean':
                loss = tf.reduce_mean(loss)
            loss = tf.reshape(loss, (1,))
        else:
            loss = None
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return TFCausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name='wav2vec2')
        self.num_layers = config.num_hidden_layers + 1
        with tf.name_scope(self._name_scope()):
            if config.use_weighted_layer_sum:
                self.layer_weights = self.add_weight(shape=(self.num_layers,), initializer='ones', trainable=True, name='layer_weights')
        self.config = config
        self.projector = tf.keras.layers.Dense(units=config.classifier_proj_size, name='projector')
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, activation=None, name='classifier')

    def freeze_feature_extractor(self):
        if False:
            return 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for layer in self.wav2vec2.layers:
            layer.trainable = False

    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, labels: tf.Tensor | None=None, training: bool=False) -> TFSequenceClassifierOutput | Tuple[tf.Tensor]:
        if False:
            return 10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = tf.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(shape_list(hidden_states)[1], attention_mask)
            padding_mask_float = tf.cast(padding_mask, hidden_states.dtype)
            hidden_states = tf.multiply(hidden_states, tf.expand_dims(padding_mask_float, axis=-1))
            pooled_output = tf.divide(tf.reduce_sum(hidden_states, axis=1), tf.expand_dims(tf.reduce_sum(padding_mask_float, axis=1), axis=1))
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, self.config.num_labels]))
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)