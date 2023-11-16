""" TF 2.0 Swin Transformer model."""
from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_swin import SwinConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'SwinConfig'
_CHECKPOINT_FOR_DOC = 'microsoft/swin-tiny-patch4-window7-224'
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]
_IMAGE_CLASS_CHECKPOINT = 'microsoft/swin-tiny-patch4-window7-224'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'
TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/swin-tiny-patch4-window7-224']

@dataclass
class TFSwinEncoderOutput(ModelOutput):
    """
    Swin encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None

@dataclass
class TFSwinModelOutput(ModelOutput):
    """
    Swin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None

@dataclass
class TFSwinMaskedImageModelingOutput(ModelOutput):
    """
    Swin masked image model outputs.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: tf.Tensor | None = None
    reconstruction: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None

    @property
    def logits(self):
        if False:
            print('Hello World!')
        warnings.warn('logits attribute is deprecated and will be removed in version 5 of Transformers. Please use the reconstruction attribute to retrieve the final output instead.', FutureWarning)
        return self.reconstruction

@dataclass
class TFSwinImageClassifierOutput(ModelOutput):
    """
    Swin outputs for image classification.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None

def window_partition(input_feature: tf.Tensor, window_size: int) -> tf.Tensor:
    if False:
        for i in range(10):
            print('nop')
    '\n    Partitions the given input into windows.\n    '
    (batch_size, height, width, num_channels) = shape_list(input_feature)
    input_feature = tf.reshape(input_feature, (batch_size, height // window_size, window_size, width // window_size, window_size, num_channels))
    windows = tf.transpose(input_feature, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(windows, (-1, window_size, window_size, num_channels))
    return windows

def window_reverse(windows: tf.Tensor, window_size: int, height: int, width: int) -> tf.Tensor:
    if False:
        while True:
            i = 10
    '\n    Merges windows to produce higher resolution features.\n    '
    x = tf.shape(windows)[0]
    y = tf.cast(height * width / (window_size * window_size), tf.int32)
    batch_size = tf.math.floordiv(x, y)
    windows = tf.reshape(windows, (batch_size, height // window_size, width // window_size, window_size, window_size, -1))
    windows = tf.transpose(windows, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(windows, (batch_size, height, width, -1))
    return windows

def drop_path(input: tf.Tensor, drop_prob: float=0.0, training: bool=False, scale_by_keep: bool=True) -> tf.Tensor:
    if False:
        while True:
            i = 10
    '\n    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).\n    '
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    input_shape = shape_list(input)
    ndim = len(input_shape)
    shape = [input_shape[0]] + [1] * (ndim - 1)
    random_tensor = tf.random.uniform(shape)
    random_tensor = tf.where(random_tensor <= keep_prob, 1.0, 0.0)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob
    return input * random_tensor

class TFSwinEmbeddings(tf.keras.layers.Layer):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SwinConfig, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.patch_embeddings = TFSwinPatchEmbeddings(config, name='patch_embeddings')
        self.num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.embed_dim = config.embed_dim
        self.use_mask_token = use_mask_token
        self.use_absolute_embeddings = config.use_absolute_embeddings
        self.norm = tf.keras.layers.LayerNormalization(name='norm', epsilon=1e-05)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')

    def build(self, input_shape: tf.TensorShape) -> None:
        if False:
            print('Hello World!')
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer='zeros', name='mask_token')
        else:
            self.mask_token = None
        if self.use_absolute_embeddings:
            self.position_embeddings = self.add_weight((1, self.num_patches + 1, self.embed_dim), initializer='zeros', name='positional_embeddings')
        else:
            self.position_embeddings = None
        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: bool=None, training: bool=False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        if False:
            while True:
                i = 10
        (embeddings, output_dimensions) = self.patch_embeddings(pixel_values, training=training)
        embeddings = self.norm(embeddings, training=training)
        (batch_size, seq_len, _) = shape_list(embeddings)
        if bool_masked_pos is not None:
            mask_tokens = tf.repeat(self.mask_token, batch_size, 0)
            mask_tokens = tf.repeat(mask_tokens, seq_len, 1)
            mask = tf.expand_dims(bool_masked_pos, -1)
            mask = tf.cast(mask, mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return (embeddings, output_dimensions)

class TFSwinPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        (image_size, patch_size) = (config.image_size, config.patch_size)
        (num_channels, hidden_size) = (config.num_channels, config.embed_dim)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.projection = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=self.patch_size, strides=self.patch_size, padding='valid', name='projection')

    def maybe_pad(self, pixel_values: tf.Tensor, height: int, width: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        if width % self.patch_size[1] != 0:
            pad_values = ((0, 0), (0, 0), (0, 0), (0, self.patch_size[1] - width % self.patch_size[1]))
            pixel_values = tf.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = ((0, 0), (0, 0), (0, self.patch_size[0] - height % self.patch_size[0]), (0, 0))
            pixel_values = tf.pad(pixel_values, pad_values)
        return pixel_values

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        if False:
            for i in range(10):
                print('nop')
        (_, num_channels, height, width) = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = self.maybe_pad(pixel_values, height, width)
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))
        embeddings = self.projection(pixel_values, training=training)
        embeddings = tf.transpose(embeddings, (0, 3, 1, 2))
        (batch_size, channels, height, width) = shape_list(embeddings)
        output_dimensions = (height, width)
        embeddings = tf.reshape(embeddings, (batch_size, channels, -1))
        embeddings = tf.transpose(embeddings, (0, 2, 1))
        return (embeddings, output_dimensions)

class TFSwinPatchMerging(tf.keras.layers.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`tf.keras.layer.Layer`, *optional*, defaults to `tf.keras.layers.LayerNormalization`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: Optional[Callable]=None, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False, name='reduction')
        if norm_layer is None:
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='norm')
        else:
            self.norm = norm_layer(name='norm')

    def maybe_pad(self, input_feature: tf.Tensor, height: int, width: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        should_pad = height % 2 == 1 or width % 2 == 1
        if should_pad:
            pad_values = ((0, 0), (0, height % 2), (0, width % 2), (0, 0))
            input_feature = tf.pad(input_feature, pad_values)
        return input_feature

    def call(self, input_feature: tf.Tensor, input_dimensions: Tuple[int, int], training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        (height, width) = input_dimensions
        (batch_size, _, num_channels) = shape_list(input_feature)
        input_feature = tf.reshape(input_feature, (batch_size, height, width, num_channels))
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = tf.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = tf.reshape(input_feature, (batch_size, -1, 4 * num_channels))
        input_feature = self.norm(input_feature, training=training)
        input_feature = self.reduction(input_feature, training=training)
        return input_feature

class TFSwinDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float=None, scale_by_keep: bool=True, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(TFSwinDropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, input: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            while True:
                i = 10
        return drop_path(input, self.drop_prob, training, self.scale_by_keep)

class TFSwinSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f'The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})')
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        window_size = config.window_size
        self.window_size = window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        self.query = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='value')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def build(self, input_shape: tf.TensorShape) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.relative_position_bias_table = self.add_weight(shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_attention_heads), initializer='zeros', name='relative_position_bias_table')
        self.relative_position_index = self.add_weight(shape=(self.window_size[0] ** 2, self.window_size[1] ** 2), trainable=False, dtype=tf.int32, name='relative_position_index')
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = tf.reshape(coords, (shape_list(coords)[0], -1))
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))
        (stack_0, stack_1) = tf.unstack(relative_coords, axis=2)
        stack_0 += self.window_size[0] - 1
        stack_0 *= 2 * self.window_size[1] - 1
        stack_1 += self.window_size[1] - 1
        relative_coords = tf.stack([stack_0, stack_1], axis=2)
        self.relative_position_index.assign(tf.cast(tf.reduce_sum(relative_coords, axis=-1), tf.int32))
        super().build(input_shape)

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor, ...]:
        if False:
            for i in range(10):
                print('nop')
        (batch_size, dim, _) = shape_list(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1,)))
        relative_position_bias = tf.reshape(relative_position_bias, (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attention_scores = attention_scores + tf.expand_dims(relative_position_bias, 0)
        if attention_mask is not None:
            mask_shape = shape_list(attention_mask)[0]
            attention_scores = tf.reshape(attention_scores, (batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim))
            attention_mask = tf.expand_dims(attention_mask, 1)
            attention_mask = tf.expand_dims(attention_mask, 0)
            attention_scores = attention_scores + attention_mask
            attention_scores = tf.reshape(attention_scores, (-1, self.num_attention_heads, dim, dim))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class TFSwinSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(dim, name='dense')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob, name='dropout')

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class TFSwinAttention(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.self = TFSwinSelfAttention(config, dim, num_heads, name='self')
        self.self_output = TFSwinSelfOutput(config, dim, name='output')
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. See base class PreTrainedModel heads: dict of {layer_num: list of heads to prune in\n        this layer}\n        '
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFSwinIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(int(config.mlp_ratio * dim), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFSwinOutput(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(dim, name='dense')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, 'dropout')

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class TFSwinLayer(tf.keras.layers.Layer):

    def __init__(self, config, dim, input_resolution: Tuple[int, int], num_heads: int, shift_size: int=0, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        min_res = tf.reduce_min(input_resolution)
        self.window_size = min_res if min_res <= config.window_size else config.window_size
        self.shift_size = 0 if min_res <= self.window_size else shift_size
        self.input_resolution = input_resolution
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.attention = TFSwinAttention(config, dim, num_heads, name='attention')
        self.drop_path = TFSwinDropPath(config.drop_path_rate, name='drop_path') if config.drop_path_rate > 0.0 else tf.keras.layers.Activation('linear', name='drop_path')
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')
        self.intermediate = TFSwinIntermediate(config, dim, name='intermediate')
        self.swin_output = TFSwinOutput(config, dim, name='output')

    def get_attn_mask(self, height: int, width: int, window_size: int, shift_size: int) -> tf.Tensor | None:
        if False:
            return 10
        img_mask = tf.zeros((height, width))
        height_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        width_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        if shift_size > 0:
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    height_inds = tf.range(height_slice[0] % height, height_slice[1] % height + 1)
                    width_inds = tf.range(width_slice[0] % width, width_slice[1] % width + 1)
                    indices = tf.reshape(tf.stack(tf.meshgrid(height_inds, width_inds), axis=-1), (-1, 2))
                    if len(indices) >= 1:
                        updates = tf.ones((len(indices),), dtype=img_mask.dtype) * count
                        img_mask = tf.tensor_scatter_nd_update(img_mask, indices, updates)
                    count += 1
        img_mask = tf.expand_dims(img_mask, -1)
        img_mask = tf.expand_dims(img_mask, 0)
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(mask_windows, (-1, window_size * window_size))
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
        attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        return attn_mask

    def maybe_pad(self, hidden_states: tf.Tensor, window_size: int, height: int, width: int) -> Tuple[tf.Tensor, tf.Tensor]:
        if False:
            i = 10
            return i + 15
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        pad_values = [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]]
        hidden_states = tf.pad(hidden_states, pad_values)
        pad_values = tf.reshape(pad_values, (-1,))
        return (hidden_states, pad_values)

    def call(self, hidden_states: tf.Tensor, input_dimensions: Tuple[int, int], head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        min_res = tf.reduce_min(input_dimensions)
        shift_size = 0 if min_res <= self.window_size else self.shift_size
        window_size = min_res if min_res <= self.window_size else self.window_size
        (height, width) = input_dimensions
        (batch_size, _, channels) = shape_list(hidden_states)
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states, training=training)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, channels))
        (hidden_states, pad_values) = self.maybe_pad(hidden_states, window_size, height, width)
        (_, height_pad, width_pad, _) = shape_list(hidden_states)
        if shift_size > 0:
            shifted_hidden_states = tf.roll(hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = tf.reshape(hidden_states_windows, (-1, window_size * window_size, channels))
        attn_mask = self.get_attn_mask(height=height_pad, width=width_pad, window_size=window_size, shift_size=shift_size)
        attention_outputs = self.attention(hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        attention_windows = tf.reshape(attention_output, (-1, window_size, window_size, channels))
        shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad)
        if shift_size > 0:
            attention_windows = tf.roll(shifted_windows, shift=(shift_size, shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]
        attention_windows = tf.reshape(attention_windows, (batch_size, height * width, channels))
        hidden_states = shortcut + self.drop_path(attention_windows, training=training)
        layer_output = self.layernorm_after(hidden_states, training=training)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.swin_output(layer_output, training=training)
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

class TFSwinStage(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int, drop_path: List[float], downsample: Optional[Callable], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.dim = dim
        self.blocks = [TFSwinLayer(config=config, dim=dim, input_resolution=input_resolution, num_heads=num_heads, shift_size=0 if i % 2 == 0 else config.window_size // 2, name=f'blocks.{i}') for i in range(depth)]
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-05), name='downsample')
        else:
            self.downsample = None
        self.pointing = False

    def call(self, hidden_states: tf.Tensor, input_dimensions: Tuple[int, int], head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor, ...]:
        if False:
            i = 10
            return i + 15
        (height, width) = input_dimensions
        for (i, layer_module) in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training)
            hidden_states = layer_outputs[0]
        if self.downsample is not None:
            (height_downsampled, width_downsampled) = ((height + 1) // 2, (width + 1) // 2)
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(layer_outputs[0], input_dimensions, training=training)
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (hidden_states, output_dimensions)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs

class TFSwinEncoder(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, grid_size: Tuple[int, int], **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.num_layers = len(config.depths)
        self.config = config
        dpr = list((tf.linspace(0, 1, sum(config.depths)) * config.drop_path_rate).numpy())
        self.layers = [TFSwinStage(config=config, dim=int(config.embed_dim * 2 ** i_layer), input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer), depth=config.depths[i_layer], num_heads=config.num_heads[i_layer], drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=TFSwinPatchMerging if i_layer < self.num_layers - 1 else None, name=f'layers.{i_layer}') for i_layer in range(self.num_layers)]
        self.gradient_checkpointing = False

    def call(self, hidden_states: tf.Tensor, input_dimensions: Tuple[int, int], head_mask: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> Union[Tuple[tf.Tensor, ...], TFSwinEncoderOutput]:
        if False:
            while True:
                i = 10
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            (batch_size, _, hidden_size) = shape_list(hidden_states)
            reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
            reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for (i, layer_module) in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training)
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                (batch_size, _, hidden_size) = shape_list(hidden_states)
                reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
                reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[2:]
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFSwinEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, reshaped_hidden_states=all_reshaped_hidden_states)

class TFSwinPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SwinConfig
    base_model_prefix = 'swin'
    main_input_name = 'pixel_values'
SWIN_START_DOCSTRING = '\n    This model is a Tensorflow\n    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a\n    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`SwinConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
SWIN_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]\n            for details.\n        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

def normalize_data_format(value: str) -> str:
    if False:
        return 10
    '\n    From tensorflow addons\n    https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/utils/keras_utils.py#L71\n    '
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + str(value))
    return data_format

class AdaptiveAveragePooling1D(tf.keras.layers.Layer):
    """
    Args:
    Average 1D Pooling with adaptive kernel size.
      output_size: An integer or tuple/list of a single integer, specifying pooled_features.
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, steps, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, steps)`.
    Input shape:
      - If `data_format='channels_last'`: 3D tensor with shape `(batch, steps, channels)`.
      - If `data_format='channels_first'`: 3D tensor with shape `(batch, channels, steps)`.
    Output shape:
      - If `data_format='channels_last'`: 3D tensor with shape `(batch_size, pooled_steps, channels)`.
      - If `data_format='channels_first'`: 3D tensor with shape `(batch_size, channels, pooled_steps)`.

    Adapted from [tensorflow-addon's adaptive pooling.py](
        https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/layers/adaptive_pooling.py#L90-L120
    )
    """

    def __init__(self, output_size: Union[int, Iterable[int]], reduce_function: Callable=tf.reduce_mean, data_format: Optional[str]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self.data_format = normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = (output_size,) if isinstance(output_size, int) else tuple(output_size)
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, *args) -> None:
        if False:
            print('Hello World!')
        bins = self.output_size[0]
        if self.data_format == 'channels_last':
            splits = tf.split(inputs, bins, axis=1)
            splits = tf.stack(splits, axis=1)
            out_vect = self.reduce_function(splits, axis=2)
        else:
            splits = tf.split(inputs, bins, axis=2)
            splits = tf.stack(splits, axis=2)
            out_vect = self.reduce_function(splits, axis=3)
        return out_vect

    def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape:
        if False:
            return 10
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            shape = tf.TensorShape([input_shape[0], self.output_size[0], input_shape[2]])
        else:
            shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size[0]])
        return shape

    def get_config(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        config = {'output_size': self.output_size, 'data_format': self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}

@keras_serializable
class TFSwinMainLayer(tf.keras.layers.Layer):
    config_class = SwinConfig

    def __init__(self, config: SwinConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = TFSwinEmbeddings(config, use_mask_token=use_mask_token, name='embeddings')
        self.encoder = TFSwinEncoder(config, self.embeddings.patch_grid, name='encoder')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = AdaptiveAveragePooling1D(output_size=(1,)) if add_pooling_layer else None

    def get_input_embeddings(self) -> TFSwinPatchEmbeddings:
        if False:
            i = 10
            return i + 15
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List]):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_head_mask(self, head_mask: Optional[Any]) -> List:
        if False:
            i = 10
            return i + 15
        if head_mask is not None:
            raise NotImplementedError
        return [None] * len(self.config.depths)

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFSwinModelOutput, Tuple[tf.Tensor, ...]]:
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask)
        (embedding_output, input_dimensions) = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)
        encoder_outputs = self.encoder(embedding_output, input_dimensions, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        pooled_output = None
        if self.pooler is not None:
            (batch_size, _, num_features) = shape_list(sequence_output)
            pooled_output = self.pooler(sequence_output)
            pooled_output = tf.reshape(pooled_output, (batch_size, num_features))
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output
        return TFSwinModelOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, reshaped_hidden_states=encoder_outputs.reshaped_hidden_states)

@add_start_docstrings('The bare Swin Model transformer outputting raw hidden-states without any specific head on top.', SWIN_START_DOCSTRING)
class TFSwinModel(TFSwinPreTrainedModel):

    def __init__(self, config: SwinConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, **kwargs)
        self.config = config
        self.swin = TFSwinMainLayer(config, name='swin')

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSwinModelOutput, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFSwinModelOutput, Tuple[tf.Tensor, ...]]:
        if False:
            return 10
        "\n        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        swin_outputs = self.swin(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return swin_outputs

class TFSwinPixelShuffle(tf.keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f'upscale_factor must be an integer value >= 2 got {upscale_factor}')
        self.upscale_factor = upscale_factor

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = x
        (batch_size, _, _, num_input_channels) = shape_list(hidden_states)
        block_size_squared = self.upscale_factor ** 2
        output_depth = int(num_input_channels / block_size_squared)
        permutation = tf.constant([[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]])
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format='NHWC')
        return hidden_states

class TFSwinDecoder(tf.keras.layers.Layer):

    def __init__(self, config: SwinConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(filters=config.encoder_stride ** 2 * config.num_channels, kernel_size=1, strides=1, name='0')
        self.pixel_shuffle = TFSwinPixelShuffle(config.encoder_stride, name='1')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = x
        hidden_states = tf.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv2d(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = tf.transpose(hidden_states, (0, 3, 1, 2))
        return hidden_states

@add_start_docstrings('Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).', SWIN_START_DOCSTRING)
class TFSwinForMaskedImageModeling(TFSwinPreTrainedModel):

    def __init__(self, config: SwinConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.swin = TFSwinMainLayer(config, add_pooling_layer=False, use_mask_token=True, name='swin')
        self.decoder = TFSwinDecoder(config, name='decoder')

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFSwinMaskedImageModelingOutput]:
        if False:
            print('Hello World!')
        '\n        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren\'t (0).\n\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoImageProcessor, TFSwinForMaskedImageModeling\n        >>> import tensorflow as tf\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")\n        >>> model = TFSwinForMaskedImageModeling.from_pretrained("microsoft/swin-tiny-patch4-window7-224")\n\n        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2\n        >>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values\n        >>> # create random boolean mask of shape (batch_size, num_patches)\n        >>> bool_masked_pos = tf.random.uniform((1, num_patches)) >= 0.5\n\n        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)\n        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction\n        >>> list(reconstructed_pixel_values.shape)\n        [1, 3, 224, 224]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swin(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = tf.transpose(sequence_output, (0, 2, 1))
        (batch_size, num_channels, sequence_length) = shape_list(sequence_output)
        height = width = int(sequence_length ** 0.5)
        sequence_output = tf.reshape(sequence_output, (batch_size, num_channels, height, width))
        reconstructed_pixel_values = self.decoder(sequence_output)
        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = tf.reshape(bool_masked_pos, (-1, size, size))
            mask = tf.repeat(bool_masked_pos, self.config.patch_size, 1)
            mask = tf.repeat(mask, self.config.patch_size, 2)
            mask = tf.expand_dims(mask, 1)
            mask = tf.cast(mask, tf.float32)
            reconstruction_loss = tf.keras.losses.mean_absolute_error(tf.transpose(pixel_values, (1, 2, 3, 0)), tf.transpose(reconstructed_pixel_values, (1, 2, 3, 0)))
            reconstruction_loss = tf.expand_dims(reconstruction_loss, 0)
            total_loss = tf.reduce_sum(reconstruction_loss * mask)
            num_masked_pixels = (tf.reduce_sum(mask) + 1e-05) * self.config.num_channels
            masked_im_loss = total_loss / num_masked_pixels
            masked_im_loss = tf.reshape(masked_im_loss, (1,))
        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]
            return (masked_im_loss,) + output if masked_im_loss is not None else output
        return TFSwinMaskedImageModelingOutput(loss=masked_im_loss, reconstruction=reconstructed_pixel_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, reshaped_hidden_states=outputs.reshaped_hidden_states)

@add_start_docstrings('\n    Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', SWIN_START_DOCSTRING)
class TFSwinForImageClassification(TFSwinPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: SwinConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.swin = TFSwinMainLayer(config, name='swin')
        self.classifier = tf.keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else tf.keras.layers.Activation('linear', name='classifier')

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSwinImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor, ...], TFSwinImageClassifierOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swin(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSwinImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, reshaped_hidden_states=outputs.reshaped_hidden_states)