""" TF 2.0 ViT MAE (masked autoencoder) model."""
from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'ViTMAEConfig'
_CHECKPOINT_FOR_DOC = 'facebook/vit-mae-base'

@dataclass
class TFViTMAEModelOutput(ModelOutput):
    """
    Class for TFViTMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    last_hidden_state: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFViTMAEDecoderOutput(ModelOutput):
    """
    Class for TFViTMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFViTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for TFViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`tf.Tensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    if False:
        i = 10
        return i + 15
    '\n    Create 2D sin/cos positional embeddings.\n\n    Args:\n        embed_dim (`int`):\n            Embedding dimension.\n        grid_size (`int`):\n            The grid height and width.\n        add_cls_token (`bool`, *optional*, defaults to `False`):\n            Whether or not to add a classification (CLS) token.\n\n    Returns:\n        (`tf.Tensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the position\n        embeddings (with or without classification token)\n    '
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid = tf.meshgrid(grid_w, grid_h)
    grid = tf.stack(grid, axis=0)
    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if False:
        i = 10
        return i + 15
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even')
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = tf.concat([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if False:
        print('Hello World!')
    '\n    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)\n    '
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even')
    omega = tf.range(embed_dim // 2, dtype='float32')
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = tf.reshape(pos, [-1])
    out = tf.einsum('m,d->md', pos, omega)
    emb_sin = tf.sin(out)
    emb_cos = tf.cos(out)
    emb = tf.concat([emb_sin, emb_cos], axis=1)
    return emb

class TFViTMAEEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.patch_embeddings = TFViTMAEPatchEmbeddings(config, name='patch_embeddings')
        self.num_patches = self.patch_embeddings.num_patches
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        if False:
            while True:
                i = 10
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='cls_token')
        self.position_embeddings = self.add_weight(shape=(1, self.num_patches + 1, self.config.hidden_size), initializer='zeros', trainable=False, name='position_embeddings')
        pos_embed = get_2d_sincos_pos_embed(self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches ** 0.5), add_cls_token=True)[None, ...]
        self.position_embeddings.assign(pos_embed)
        super().build(input_shape)

    def random_masking(self, sequence: tf.Tensor, noise: tf.Tensor | None=None):
        if False:
            while True:
                i = 10
        '\n        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random\n        noise.\n\n        Args:\n            sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)\n            noise (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*) which is\n                mainly used for testing purposes to control randomness and maintain the reproducibility\n        '
        (batch_size, seq_length, dim) = shape_list(sequence)
        len_keep = int(seq_length * (1 - self.config.mask_ratio))
        if noise is None:
            noise = tf.random.uniform(shape=(batch_size, seq_length), minval=0.0, maxval=1.0)
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = tf.gather(sequence, axis=1, batch_dims=1, indices=ids_keep)
        mask_keep = tf.zeros((batch_size, len_keep))
        mask_remove = tf.ones((batch_size, seq_length - len_keep))
        mask = tf.concat([mask_keep, mask_remove], axis=-1)
        mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)
        return (sequence_unmasked, mask, ids_restore)

    def call(self, pixel_values: tf.Tensor, noise: tf.Tensor=None) -> tf.Tensor:
        if False:
            print('Hello World!')
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
        (embeddings, mask, ids_restore) = self.random_masking(embeddings, noise)
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        return (embeddings, mask, ids_restore)

class TFViTMAEPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        (image_size, patch_size) = (config.image_size, config.patch_size)
        (num_channels, hidden_size) = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config
        self.projection = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid', data_format='channels_last', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='projection')

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        (batch_size, num_channels, height, width) = shape_list(pixel_values)
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        projection = self.projection(pixel_values)
        num_patches = width // self.patch_size[1] * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))
        return x

class TFViTMAESelfAttention(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=attention_probs, training=training)
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

class TFViTMAESelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        return hidden_states

class TFViTMAEAttention(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.self_attention = TFViTMAESelfAttention(config, name='attention')
        self.dense_output = TFViTMAESelfOutput(config, name='output')

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            print('Hello World!')
        self_outputs = self.self_attention(hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFViTMAEIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFViTMAEOutput(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class TFViTMAELayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.attention = TFViTMAEAttention(config, name='attention')
        self.intermediate = TFViTMAEIntermediate(config, name='intermediate')
        self.vit_output = TFViTMAEOutput(config, name='output')
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
        attention_outputs = self.attention(input_tensor=self.layernorm_before(inputs=hidden_states), head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(inputs=hidden_states)
        intermediate_output = self.intermediate(hidden_states=layer_output)
        layer_output = self.vit_output(hidden_states=intermediate_output, input_tensor=hidden_states, training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class TFViTMAEEncoder(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.layer = [TFViTMAELayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states=hidden_states, head_mask=head_mask[i], output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

@keras_serializable
class TFViTMAEMainLayer(tf.keras.layers.Layer):
    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFViTMAEEmbeddings(config, name='embeddings')
        self.encoder = TFViTMAEEncoder(config, name='encoder')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            return 10
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, noise: tf.Tensor=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        (embedding_output, mask, ids_restore) = self.embeddings(pixel_values=pixel_values, training=training, noise=noise)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]
        return TFViTMAEModelOutput(last_hidden_state=sequence_output, mask=mask, ids_restore=ids_restore, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class TFViTMAEPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ViTMAEConfig
    base_model_prefix = 'vit'
    main_input_name = 'pixel_values'
VIT_MAE_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`ViTMAEConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
VIT_MAE_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]\n            for details.\n\n        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used\n            in eager mode, in graph mode the value will always be set to True.\n\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'

@add_start_docstrings('The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top.', VIT_MAE_START_DOCSTRING)
class TFViTMAEModel(TFViTMAEPreTrainedModel):

    def __init__(self, config: ViTMAEConfig, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.vit = TFViTMAEMainLayer(config, name='vit')

    def get_input_embeddings(self):
        if False:
            return 10
        return self.vit.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: TFModelInputType | None=None, noise: tf.Tensor=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFViTMAEModel\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")\n        >>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")\n\n        >>> inputs = image_processor(images=image, return_tensors="tf")\n        >>> outputs = model(**inputs)\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        outputs = self.vit(pixel_values=pixel_values, noise=noise, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFViTMAEDecoder(tf.keras.layers.Layer):

    def __init__(self, config, num_patches, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.decoder_embed = tf.keras.layers.Dense(config.decoder_hidden_size, name='decoder_embed')
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = [TFViTMAELayer(decoder_config, name=f'decoder_layers.{j}') for j in range(config.decoder_num_hidden_layers)]
        self.decoder_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='decoder_norm')
        self.decoder_pred = tf.keras.layers.Dense(config.patch_size ** 2 * config.num_channels, kernel_initializer=get_initializer(config.initializer_range), name='decoder_pred')
        self.config = config
        self.num_patches = num_patches

    def build(self, input_shape: tf.TensorShape):
        if False:
            while True:
                i = 10
        self.mask_token = self.add_weight(shape=(1, 1, self.config.decoder_hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='mask_token')
        self.decoder_pos_embed = self.add_weight(shape=(1, self.num_patches + 1, self.config.decoder_hidden_size), initializer='zeros', trainable=False, name='decoder_pos_embed')
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** 0.5), add_cls_token=True)[None, ...]
        self.decoder_pos_embed.assign(decoder_pos_embed)
        super().build(input_shape)

    def call(self, hidden_states, ids_restore, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            while True:
                i = 10
        x = self.decoder_embed(hidden_states)
        mask_tokens = tf.tile(self.mask_token, (shape_list(x)[0], shape_list(ids_restore)[1] + 1 - shape_list(x)[1], 1))
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)
        x_ = tf.gather(x_, axis=1, batch_dims=1, indices=ids_restore)
        x = tf.concat([x[:, :1, :], x_], axis=1)
        hidden_states = x + self.decoder_pos_embed
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]
        if not return_dict:
            return tuple((v for v in [logits, all_hidden_states, all_self_attentions] if v is not None))
        return TFViTMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)

@add_start_docstrings('The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.', VIT_MAE_START_DOCSTRING)
class TFViTMAEForPreTraining(TFViTMAEPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.config = config
        self.vit = TFViTMAEMainLayer(config, name='vit')
        self.decoder = TFViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches, name='decoder')

    def get_input_embeddings(self):
        if False:
            return 10
        return self.vit.get_input_embeddings()

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def patchify(self, pixel_values):
        if False:
            return 10
        '\n        Args:\n            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)` or `(batch_size, num_channels, height, width)`):\n                Pixel values.\n\n        Returns:\n            `tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:\n                Patchified pixel values.\n        '
        (patch_size, num_channels) = (self.config.patch_size, self.config.num_channels)
        if shape_list(pixel_values)[1] == num_channels:
            pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        tf.debugging.assert_equal(shape_list(pixel_values)[1], shape_list(pixel_values)[2], message='Make sure the pixel values have a squared size')
        tf.debugging.assert_equal(shape_list(pixel_values)[1] % patch_size, 0, message='Make sure the pixel values have a size that is divisible by the patch size')
        tf.debugging.assert_equal(shape_list(pixel_values)[3], num_channels, message='Make sure the number of channels of the pixel values is equal to the one set in the configuration')
        batch_size = shape_list(pixel_values)[0]
        num_patches_one_direction = shape_list(pixel_values)[2] // patch_size
        patchified_pixel_values = tf.reshape(pixel_values, (batch_size, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size, num_channels))
        patchified_pixel_values = tf.einsum('nhpwqc->nhwpqc', patchified_pixel_values)
        patchified_pixel_values = tf.reshape(patchified_pixel_values, (batch_size, num_patches_one_direction * num_patches_one_direction, patch_size ** 2 * num_channels))
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        if False:
            return 10
        '\n        Args:\n            patchified_pixel_values (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:\n                Patchified pixel values.\n\n        Returns:\n            `tf.Tensor` of shape `(batch_size, height, width, num_channels)`:\n                Pixel values.\n        '
        (patch_size, num_channels) = (self.config.patch_size, self.config.num_channels)
        num_patches_one_direction = int(shape_list(patchified_pixel_values)[1] ** 0.5)
        tf.debugging.assert_equal(num_patches_one_direction * num_patches_one_direction, shape_list(patchified_pixel_values)[1], message='Make sure that the number of patches can be squared')
        batch_size = shape_list(patchified_pixel_values)[0]
        patchified_pixel_values = tf.reshape(patchified_pixel_values, (batch_size, num_patches_one_direction, num_patches_one_direction, patch_size, patch_size, num_channels))
        patchified_pixel_values = tf.einsum('nhwpqc->nhpwqc', patchified_pixel_values)
        pixel_values = tf.reshape(patchified_pixel_values, (batch_size, num_patches_one_direction * patch_size, num_patches_one_direction * patch_size, num_channels))
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)`):\n                Pixel values.\n            pred (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:\n                Predicted pixel values.\n            mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n                Tensor indicating which patches are masked (1) and which are not (0).\n\n        Returns:\n            `tf.Tensor`: Pixel reconstruction loss.\n        '
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = tf.reduce_mean(target, axis=-1, keepdims=True)
            var = tf.math.reduce_variance(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, axis=-1)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        loss = tf.reshape(loss, (1,))
        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: TFModelInputType | None=None, noise: tf.Tensor=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFViTMAEForPreTrainingOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFViTMAEForPreTraining\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")\n        >>> model = TFViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> loss = outputs.loss\n        >>> mask = outputs.mask\n        >>> ids_restore = outputs.ids_restore\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(pixel_values=pixel_values, noise=noise, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits
        loss = self.forward_loss(pixel_values, logits, mask)
        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFViTMAEForPreTrainingOutput(loss=loss, logits=logits, mask=mask, ids_restore=ids_restore, hidden_states=outputs.hidden_states, attentions=outputs.attentions)