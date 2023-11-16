""" TF 2.0 GroupViT model."""
from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_tensorflow_probability_available, logging, replace_return_docstrings
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
logger = logging.get_logger(__name__)
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        logger.error("GroupViT models are not usable since `tensorflow_probability` can't be loaded. It seems you have `tensorflow_probability` installed with the wrong tensorflow version.Please try to reinstall it following the instructions here: https://github.com/tensorflow/probability.")
_CHECKPOINT_FOR_DOC = 'nvidia/groupvit-gcc-yfcc'
TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['nvidia/groupvit-gcc-yfcc']
LARGE_NEGATIVE = -100000000.0

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

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    if False:
        i = 10
        return i + 15
    return tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True))

def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    if False:
        return 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor:
    if False:
        for i in range(10):
            print('nop')
    y_soft = stable_softmax(logits, dim)
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(index, depth=shape_list(logits)[dim], axis=range(len(shape_list(logits)))[dim], dtype=y_soft.dtype)
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    return ret

def gumbel_softmax(logits: tf.Tensor, tau: float=1, hard: bool=False, dim: int=-1) -> tf.Tensor:
    if False:
        print('Hello World!')
    gumbel_dist = tfp.distributions.Gumbel(0.0, 1.0)
    gumbels = gumbel_dist.sample(tf.shape(logits), dtype=logits.dtype)
    gumbels = (logits + gumbels) / tau
    y_soft = stable_softmax(gumbels, dim)
    if hard:
        index = tf.argmax(y_soft, dim)
        y_hard = tf.one_hot(index, depth=shape_list(logits)[dim], axis=range(len(shape_list(logits)))[dim], dtype=y_soft.dtype)
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        ret = y_soft
    return ret

def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: bool=False) -> tf.Tensor:
    if False:
        print('Hello World!')
    '\n    Args:\n        attentions (`tf.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]\n        height (`int`): height of the output attention map\n        width (`int`): width of the output attention map\n        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.\n\n    Returns:\n        `tf.Tensor`: resized attention map of shape [batch_size, groups, height, width]\n    '
    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height
    batch_size = shape_list(attentions)[0]
    groups = shape_list(attentions)[1]
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    if align_corners:
        attentions = tf.compat.v1.image.resize(attentions, size=(height, width), method='bilinear', align_corners=align_corners)
    else:
        attentions = tf.image.resize(attentions, size=(height, width), method='bilinear')
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    return attentions

def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: Tuple[int]) -> tf.Tensor:
    if False:
        while True:
            i = 10
    '\n    Args:\n        attentions (`tuple(tf.Tensor)`: tuple of attention maps returned by `TFGroupViTVisionTransformer`\n        hw_shape (`tuple(int)`): height and width of the output attention map\n    Returns:\n        `tf.Tensor`: the attention map of shape [batch_size, groups, height, width]\n    '
    attn_maps = []
    prev_attn_masks = None
    for attn_masks in attentions:
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = tf.matmul(prev_attn_masks, attn_masks)
        cur_attn_map = resize_attention_map(tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape)
        attn_maps.append(cur_attn_map)
    final_grouping = attn_maps[-1]
    return tf.stop_gradient(final_grouping)

@dataclass
class TFGroupViTModelOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTTextModel`].
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTVisionModel`].
        text_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTTextModel`].
        vision_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTVisionModel`].
    """
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    segmentation_logits: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class TFGroupViTCrossAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.attn = TFGroupViTAttention(config, name='attn')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm2')
        self.mlp = TFGroupViTMLP(config, name='mlp')
        self.norm_post = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_post')

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        x = query
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        x = x + self.mlp(self.norm2(x))
        x = self.norm_post(x)
        return x

class TFGroupViTAssignAttention(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.scale = config.hidden_size ** (-0.5)
        self.q_proj = tf.keras.layers.Dense(config.hidden_size, name='q_proj')
        self.k_proj = tf.keras.layers.Dense(config.hidden_size, name='k_proj')
        self.v_proj = tf.keras.layers.Dense(config.hidden_size, name='v_proj')
        self.proj = tf.keras.layers.Dense(config.hidden_size, name='proj')
        self.assign_eps = config.assign_eps

    def get_attn(self, attn: tf.Tensor, gumbel: bool=True, hard: bool=True, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        if gumbel and training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        elif hard:
            attn = hard_softmax(attn, dim=-2)
        else:
            attn = stable_softmax(attn, axis=-2)
        return attn

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool=False):
        if False:
            while True:
                i = 10
        value = key
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        raw_attn = tf.matmul(query, key, transpose_b=True) * self.scale
        attn = self.get_attn(raw_attn, training=training)
        soft_attn = self.get_attn(raw_attn, training=training, gumbel=False, hard=False)
        attn = attn / (tf.math.reduce_sum(attn, axis=-1, keepdims=True) + self.assign_eps)
        out = tf.matmul(attn, value)
        out = self.proj(out)
        return (out, soft_attn)

class TFGroupViTTokenAssign(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, num_group_token: int, num_output_group: int, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.num_output_group = num_output_group
        self.norm_tokens = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_tokens')
        assign_mlp_ratio = config.assign_mlp_ratio if isinstance(config.assign_mlp_ratio, collections.abc.Iterable) else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        (tokens_dim, channels_dim) = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        self.mlp_inter = TFGroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group, name='mlp_inter')
        self.norm_post_tokens = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_post_tokens')
        self.norm_x = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_x')
        self.pre_assign_attn = TFGroupViTCrossAttentionLayer(config, name='pre_assign_attn')
        self.assign = TFGroupViTAssignAttention(config, name='assign')
        self.norm_new_x = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_new_x')
        self.mlp_channels = TFGroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size, name='mlp_channels')

    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            group_tokens (tf.Tensor): group tokens, [batch_size, num_group_tokens, channels]\n\n        Returns:\n            projected_group_tokens (tf.Tensor): [batch_size, num_output_groups, channels]\n        '
        projected_group_tokens = self.mlp_inter(group_tokens)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def call(self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            image_tokens (`tf.Tensor`): image tokens, of shape [batch_size, input_length, channels]\n            group_tokens (`tf.Tensor`): group tokens, [batch_size, num_group_tokens, channels]\n        '
        group_tokens = self.norm_tokens(group_tokens)
        image_tokens = self.norm_x(image_tokens)
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        (new_image_tokens, attention) = self.assign(projected_group_tokens, image_tokens)
        new_image_tokens += projected_group_tokens
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))
        return (new_image_tokens, attention)

class TFGroupViTPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: GroupViTConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        (image_size, patch_size) = (config.image_size, config.patch_size)
        num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config
        self.projection = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid', data_format='channels_last', use_bias=True, kernel_initializer=get_initializer(self.config.initializer_range), bias_initializer='zeros', name='projection')

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool=False, training: bool=False) -> tf.Tensor:
        if False:
            while True:
                i = 10
        (batch_size, num_channels, height, width) = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if not interpolate_pos_encoding and tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        projection = self.projection(pixel_values)
        num_patches = width // self.patch_size[1] * (height // self.patch_size[0])
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, self.hidden_size))
        return embeddings

class TFGroupViTVisionEmbeddings(tf.keras.layers.Layer):
    """
    Construct the position and patch embeddings.

    """

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.patch_embeddings = TFGroupViTPatchEmbeddings(config, name='patch_embeddings')
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout, name='dropout')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        if False:
            return 10
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(shape=(1, num_patches, self.config.hidden_size), initializer='zeros', trainable=True, name='position_embeddings')
        super().build(input_shape)

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher\n        resolution images.\n\n        Source:\n        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174\n        '
        (batch_size, num_patches, dim) = shape_list(embeddings)
        num_positions = shape_list(self.position_embeddings)[1]
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(images=tf.reshape(patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)), size=(h0, w0), method='bicubic')
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return patch_pos_embed

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool=False, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        (_, _, height, width) = shape_list(pixel_values)
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.layernorm(embeddings)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TFGroupViTTextEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.config = config

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            for i in range(10):
                print('nop')
        with tf.name_scope('token_embedding'):
            self.weight = self.add_weight(shape=(self.config.vocab_size, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='weight')
        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(shape=(self.config.max_position_embeddings, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='embeddings')
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Applies embedding based on inputs tensor.\n\n        Returns:\n            final_embeddings (`tf.Tensor`): output embedding tensor.\n        '
        if input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds
        return final_embeddings

class TFGroupViTStage(tf.keras.layers.Layer):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(self, config: GroupViTVisionConfig, depth: int, num_prev_group_token: int, num_group_token: int, num_output_group: int, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.depth = depth
        self.num_group_token = num_group_token
        self.layers = [TFGroupViTEncoderLayer(config, name=f'layers_._{i}') for i in range(depth)]
        if num_group_token > 0:
            self.downsample = TFGroupViTTokenAssign(config=config, num_group_token=num_group_token, num_output_group=num_output_group, name='downsample')
        else:
            self.downsample = None
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = [tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='group_projector.0'), TFGroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token, name='group_projector.1')]
        else:
            self.group_projector = None

    def build(self, input_shape: tf.TensorShape):
        if False:
            return 10
        if self.num_group_token > 0:
            self.group_token = self.add_weight(shape=(1, self.num_group_token, self.config.hidden_size), initializer='zeros', trainable=True, name='group_token')
        else:
            self.group_token = None
        super().build(input_shape)

    @property
    def with_group_token(self):
        if False:
            for i in range(10):
                print('nop')
        return self.group_token is not None

    def split_x(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        if self.with_group_token:
            return (x[:, :-self.num_group_token], x[:, -self.num_group_token:])
        else:
            return (x, None)

    def concat_x(self, x: tf.Tensor, group_token: tf.Tensor | None=None) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        if group_token is None:
            return x
        return tf.concat([x, group_token], axis=1)

    def call(self, hidden_states: tf.Tensor, prev_group_token: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            while True:
                i = 10
        '\n        Args:\n            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`tf.Tensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the grouping tensors of Grouping block.\n        '
        if self.with_group_token:
            group_token = tf.tile(self.group_token, multiples=(shape_list(hidden_states)[0], 1, 1))
            if self.group_projector is not None:
                for layer in self.group_projector:
                    prev_group_token = layer(prev_group_token)
                group_token = group_token + prev_group_token
        else:
            group_token = None
        x = hidden_states
        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None, output_attentions=None)
            cat_x = layer_out[0]
        (x, group_token) = self.split_x(cat_x)
        attention = None
        if self.downsample is not None:
            (x, attention) = self.downsample(x, group_token)
        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)
        return outputs

class TFGroupViTMLP(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, hidden_size: Optional[int]=None, intermediate_size: Optional[int]=None, output_size: Optional[int]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.activation_fn = get_tf_activation(config.hidden_act)
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        output_size = output_size if output_size is not None else hidden_size
        self.fc1 = tf.keras.layers.Dense(intermediate_size, name='fc1')
        self.fc2 = tf.keras.layers.Dense(output_size, name='fc2')

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TFGroupViTMixerMLP(TFGroupViTMLP):

    def call(self, x, training: bool=False):
        if False:
            i = 10
            return i + 15
        x = super().call(hidden_states=tf.transpose(x, perm=(0, 2, 1)))
        return tf.transpose(x, perm=(0, 2, 1))

class TFGroupViTAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GroupViTConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_attention_heads}).')
        factor = config.initializer_factor
        in_proj_std = self.embed_dim ** (-0.5) * (2 * config.num_hidden_layers) ** (-0.5) * factor
        out_proj_std = self.embed_dim ** (-0.5) * factor
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.q_proj = tf.keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='q_proj')
        self.k_proj = tf.keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='k_proj')
        self.v_proj = tf.keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='v_proj')
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)
        self.out_proj = tf.keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name='out_proj')

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor=None, causal_attention_mask: tf.Tensor=None, output_attentions: bool=None, encoder_hidden_states: tf.Tensor=None, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        batch_size = shape_list(hidden_states)[0]
        is_cross_attention = encoder_hidden_states is not None
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        if is_cross_attention:
            mixed_key_layer = self.k_proj(inputs=encoder_hidden_states)
            mixed_value_layer = self.v_proj(inputs=encoder_hidden_states)
        else:
            mixed_key_layer = self.k_proj(inputs=hidden_states)
            mixed_value_layer = self.v_proj(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        if causal_attention_mask is not None:
            attention_scores = tf.add(attention_scores, causal_attention_mask)
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=_attention_probs)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))
        attention_output = self.out_proj(attention_output)
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)
        return outputs

class TFGroupViTEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFGroupViTAttention(config, name='self_attn')
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm1')
        self.mlp = TFGroupViTMLP(config, name='mlp')
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm2')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, causal_attention_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`tf.Tensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            causal_attention_mask (`tf.Tensor`): causal attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            output_attentions (`bool`):\n                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned\n                tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(inputs=hidden_states)
        attention_outputs = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, training=training)
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(inputs=hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs

class TFGroupViTTextEncoder(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.layers = [TFGroupViTEncoderLayer(config, name=f'layers_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states, attention_mask: tf.Tensor, causal_attention_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        if False:
            print('Hello World!')
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class TFGroupViTVisionEncoder(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.stages = [TFGroupViTStage(config=config, depth=config.depths[i], num_group_token=config.num_group_tokens[i], num_output_group=config.num_output_groups[i], num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0, name=f'stages_._{i}') for i in range(len(config.depths))]

    def call(self, hidden_states: tf.Tensor, output_hidden_states: bool, output_attentions: bool, return_dict: bool, training: bool=False) -> Union[tuple, TFBaseModelOutput]:
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None
        group_tokens = None
        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)
            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings)

class TFGroupViTTextTransformer(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.embeddings = TFGroupViTTextEmbeddings(config, name='embeddings')
        self.encoder = TFGroupViTTextEncoder(config, name='encoder')
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='final_layer_norm')
        self.eos_token_id = config.eos_token_id

    def call(self, input_ids: TFModelInputType, attention_mask: tf.Tensor, position_ids: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        input_shape = shape_list(input_ids)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        (batch_size, seq_length) = input_shape
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)
        attention_mask = _expand_mask(attention_mask)
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layer_norm(inputs=sequence_output)
        if self.eos_token_id == 2:
            pooled_output = tf.gather_nd(params=sequence_output, indices=tf.stack(values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1))
        else:
            pooled_output = tf.gather_nd(params=sequence_output, indices=tf.stack(values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(tf.cast(input_ids == self.eos_token_id, dtype=tf.int8), axis=-1)), axis=1))
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        if False:
            print('Hello World!')
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)
        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

class TFGroupViTVisionTransformer(tf.keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.embeddings = TFGroupViTVisionEmbeddings(config, name='embeddings')
        self.encoder = TFGroupViTVisionEncoder(config, name='encoder')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')

    def call(self, pixel_values: TFModelInputType, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(hidden_states=embedding_output, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@keras_serializable
class TFGroupViTTextMainLayer(tf.keras.layers.Layer):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFGroupViTTextTransformer(config, name='text_model')

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            print('Hello World!')
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if False:
            for i in range(10):
                print('nop')
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        if input_ids is None:
            raise ValueError('You have to specify input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return text_model_outputs

@keras_serializable
class TFGroupViTVisionMainLayer(tf.keras.layers.Layer):
    config_class = GroupViTVisionConfig

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.vision_model = TFGroupViTVisionTransformer(config, name='vision_model')

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            return 10
        return self.vision_model.embeddings

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_model_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return vision_model_outputs

@keras_serializable
class TFGroupViTMainLayer(tf.keras.layers.Layer):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(f'config.text_config is expected to be of type GroupViTTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type GroupViTVisionConfig but is of type {type(config.vision_config)}.')
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = TFGroupViTTextTransformer(text_config, name='text_model')
        self.vision_model = TFGroupViTVisionTransformer(vision_config, name='vision_model')
        self.visual_projection = [tf.keras.layers.Dense(self.projection_intermediate_dim, name='visual_projection.0'), tf.keras.layers.BatchNormalization(name='visual_projection.1', momentum=0.9, epsilon=1e-05), tf.keras.layers.ReLU(name='visual_projection.2'), tf.keras.layers.Dense(self.projection_dim, name='visual_projection.3')]
        self.text_projection = [tf.keras.layers.Dense(self.projection_intermediate_dim, name='text_projection.0'), tf.keras.layers.BatchNormalization(name='text_projection.1', momentum=0.9, epsilon=1e-05), tf.keras.layers.ReLU(name='text_projection.2'), tf.keras.layers.Dense(self.projection_dim, name='text_projection.3')]

    def build(self, input_shape: tf.TensorShape):
        if False:
            print('Hello World!')
        self.logit_scale = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True, name='logit_scale')
        super().build(input_shape)

    @unpack_inputs
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = text_outputs[1]
        for layer in self.text_projection:
            pooled_output = layer(pooled_output)
        text_features = pooled_output
        return text_features

    @unpack_inputs
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = vision_outputs[1]
        for layer in self.visual_projection:
            pooled_output = layer(pooled_output)
        image_features = pooled_output
        return image_features

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_segmentation: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        if output_segmentation:
            output_attentions = True
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        for layer in self.visual_projection:
            image_embeds = layer(image_embeds)
        text_embeds = text_outputs[1]
        for layer in self.text_projection:
            text_embeds = layer(text_embeds)
        image_embeds = image_embeds / tf.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, axis=-1, keepdims=True)
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        seg_logits = None
        if output_segmentation:
            image_group_embeds = vision_outputs[0]
            image_group_embeds = tf.reshape(image_group_embeds, shape=(-1, shape_list(image_group_embeds)[-1]))
            for layer in self.visual_projection:
                image_group_embeds = layer(image_group_embeds)
            if output_hidden_states:
                attentions = vision_outputs[3]
            else:
                attentions = vision_outputs[2]
            grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])
            image_group_embeds = image_group_embeds / tf.norm(tensor=image_group_embeds, ord='euclidean', axis=-1, keepdims=True)
            logits_per_image_group = tf.matmul(image_group_embeds, text_embeds, transpose_b=True) * logit_scale
            logits_per_image_group = tf.reshape(logits_per_image_group, shape=(image_embeds.shape[0], -1, text_embeds.shape[0]))
            logits_per_image_group = tf.transpose(logits_per_image_group, perm=(0, 2, 1))
            flatten_grouping = tf.reshape(grouping, shape=(shape_list(grouping)[0], shape_list(grouping)[1], -1))
            seg_logits = tf.matmul(logits_per_image_group, flatten_grouping) * logit_scale
            seg_logits = tf.reshape(seg_logits, shape=(seg_logits.shape[0], seg_logits.shape[1], grouping.shape[2], grouping.shape[3]))
        loss = None
        if return_loss:
            loss = groupvit_loss(logits_per_text)[None, ...]
        if not return_dict:
            if seg_logits is not None:
                output = (logits_per_image, logits_per_text, seg_logits, text_embeds, image_embeds, text_outputs, vision_outputs)
            else:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFGroupViTModelOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, segmentation_logits=seg_logits, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

class TFGroupViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GroupViTConfig
    base_model_prefix = 'groupvit'
GROUPVIT_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TF 2.0 models accepts two formats as inputs:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional arguments.\n\n    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the\n    tensors in the first argument of the model call function: `model(inputs)`.\n\n    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the\n    first positional argument :\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n      `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n      `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    </Tip>\n\n    Args:\n        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GROUPVIT_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'
GROUPVIT_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'
GROUPVIT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details.\n        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'

class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTTextConfig
    main_input_name = 'input_ids'

    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.groupvit = TFGroupViTTextMainLayer(config, name='groupvit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import CLIPTokenizer, TFGroupViTTextModel\n\n        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        outputs = self.groupvit(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFGroupViTVisionModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: GroupViTVisionConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.groupvit = TFGroupViTVisionMainLayer(config, name='groupvit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def call(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFGroupViTVisionModel\n\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled CLS states\n        ```'
        outputs = self.groupvit(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class TFGroupViTModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.groupvit = TFGroupViTMainLayer(config, name='groupvit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        '\n        Returns:\n            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying\n            the projection layer to the pooled output of [`TFGroupViTTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import CLIPTokenizer, TFGroupViTModel\n\n        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        text_features = self.groupvit.get_text_features(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return text_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        '\n        Returns:\n            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying\n            the projection layer to the pooled output of [`TFGroupViTVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFGroupViTModel\n\n        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        image_features = self.groupvit.get_image_features(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFGroupViTModelOutput, config_class=GroupViTConfig)
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_segmentation: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFGroupViTModel\n        >>> import tensorflow as tf\n\n        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True\n        ... )\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = tf.math.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities\n        ```'
        outputs = self.groupvit(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, return_loss=return_loss, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_segmentation=output_segmentation, return_dict=return_dict, training=training)
        return outputs

    def serving_output(self, output: TFGroupViTModelOutput) -> TFGroupViTModelOutput:
        if False:
            for i in range(10):
                print('nop')
        return output