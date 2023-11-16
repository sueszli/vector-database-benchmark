""" TF 2.0 CLIP model."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'openai/clip-vit-base-patch32'
TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ['openai/clip-vit-base-patch32']
LARGE_NEGATIVE = -100000000.0

def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int]=None):
    if False:
        while True:
            i = 10
    '\n    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.\n    '
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    return (one_cst - expanded_mask) * LARGE_NEGATIVE

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    if False:
        return 10
    return tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True))

def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    if False:
        return 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

@dataclass
class TFCLIPOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`TFCLIPTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFCLIPVisionModel`].
        text_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPTextModel`].
        vision_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPVisionModel`].
    """
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            for i in range(10):
                print('nop')
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class TFCLIPVisionEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.config = config
        self.patch_embedding = tf.keras.layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='valid', data_format='channels_last', use_bias=False, kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor), name='patch_embedding')

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            for i in range(10):
                print('nop')
        factor = self.config.initializer_factor
        self.class_embedding = self.add_weight(shape=(self.embed_dim,), initializer=get_initializer(self.embed_dim ** (-0.5) * factor), trainable=True, name='class_embedding')
        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(shape=(self.num_positions, self.embed_dim), initializer=get_initializer(self.config.initializer_range * factor), trainable=True, name='embeddings')
        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        '`pixel_values` is expected to be of NCHW format.'
        (batch_size, num_channels, height, width) = shape_list(pixel_values)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))
        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.embed_dim))
        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)
        embeddings = embeddings + self.position_embedding
        return embeddings

class TFCLIPTextEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: CLIPTextConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.config = config

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            i = 10
            return i + 15
        with tf.name_scope('token_embedding'):
            self.weight = self.add_weight(shape=(self.config.vocab_size, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='weight')
        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(shape=(self.config.max_position_embeddings, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='embeddings')
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None) -> tf.Tensor:
        if False:
            while True:
                i = 10
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

class TFCLIPAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, causal_attention_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.q_proj(inputs=hidden_states)
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
        attention_probs = self.dropout(inputs=_attention_probs, training=training)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))
        attention_output = self.out_proj(attention_output, training=training)
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)
        return outputs

class TFCLIPMLP(tf.keras.layers.Layer):

    def __init__(self, config: CLIPConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.activation_fn = get_tf_activation(config.hidden_act)
        factor = config.initializer_factor
        in_proj_std = config.hidden_size ** (-0.5) * (2 * config.num_hidden_layers) ** (-0.5) * factor
        fc_std = (2 * config.hidden_size) ** (-0.5) * factor
        self.fc1 = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name='fc1')
        self.fc2 = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name='fc2')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states

class TFCLIPEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: CLIPConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFCLIPAttention(config, name='self_attn')
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm1')
        self.mlp = TFCLIPMLP(config, name='mlp')
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

class TFCLIPEncoder(tf.keras.layers.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFCLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.layers = [TFCLIPEncoderLayer(config, name=f'layers_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, causal_attention_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class TFCLIPTextTransformer(tf.keras.layers.Layer):

    def __init__(self, config: CLIPTextConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.embeddings = TFCLIPTextEmbeddings(config, name='embeddings')
        self.encoder = TFCLIPEncoder(config, name='encoder')
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
            while True:
                i = 10
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)
        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

@keras_serializable
class TFCLIPTextMainLayer(tf.keras.layers.Layer):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFCLIPTextTransformer(config, name='text_model')

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            while True:
                i = 10
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if False:
            print('Hello World!')
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            return 10
        if input_ids is None:
            raise ValueError('You have to specify input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return text_model_outputs

class TFCLIPVisionTransformer(tf.keras.layers.Layer):

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.embeddings = TFCLIPVisionEmbeddings(config, name='embeddings')
        self.pre_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='pre_layrnorm')
        self.encoder = TFCLIPEncoder(config, name='encoder')
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='post_layernorm')

    def call(self, pixel_values: TFModelInputType, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        embedding_output = self.embeddings(pixel_values=pixel_values)
        embedding_output = self.pre_layernorm(inputs=embedding_output)
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.post_layernorm(inputs=pooled_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@keras_serializable
class TFCLIPVisionMainLayer(tf.keras.layers.Layer):
    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.vision_model = TFCLIPVisionTransformer(config, name='vision_model')

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_model_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return vision_model_outputs

@keras_serializable
class TFCLIPMainLayer(tf.keras.layers.Layer):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(f'config.text_config is expected to be of type CLIPTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}.')
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_model = TFCLIPTextTransformer(text_config, name='text_model')
        self.vision_model = TFCLIPVisionTransformer(vision_config, name='vision_model')
        self.visual_projection = tf.keras.layers.Dense(units=self.projection_dim, kernel_initializer=get_initializer(vision_config.hidden_size ** (-0.5) * self.config.initializer_factor), use_bias=False, name='visual_projection')
        self.text_projection = tf.keras.layers.Dense(units=self.projection_dim, kernel_initializer=get_initializer(text_config.hidden_size ** (-0.5) * self.config.initializer_factor), use_bias=False, name='text_projection')

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            print('Hello World!')
        self.logit_scale = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True, name='logit_scale')
        super().build(input_shape)

    @unpack_inputs
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = text_outputs[1]
        text_features = self.text_projection(inputs=pooled_output)
        return text_features

    @unpack_inputs
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(inputs=pooled_output)
        return image_features

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(inputs=image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(inputs=text_embeds)
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord='euclidean', axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord='euclidean', axis=-1, keepdims=True)
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFCLIPOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

class TFCLIPPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CLIPConfig
    base_model_prefix = 'clip'
    _keys_to_ignore_on_load_missing = ['position_ids']
    _keys_to_ignore_on_load_unexpected = ['position_ids']
CLIP_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
CLIP_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'
CLIP_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details. output_attentions (`bool`, *optional*): Whether or not to\n            return the attentions tensors of all attention layers. See `attentions` under returned tensors for more\n            detail. This argument can be used only in eager mode, in graph mode the value in the config will be used\n            instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'
CLIP_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details.\n        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'

class TFCLIPTextModel(TFCLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.clip = TFCLIPTextMainLayer(config, name='clip')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, TFCLIPTextModel\n\n        >>> model = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")\n        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.clip = TFCLIPVisionMainLayer(config, name='clip')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def call(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFCLIPVisionModel\n\n        >>> model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled CLS states\n        ```'
        outputs = self.clip(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings(CLIP_START_DOCSTRING)
class TFCLIPModel(TFCLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.clip = TFCLIPMainLayer(config, name='clip')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Returns:\n            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying\n            the projection layer to the pooled output of [`TFCLIPTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, TFCLIPModel\n\n        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")\n        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return text_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Returns:\n            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying\n            the projection layer to the pooled output of [`TFCLIPVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFCLIPModel\n\n        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        image_features = self.clip.get_image_features(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=CLIPConfig)
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFCLIPModel\n\n        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True\n        ... )\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities\n        ```'
        outputs = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, return_loss=return_loss, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return outputs

    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput:
        if False:
            print('Hello World!')
        return output