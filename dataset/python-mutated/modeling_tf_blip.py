""" TensorFlow BLIP model."""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, get_tf_activation, keras_serializable, shape_list, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'Salesforce/blip-vqa-base'
TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ['Salesforce/blip-vqa-base', 'Salesforce/blip-vqa-capfilt-large', 'Salesforce/blip-image-captioning-base', 'Salesforce/blip-image-captioning-large', 'Salesforce/blip-itm-base-coco', 'Salesforce/blip-itm-large-coco', 'Salesforce/blip-itm-base-flickr', 'Salesforce/blip-itm-large-flickr']

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    if False:
        i = 10
        return i + 15
    return tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True))

def blip_loss(similarity: tf.Tensor) -> tf.Tensor:
    if False:
        i = 10
        return i + 15
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor`, *optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.`
    """
    loss: Tuple[tf.Tensor] | None = None
    logits: Tuple[tf.Tensor] | None = None
    image_embeds: tf.Tensor | None = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

    @property
    def decoder_logits(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers. Please use the `logits` attribute to retrieve the final output instead.', FutureWarning)
        return self.logits

@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: tf.Tensor | None = None
    image_embeds: tf.Tensor | None = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        itm_score (`tf.Tensor`):
            The image-text similarity scores.
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        vision_pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        question_embeds (`tf.Tensor`):
            The question embeddings obtained by the text projection layer.
    """
    itm_score: tf.Tensor | None = None
    loss: tf.Tensor | None = None
    image_embeds: tf.Tensor | None = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    vision_pooler_output: tf.Tensor | None = None
    attentions: Tuple[tf.Tensor] | None = None
    question_embeds: Tuple[tf.Tensor] | None = None

@dataclass
class TFBlipOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
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
            while True:
                i = 10
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class TFBlipVisionEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: BlipVisionConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = tf.keras.layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, kernel_initializer=get_initializer(self.config.initializer_range), data_format='channels_last', name='patch_embedding')
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.class_embedding = self.add_weight(shape=(1, 1, self.embed_dim), initializer=get_initializer(self.config.initializer_range), trainable=True, name='class_embedding')
        self.position_embedding = self.add_weight(shape=(1, self.num_positions, self.embed_dim), initializer=get_initializer(self.config.initializer_range), trainable=True, name='position_embedding')
        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        batch_size = tf.shape(pixel_values)[0]
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = tf.reshape(patch_embeds, (batch_size, self.num_patches, -1))
        class_embeds = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, :tf.shape(embeddings)[1], :]
        return embeddings

class TFBlipTextEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.config = config

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            return 10
        with tf.name_scope('token_embedding'):
            self.weight = self.add_weight(shape=(self.config.vocab_size, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='weight')
        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(shape=(self.config.max_position_embeddings, self.embed_dim), initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range), trainable=True, name='embeddings')
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None) -> tf.Tensor:
        if False:
            return 10
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

class TFBlipAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout, name='dropout')
        self.qkv = tf.keras.layers.Dense(3 * self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='qkv')
        self.projection = tf.keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='projection')

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: Optional[bool]=None) -> Tuple[tf.Tensor, tf.Tensor | None, Tuple[tf.Tensor] | None]:
        if False:
            while True:
                i = 10
        'Input shape: Batch x Time x Channel'
        (bsz, tgt_len, embed_dim) = shape_list(hidden_states)
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = tf.reshape(mixed_qkv, (bsz, tgt_len, 3, self.num_heads, self.head_dim))
        mixed_qkv = tf.transpose(mixed_qkv, perm=(2, 0, 3, 1, 4))
        (query_states, key_states, value_states) = (mixed_qkv[0], mixed_qkv[1], mixed_qkv[2])
        attention_scores = query_states @ tf.transpose(key_states, (0, 1, 3, 2))
        attention_scores = attention_scores * self.scale
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = tf.transpose(attention_probs @ value_states, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.embed_dim]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        output = self.projection(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output, None)
        return outputs

class TFBlipMLP(tf.keras.layers.Layer):

    def __init__(self, config: BlipConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.activation_fn = get_tf_activation(config.hidden_act)
        in_proj_std = config.hidden_size ** (-0.5) * (2 * config.num_hidden_layers) ** (-0.5)
        fc_std = (2 * config.hidden_size) ** (-0.5)
        self.fc1 = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name='fc1')
        self.fc2 = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name='fc2')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states

class TFBlipEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: BlipConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFBlipAttention(config, name='self_attn')
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm1')
        self.mlp = TFBlipMLP(config, name='mlp')
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm2')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, output_attentions: Optional[bool]=False, training: Optional[bool]=None) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`tf.Tensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, head_mask=attention_mask, output_attentions=output_attentions, training=training)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class TFBlipPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BlipConfig
    base_model_prefix = 'blip'
    _keys_to_ignore_on_load_missing = ['position_ids']
BLIP_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
BLIP_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
BLIP_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@keras_serializable
class TFBlipEncoder(tf.keras.layers.Layer):
    config_class = BlipConfig
    '\n    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a\n    [`BlipEncoderLayer`].\n\n    Args:\n        config (`BlipConfig`):\n            The corresponding vision configuration for the `BlipEncoder`.\n    '

    def __init__(self, config: BlipConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.layers = [TFBlipEncoderLayer(config, name=f'layers_._{i}') for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(self, inputs_embeds, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBaseModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Args:\n            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Embedded representation of the inputs. Should be float, not int tokens.\n            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class TFBlipVisionModel(TFBlipPreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig, *args, **kwargs):
        if False:
            return 10
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.embeddings = TFBlipVisionEmbeddings(config, name='embeddings')
        self.encoder = TFBlipEncoder(config, name='encoder')
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='post_layernorm')

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        if False:
            while True:
                i = 10
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        return TFBaseModelOutputWithPooling(last_hidden_state=output.last_hidden_state, pooler_output=output.pooler_output, hidden_states=hs, attentions=attns)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def call(self, pixel_values: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(tf.expand_dims(pooled_output, 1))
        pooled_output = tf.squeeze(pooled_output, 1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings

class TFBlipMainLayer(tf.keras.layers.Layer):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(f'config.text_config is expected to be of type BlipTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type BlipVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = TFBlipTextModel(text_config, name='text_model')
        self.vision_model = TFBlipVisionModel(vision_config, name='vision_model')
        self.visual_projection = tf.keras.layers.Dense(self.projection_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='visual_projection')
        self.text_projection = tf.keras.layers.Dense(self.projection_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='text_projection')
        self.config = config

    def build(self, input_shape=None):
        if False:
            while True:
                i = 10
        self.logit_scale = self.add_weight(name='logit_scale', shape=[], initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True)
        super().build(input_shape)

    @unpack_inputs
    def call(self, input_ids: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipOutput]:
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / tf.norm(image_embeds, ord=2, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, ord=2, axis=-1, keepdims=True)
        logit_scale = tf.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFBlipOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

class TFBlipModel(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']
    main_input_name = 'input_ids'

    def __init__(self, config: BlipConfig, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.blip = TFBlipMainLayer(config, name='blip')

    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput:
        if False:
            while True:
                i = 10
        return TFBlipOutput(logits_per_image=output.logits_per_image, logits_per_text=output.logits_per_text, text_embeds=output.text_embeds, image_embeds=output.image_embeds)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(self, input_ids: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipModel\n\n        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True\n        ... )\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities\n        ```'
        outputs = self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, return_loss=return_loss, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_dict: Optional[bool]=None) -> tf.Tensor:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying\n            the projection layer to the pooled output of [`TFBlipTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoProcessor, TFBlipModel\n\n        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n\n        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.blip.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_features = self.blip.text_projection(pooled_output)
        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: tf.Tensor | None=None, return_dict: Optional[bool]=None) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Returns:\n            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying\n            the projection layer to the pooled output of [`TFBlipVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipModel\n\n        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        image_features = self.blip.visual_projection(pooled_output)
        return image_features

@add_start_docstrings('\n    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass\n    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,\n    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption\n    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.\n    ', BLIP_START_DOCSTRING)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']
    main_input_name = 'pixel_values'

    def __init__(self, config: BlipConfig, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name='text_decoder')
        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings.patch_embedding

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)
    def call(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration\n\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> text = "A picture of"\n\n        >>> inputs = processor(images=image, text=text, return_tensors="tf")\n\n        >>> outputs = model(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        outputs = self.text_decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, labels=labels, return_dict=return_dict, training=training)
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        if outputs.loss is not None and outputs.loss.shape.rank == 0:
            outputs.loss = tf.reshape(outputs.loss, (1,))
        return TFBlipForConditionalGenerationModelOutput(loss=outputs.loss, logits=outputs.logits, image_embeds=image_embeds, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions)

    def generate(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, **generate_kwargs) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overrides *generate* function to be able to use the model as a conditional generator\n\n        Parameters:\n            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:\n                Input image to be processed\n            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                The sequence used as a prompt for the generation.\n            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration\n\n        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="tf")\n\n        >>> outputs = model.generate(**inputs)\n        >>> print(processor.decode(outputs[0], skip_special_tokens=True))\n        two cats sleeping on a couch\n        ```\n        '
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)
        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        elif input_ids is None:
            input_ids = tf.convert_to_tensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]], dtype=tf.int32)
            input_ids = tf.tile(input_ids, (batch_size, 1))
        input_ids = tf.concat([tf.ones((batch_size, 1), dtype=tf.int32) * self.config.text_config.bos_token_id, input_ids[:, 1:]], axis=1)
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None
        outputs = self.text_decoder.generate(input_ids=input_ids[:, :-1], eos_token_id=self.config.text_config.sep_token_id, pad_token_id=self.config.text_config.pad_token_id, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, **generate_kwargs)
        return outputs

@add_start_docstrings('\n    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text\n    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together\n    with the encoding of the image, and the text decoder will output the answer to the question.\n    ', BLIP_START_DOCSTRING)
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']

    def __init__(self, config: BlipConfig, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_encoder = TFBlipTextModel(config.text_config, name='text_encoder', add_pooling_layer=False)
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name='text_decoder')
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            return 10
        return self.vision_model.embeddings.patch_embedding

    def _shift_right(self, input_ids):
        if False:
            print('Hello World!')
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.decoder_pad_token_id
        if decoder_start_token_id is None or pad_token_id is None:
            raise ValueError('decoder_start_token_id and pad_token_id must be defined!')
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
        shifted_input_ids = tf.where(shifted_input_ids == -100, tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype), shifted_input_ids)
        tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))
        return shifted_input_ids

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None=None, decoder_input_ids: tf.Tensor | None=None, decoder_attention_mask: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipTextVisionModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering\n\n        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> # training\n        >>> text = "How many cats are in the picture?"\n        >>> label = "2"\n        >>> inputs = processor(images=image, text=text, return_tensors="tf")\n        >>> labels = processor(text=label, return_tensors="tf").input_ids\n\n        >>> inputs["labels"] = labels\n        >>> outputs = model(**inputs)\n        >>> loss = outputs.loss\n\n        >>> # inference\n        >>> text = "How many cats are in the picture?"\n        >>> inputs = processor(images=image, text=text, return_tensors="tf")\n        >>> outputs = model.generate(**inputs)\n        >>> print(processor.decode(outputs[0], skip_special_tokens=True))\n        2\n        ```'
        if labels is None and decoder_input_ids is None:
            raise ValueError('Either `decoder_input_ids` or `labels` should be passed when calling `TFBlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=return_dict, training=training)
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = labels
        answer_output = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=question_embeds, encoder_attention_mask=attention_mask, labels=labels, return_dict=return_dict, training=training)
        if labels is not None:
            decoder_loss = tf.reduce_mean(answer_output.loss) if return_dict else tf.reduce_mean(answer_output[0])
        else:
            decoder_loss = None
        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return TFBlipTextVisionModelOutput(loss=decoder_loss, image_embeds=image_embeds, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions)

    def generate(self, input_ids: tf.Tensor, pixel_values: tf.Tensor, attention_mask: tf.Tensor | None=None, **generate_kwargs) -> tf.Tensor:
        if False:
            while True:
                i = 10
        '\n        Overrides *generate* function to be able to use the model as a conditional generator\n\n        Parameters:\n            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n                The sequence used as a prompt for the generation.\n            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:\n                Input image to be processed\n            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for\n                tokens that are NOT MASKED, `0` for MASKED tokens.\n            generate_kwargs (dict, *optional*):\n                Additional arguments passed to the `generate` function of the decoder\n\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering\n\n        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> text = "How many cats are in the picture?"\n\n        >>> inputs = processor(images=image, text=text, return_tensors="tf")\n\n        >>> outputs = model.generate(**inputs)\n        >>> print(processor.decode(outputs[0], skip_special_tokens=True))\n        2\n        ```\n        '
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)
        if isinstance(input_ids, list):
            input_ids = tf.Tensor(input_ids)
        question_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=False)
        question_embeds = question_outputs[0]
        question_attention_mask = tf.ones(shape_list(question_embeds)[:-1], dtype=tf.int32)
        bos_ids = tf.fill((tf.shape(question_embeds)[0], 1), value=tf.cast(self.decoder_start_token_id, input_ids.dtype))
        outputs = self.text_decoder.generate(input_ids=bos_ids, eos_token_id=self.config.text_config.sep_token_id, pad_token_id=self.config.text_config.pad_token_id, encoder_hidden_states=question_embeds, encoder_attention_mask=question_attention_mask, **generate_kwargs)
        return outputs

@add_start_docstrings('\n    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of\n    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to\n    the image.\n    ', BLIP_START_DOCSTRING)
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_encoder = TFBlipTextModel(config.text_config, name='text_encoder', add_pooling_layer=False)
        self.vision_proj = tf.keras.layers.Dense(config.image_text_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='vision_proj')
        self.text_proj = tf.keras.layers.Dense(config.image_text_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='text_proj')
        self.itm_head = tf.keras.layers.Dense(2, kernel_initializer=get_initializer(config.initializer_range), name='itm_head')
        self.decoder_pad_token_id = config.text_config.pad_token_id if not hasattr(config, 'decoder_pad_token_id') else config.decoder_pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id if not hasattr(config, 'decoder_start_token_id') else config.decoder_start_token_id

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings.patch_embedding

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None=None, use_itm_head: Optional[bool]=True, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipImageTextMatchingModelOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, TFBlipForImageTextRetrieval\n\n        >>> model = TFBlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> text = "an image of a cat"\n\n        >>> inputs = processor(images=image, text=text, return_tensors="tf")\n        >>> outputs = model(**inputs)\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        image_atts = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)
        itm_question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=return_dict, training=training)
        itm_question_embeds = itm_question_embeds[0] if not return_dict else itm_question_embeds.last_hidden_state
        itm_output = self.itm_head(itm_question_embeds[:, 0, :])
        no_itm_question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict, training=training)
        no_itm_question_embeds = no_itm_question_embeds[0] if not return_dict else no_itm_question_embeds.last_hidden_state
        (image_feat, _) = tf.linalg.normalize(self.vision_proj(image_embeds[:, 0, :]), ord=2, axis=-1)
        (text_feat, _) = tf.linalg.normalize(self.text_proj(no_itm_question_embeds[:, 0, :]), ord=2, axis=-1)
        no_itm_output = tf.matmul(image_feat, text_feat, transpose_b=True)
        if use_itm_head:
            output = itm_output
            question_embeds = itm_question_embeds
        else:
            output = no_itm_output
            question_embeds = no_itm_question_embeds
        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple((output for output in outputs if output is not None))
        return TFBlipImageTextMatchingModelOutput(itm_score=output, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions, question_embeds=question_embeds)