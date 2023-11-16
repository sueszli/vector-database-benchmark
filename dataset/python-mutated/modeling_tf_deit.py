""" TensorFlow DeiT model."""
from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFImageClassifierOutput, TFMaskedImageModelingOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_deit import DeiTConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'DeiTConfig'
_CHECKPOINT_FOR_DOC = 'facebook/deit-base-distilled-patch16-224'
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]
_IMAGE_CLASS_CHECKPOINT = 'facebook/deit-base-distilled-patch16-224'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'
TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/deit-base-distilled-patch16-224']

@dataclass
class TFDeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`DeiTForImageClassificationWithTeacher`].

    Args:
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
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
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

class TFDeiTEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.use_mask_token = use_mask_token
        self.patch_embeddings = TFDeiTPatchEmbeddings(config=config, name='patch_embeddings')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')

    def build(self, input_shape: tf.TensorShape):
        if False:
            i = 10
            return i + 15
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.keras.initializers.zeros(), trainable=True, name='cls_token')
        self.distillation_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.keras.initializers.zeros(), trainable=True, name='distillation_token')
        self.mask_token = None
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.keras.initializers.zeros(), trainable=True, name='mask_token')
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(shape=(1, num_patches + 2, self.config.hidden_size), initializer=tf.keras.initializers.zeros(), trainable=True, name='position_embeddings')
        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None=None, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        embeddings = self.patch_embeddings(pixel_values)
        (batch_size, seq_length, _) = shape_list(embeddings)
        if bool_masked_pos is not None:
            mask_tokens = tf.tile(self.mask_token, [batch_size, seq_length, 1])
            mask = tf.expand_dims(bool_masked_pos, axis=-1)
            mask = tf.cast(mask, dtype=mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        distillation_tokens = tf.repeat(self.distillation_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

class TFDeiTPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        (image_size, patch_size) = (config.image_size, config.patch_size)
        (num_channels, hidden_size) = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Conv2D(hidden_size, kernel_size=patch_size, strides=patch_size, name='projection')

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        (batch_size, height, width, num_channels) = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        x = self.projection(pixel_values)
        (batch_size, height, width, num_channels) = shape_list(x)
        x = tf.reshape(x, (batch_size, height * width, num_channels))
        return x

class TFDeiTSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
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
            print('Hello World!')
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
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

class TFDeiTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFDeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        return hidden_states

class TFDeiTAttention(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.self_attention = TFDeiTSelfAttention(config, name='attention')
        self.dense_output = TFDeiTSelfOutput(config, name='output')

    def prune_heads(self, heads):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        self_outputs = self.self_attention(hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFDeiTIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFDeiTOutput(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
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

class TFDeiTLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.attention = TFDeiTAttention(config, name='attention')
        self.intermediate = TFDeiTIntermediate(config, name='intermediate')
        self.deit_output = TFDeiTOutput(config, name='output')
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        attention_outputs = self.attention(input_tensor=self.layernorm_before(inputs=hidden_states, training=training), head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(inputs=hidden_states, training=training)
        intermediate_output = self.intermediate(hidden_states=layer_output, training=training)
        layer_output = self.deit_output(hidden_states=intermediate_output, input_tensor=hidden_states, training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class TFDeiTEncoder(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.layer = [TFDeiTLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
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
class TFDeiTMainLayer(tf.keras.layers.Layer):
    config_class = DeiTConfig

    def __init__(self, config: DeiTConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFDeiTEmbeddings(config, use_mask_token=use_mask_token, name='embeddings')
        self.encoder = TFDeiTEncoder(config, name='encoder')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = TFDeiTPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> TFDeiTPatchEmbeddings:
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    def get_head_mask(self, head_mask):
        if False:
            print('Hello World!')
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))
        head_mask = self.get_head_mask(head_mask)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class TFDeiTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DeiTConfig
    base_model_prefix = 'deit'
    main_input_name = 'pixel_values'
DEIT_START_DOCSTRING = '\n    This model is a TensorFlow\n    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular\n    TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.\n\n    Parameters:\n        config ([`DeiTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
DEIT_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`DeiTImageProcessor.__call__`] for details.\n\n        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.', DEIT_START_DOCSTRING)
class TFDeiTModel(TFDeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(config, **kwargs)
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token, name='deit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        if False:
            i = 10
            return i + 15
        outputs = self.deit(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFDeiTPooler(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

class TFDeitPixelShuffle(tf.keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f'upscale_factor must be an integer value >= 2 got {upscale_factor}')
        self.upscale_factor = upscale_factor

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            return 10
        hidden_states = x
        (batch_size, _, _, num_input_channels) = shape_list(hidden_states)
        block_size_squared = self.upscale_factor ** 2
        output_depth = int(num_input_channels / block_size_squared)
        permutation = tf.constant([[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]])
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format='NHWC')
        return hidden_states

class TFDeitDecoder(tf.keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(filters=config.encoder_stride ** 2 * config.num_channels, kernel_size=1, name='0')
        self.pixel_shuffle = TFDeitPixelShuffle(config.encoder_stride, name='1')

    def call(self, inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        hidden_states = inputs
        hidden_states = self.conv2d(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        return hidden_states

@add_start_docstrings('DeiT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).', DEIT_START_DOCSTRING)
class TFDeiTForMaskedImageModeling(TFDeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig) -> None:
        if False:
            return 10
        super().__init__(config)
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, use_mask_token=True, name='deit')
        self.decoder = TFDeitDecoder(config, name='decoder')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFMaskedImageModelingOutput]:
        if False:
            i = 10
            return i + 15
        '\n        bool_masked_pos (`tf.Tensor` of type bool and shape `(batch_size, num_patches)`):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren\'t (0).\n\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoImageProcessor, TFDeiTForMaskedImageModeling\n        >>> import tensorflow as tf\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")\n        >>> model = TFDeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")\n\n        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2\n        >>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values\n        >>> # create random boolean mask of shape (batch_size, num_patches)\n        >>> bool_masked_pos = tf.cast(tf.random.uniform((1, num_patches), minval=0, maxval=2, dtype=tf.int32), tf.bool)\n\n        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)\n        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction\n        >>> list(reconstructed_pixel_values.shape)\n        [1, 3, 224, 224]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:-1]
        (batch_size, sequence_length, num_channels) = shape_list(sequence_output)
        height = width = int(sequence_length ** 0.5)
        sequence_output = tf.reshape(sequence_output, (batch_size, height, width, num_channels))
        reconstructed_pixel_values = self.decoder(sequence_output, training=training)
        reconstructed_pixel_values = tf.transpose(reconstructed_pixel_values, (0, 3, 1, 2))
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
            output = (reconstructed_pixel_values,) + outputs[1:]
            return (masked_im_loss,) + output if masked_im_loss is not None else output
        return TFMaskedImageModelingOutput(loss=masked_im_loss, reconstruction=reconstructed_pixel_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', DEIT_START_DOCSTRING)
class TFDeiTForImageClassification(TFDeiTPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: DeiTConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name='deit')
        self.classifier = tf.keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else tf.keras.layers.Activation('linear', name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tf.Tensor, TFImageClassifierOutput]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFDeiTForImageClassification\n        >>> import tensorflow as tf\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> tf.keras.utils.set_random_seed(3)  # doctest: +IGNORE_RESULT\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> # note: we are loading a TFDeiTForImageClassificationWithTeacher from the hub here,\n        >>> # so the head will be randomly initialized, hence the predictions will be random\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")\n        >>> model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")\n\n        >>> inputs = image_processor(images=image, return_tensors="tf")\n        >>> outputs = model(**inputs)\n        >>> logits = outputs.logits\n        >>> # model predicts one of the 1000 ImageNet classes\n        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]\n        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])\n        Predicted class: little blue heron, Egretta caerulea\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of\n    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.\n\n    .. warning::\n\n            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet\n            supported.\n    ', DEIT_START_DOCSTRING)
class TFDeiTForImageClassificationWithTeacher(TFDeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name='deit')
        self.cls_classifier = tf.keras.layers.Dense(config.num_labels, name='cls_classifier') if config.num_labels > 0 else tf.keras.layers.Activation('linear', name='cls_classifier')
        self.distillation_classifier = tf.keras.layers.Dense(config.num_labels, name='distillation_classifier') if config.num_labels > 0 else tf.keras.layers.Activation('linear', name='distillation_classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFDeiTForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFDeiTForImageClassificationWithTeacherOutput]:
        if False:
            i = 10
            return i + 15
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
        logits = (cls_logits + distillation_logits) / 2
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output
        return TFDeiTForImageClassificationWithTeacherOutput(logits=logits, cls_logits=cls_logits, distillation_logits=distillation_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)