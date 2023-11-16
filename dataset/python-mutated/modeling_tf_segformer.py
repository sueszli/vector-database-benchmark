""" TensorFlow SegFormer model."""
from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'SegformerConfig'
_CHECKPOINT_FOR_DOC = 'nvidia/mit-b0'
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]
_IMAGE_CLASS_CHECKPOINT = 'nvidia/mit-b0'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'
TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['nvidia/segformer-b0-finetuned-ade-512-512']

class TFSegformerDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if False:
            for i in range(10):
                print('nop')
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x

class TFSegformerOverlapPatchEmbeddings(tf.keras.layers.Layer):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, hidden_size, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.padding = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.proj = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=stride, padding='VALID', name='proj')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')

    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        if False:
            while True:
                i = 10
        embeddings = self.proj(self.padding(pixel_values))
        height = shape_list(embeddings)[1]
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        embeddings = self.layer_norm(embeddings)
        return (embeddings, height, width)

class TFSegformerEfficientSelfAttention(tf.keras.layers.Layer):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads})')
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = tf.keras.layers.Dense(self.all_head_size, name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, name='value')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=sequence_reduction_ratio, strides=sequence_reduction_ratio, name='sr')
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        batch_size = shape_list(tensor)[0]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False, training: bool=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        batch_size = shape_list(hidden_states)[0]
        num_channels = shape_list(hidden_states)[2]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        if self.sr_ratio > 1:
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            hidden_states = self.sr(hidden_states)
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            hidden_states = self.layer_norm(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, scale)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class TFSegformerSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config: SegformerConfig, hidden_size: int, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name='dense')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class TFSegformerAttention(tf.keras.layers.Layer):

    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.self = TFSegformerEfficientSelfAttention(config=config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio, name='self')
        self.dense_output = TFSegformerSelfOutput(config, hidden_size=hidden_size, name='output')

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        if False:
            return 10
        self_outputs = self.self(hidden_states, height, width, output_attentions)
        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFSegformerDWConv(tf.keras.layers.Layer):

    def __init__(self, dim: int=768, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.depthwise_convolution = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding='same', groups=dim, name='dwconv')

    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        batch_size = shape_list(hidden_states)[0]
        num_channels = shape_list(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        hidden_states = self.depthwise_convolution(hidden_states)
        new_height = shape_list(hidden_states)[1]
        new_width = shape_list(hidden_states)[2]
        num_channels = shape_list(hidden_states)[3]
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        return hidden_states

class TFSegformerMixFFN(tf.keras.layers.Layer):

    def __init__(self, config: SegformerConfig, in_features: int, hidden_features: int=None, out_features: int=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        out_features = out_features or in_features
        self.dense1 = tf.keras.layers.Dense(hidden_features, name='dense1')
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name='dwconv')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = tf.keras.layers.Dense(out_features, name='dense2')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool=False) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.depthwise_convolution(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class TFSegformerLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size: int, num_attention_heads: int, drop_path: float, sequence_reduction_ratio: int, mlp_ratio: int, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm_1')
        self.attention = TFSegformerAttention(config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio, name='attention')
        self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else tf.keras.layers.Activation('linear')
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm_2')
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name='mlp')

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False, training: bool=False) -> Tuple:
        if False:
            for i in range(10):
                print('nop')
        self_attention_outputs = self.attention(self.layer_norm_1(hidden_states), height, width, output_attentions=output_attentions, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states
        outputs = (layer_output,) + outputs
        return outputs

class TFSegformerEncoder(tf.keras.layers.Layer):

    def __init__(self, config: SegformerConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, config.drop_path_rate, sum(config.depths))]
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(TFSegformerOverlapPatchEmbeddings(patch_size=config.patch_sizes[i], stride=config.strides[i], hidden_size=config.hidden_sizes[i], name=f'patch_embeddings.{i}'))
        self.embeddings = embeddings
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(TFSegformerLayer(config, hidden_size=config.hidden_sizes[i], num_attention_heads=config.num_attention_heads[i], drop_path=drop_path_decays[cur + j], sequence_reduction_ratio=config.sr_ratios[i], mlp_ratio=config.mlp_ratios[i], name=f'block.{i}.{j}'))
            blocks.append(layers)
        self.block = blocks
        self.layer_norms = [tf.keras.layers.LayerNormalization(epsilon=1e-05, name=f'layer_norm.{i}') for i in range(config.num_encoder_blocks)]

    def call(self, pixel_values: tf.Tensor, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        if False:
            i = 10
            return i + 15
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = shape_list(pixel_values)[0]
        hidden_states = pixel_values
        for (idx, x) in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            (embedding_layer, block_layer, norm_layer) = x
            (hidden_states, height, width) = embedding_layer(hidden_states)
            for (i, blk) in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions, training=training)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            hidden_states = norm_layer(hidden_states)
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = shape_list(hidden_states)[-1]
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

@keras_serializable
class TFSegformerMainLayer(tf.keras.layers.Layer):
    config_class = SegformerConfig

    def __init__(self, config: SegformerConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFSegformerEncoder(config, name='encoder')

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        encoder_outputs = self.encoder(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple((tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0]))
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]
        return TFBaseModelOutput(last_hidden_state=sequence_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class TFSegformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SegformerConfig
    base_model_prefix = 'segformer'
    main_input_name = 'pixel_values'

    @property
    def input_signature(self):
        if False:
            return 10
        return {'pixel_values': tf.TensorSpec(shape=(None, self.config.num_channels, 512, 512), dtype=tf.float32)}
SEGFORMER_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
SEGFORMER_INPUTS_DOCSTRING = '\n\n    Args:\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`SegformerImageProcessor.__call__`] for details.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n\n        training (`bool`, *optional*, defaults to `False``):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n'

@add_start_docstrings('The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.', SEGFORMER_START_DOCSTRING)
class TFSegformerModel(TFSegformerPreTrainedModel):

    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.segformer = TFSegformerMainLayer(config, name='segformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format('(batch_size, sequence_length)'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        if False:
            print('Hello World!')
        outputs = self.segformer(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('\n    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden\n    states) e.g. for ImageNet.\n    ', SEGFORMER_START_DOCSTRING)
class TFSegformerForImageClassification(TFSegformerPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.segformer = TFSegformerMainLayer(config, name='segformer')
        self.classifier = tf.keras.layers.Dense(config.num_labels, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TFSequenceClassifierOutput]:
        if False:
            i = 10
            return i + 15
        outputs = self.segformer(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        batch_size = shape_list(sequence_output)[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
        sequence_output = tf.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))
        sequence_output = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class TFSegformerMLP(tf.keras.layers.Layer):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Dense(config.decoder_hidden_size, name='proj')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        height = shape_list(hidden_states)[1]
        width = shape_list(hidden_states)[2]
        hidden_dim = shape_list(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))
        hidden_states = self.proj(hidden_states)
        return hidden_states

class TFSegformerDecodeHead(TFSegformerPreTrainedModel):

    def __init__(self, config: SegformerConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, **kwargs)
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config, name=f'linear_c.{i}')
            mlps.append(mlp)
        self.mlps = mlps
        self.linear_fuse = tf.keras.layers.Conv2D(filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name='linear_fuse')
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='batch_norm')
        self.activation = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name='classifier')
        self.config = config

    def call(self, encoder_hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = ()
        for (encoder_hidden_state, mlp) in zip(encoder_hidden_states, self.mlps):
            if self.config.reshape_last_stage is False and len(shape_list(encoder_hidden_state)) == 3:
                height = tf.math.sqrt(tf.cast(shape_list(encoder_hidden_state)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                channel_dim = shape_list(encoder_hidden_state)[-1]
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            (height, width) = shape_list(encoder_hidden_state)[1:3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            channel_dim = shape_list(encoder_hidden_state)[-1]
            encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = shape_list(temp_state)[1:-1]
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method='bilinear')
            all_hidden_states += (encoder_hidden_state,)
        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        logits = self.classifier(hidden_states)
        return logits

@add_start_docstrings('SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.', SEGFORMER_START_DOCSTRING)
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):

    def __init__(self, config: SegformerConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, **kwargs)
        self.segformer = TFSegformerMainLayer(config, name='segformer')
        self.decode_head = TFSegformerDecodeHead(config, name='decode_head')

    def hf_compute_loss(self, logits, labels):
        if False:
            for i in range(10):
                print('nop')
        label_interp_shape = shape_list(labels)[1:]
        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method='bilinear')
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        def masked_loss(real, pred):
            if False:
                i = 10
                return i + 15
            unmasked_loss = loss_fct(real, pred)
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))
        return masked_loss(labels, upsampled_logits)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TFSemanticSegmenterOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):\n            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels > 1`, a (per-pixel) classification loss is computed\n            (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")\n        >>> model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")\n\n        >>> inputs = image_processor(images=image, return_tensors="tf")\n        >>> outputs = model(**inputs, training=False)\n        >>> # logits are of shape (batch_size, num_labels, height/4, width/4)\n        >>> logits = outputs.logits\n        >>> list(logits.shape)\n        [1, 150, 128, 128]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.segformer(pixel_values, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.decode_head(encoder_hidden_states)
        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)