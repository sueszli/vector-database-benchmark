""" TF 2.0 ConvNext model."""
from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_convnext import ConvNextConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'ConvNextConfig'
_CHECKPOINT_FOR_DOC = 'facebook/convnext-tiny-224'

class TFConvNextDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if False:
            print('Hello World!')
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x

class TFConvNextEmbeddings(tf.keras.layers.Layer):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config: ConvNextConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.patch_embeddings = tf.keras.layers.Conv2D(filters=config.hidden_sizes[0], kernel_size=config.patch_size, strides=config.patch_size, name='patch_embeddings', kernel_initializer=get_initializer(config.initializer_range), bias_initializer=tf.keras.initializers.Zeros())
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-06, name='layernorm')
        self.num_channels = config.num_channels

    def call(self, pixel_values):
        if False:
            return 10
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values['pixel_values']
        tf.debugging.assert_equal(shape_list(pixel_values)[1], self.num_channels, message='Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings

class TFConvNextLayer(tf.keras.layers.Layer):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch. Since we already permuted the inputs to follow
    NHWC ordering, we can just apply the operations straight-away without the permutation.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0.0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dim = dim
        self.config = config
        self.dwconv = tf.keras.layers.Conv2D(filters=dim, kernel_size=7, padding='same', groups=dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='dwconv')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-06, name='layernorm')
        self.pwconv1 = tf.keras.layers.Dense(units=4 * dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='pwconv1')
        self.act = get_tf_activation(config.hidden_act)
        self.pwconv2 = tf.keras.layers.Dense(units=dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='pwconv2')
        self.drop_path = TFConvNextDropPath(drop_path, name='drop_path') if drop_path > 0.0 else tf.keras.layers.Activation('linear', name='drop_path')

    def build(self, input_shape: tf.TensorShape=None):
        if False:
            print('Hello World!')
        self.layer_scale_parameter = self.add_weight(shape=(self.dim,), initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value), trainable=True, name='layer_scale_parameter') if self.config.layer_scale_init_value > 0 else None
        super().build(input_shape)

    def call(self, hidden_states, training=False):
        if False:
            print('Hello World!')
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = input + self.drop_path(x, training=training)
        return x

class TFConvNextStage(tf.keras.layers.Layer):
    """ConvNext stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config (`ConvNextV2Config`):
            Model configuration class.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        depth (`int`):
            Number of residual blocks.
        drop_path_rates(`List[float]`):
            Stochastic depth rates for each layer.
    """

    def __init__(self, config: ConvNextConfig, in_channels: int, out_channels: int, kernel_size: int=2, stride: int=2, depth: int=2, drop_path_rates: Optional[List[float]]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [tf.keras.layers.LayerNormalization(epsilon=1e-06, name='downsampling_layer.0'), tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=tf.keras.initializers.Zeros(), name='downsampling_layer.1')]
        else:
            self.downsampling_layer = [tf.identity]
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = [TFConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j], name=f'layers.{j}') for j in range(depth)]

    def call(self, hidden_states):
        if False:
            return 10
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class TFConvNextEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.stages = []
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = TFConvNextStage(config, in_channels=prev_chs, out_channels=out_chs, stride=2 if i > 0 else 1, depth=config.depths[i], drop_path_rates=drop_path_rates[i], name=f'stages.{i}')
            self.stages.append(stage)
            prev_chs = out_chs

    def call(self, hidden_states, output_hidden_states=False, return_dict=True):
        if False:
            i = 10
            return i + 15
        all_hidden_states = () if output_hidden_states else None
        for (i, layer_module) in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

@keras_serializable
class TFConvNextMainLayer(tf.keras.layers.Layer):
    config_class = ConvNextConfig

    def __init__(self, config: ConvNextConfig, add_pooling_layer: bool=True, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFConvNextEmbeddings(config, name='embeddings')
        self.encoder = TFConvNextEncoder(config, name='encoder')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = tf.keras.layers.GlobalAvgPool2D(data_format='channels_first') if add_pooling_layer else None

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            return 10
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values, training=training)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        pooled_output = self.layernorm(self.pooler(last_hidden_state))
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states)

class TFConvNextPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ConvNextConfig
    base_model_prefix = 'convnext'
    main_input_name = 'pixel_values'
CONVNEXT_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
CONVNEXT_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`ConvNextImageProcessor.__call__`] for details.\n\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n'

@add_start_docstrings('The bare ConvNext model outputting raw features without any specific head on top.', CONVNEXT_START_DOCSTRING)
class TFConvNextModel(TFConvNextPreTrainedModel):

    def __init__(self, config, *inputs, add_pooling_layer=True, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.convnext = TFConvNextMainLayer(config, add_pooling_layer=add_pooling_layer, name='convnext')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFConvNextModel\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")\n        >>> model = TFConvNextModel.from_pretrained("facebook/convnext-tiny-224")\n\n        >>> inputs = image_processor(images=image, return_tensors="tf")\n        >>> outputs = model(**inputs)\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        outputs = self.convnext(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return (outputs[0],) + outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=outputs.last_hidden_state, pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states)

@add_start_docstrings('\n    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', CONVNEXT_START_DOCSTRING)
class TFConvNextForImageClassification(TFConvNextPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: ConvNextConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convnext = TFConvNextMainLayer(config, name='convnext')
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TFConvNextForImageClassification\n        >>> import tensorflow as tf\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")\n        >>> model = TFConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")\n\n        >>> inputs = image_processor(images=image, return_tensors="tf")\n        >>> outputs = model(**inputs)\n        >>> logits = outputs.logits\n        >>> # model predicts one of the 1000 ImageNet classes\n        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]\n        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])\n        ```'
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)