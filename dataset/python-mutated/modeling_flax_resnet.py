from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutputWithNoAttention, FlaxBaseModelOutputWithPoolingAndNoAttention, FlaxImageClassifierOutputWithNoAttention
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
RESNET_START_DOCSTRING = '\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)\n\n    This model is also a\n    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as\n    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and\n    behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
RESNET_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`jax.numpy.float32` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`AutoImageProcessor.__call__`] for details.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        if False:
            i = 10
            return i + 15
        return x

class FlaxResNetConvLayer(nn.Module):
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    activation: Optional[str] = 'relu'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        self.convolution = nn.Conv(self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), strides=self.stride, padding=self.kernel_size // 2, dtype=self.dtype, use_bias=False, kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal', dtype=self.dtype))
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            return 10
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        hidden_state = self.activation_func(hidden_state)
        return hidden_state

class FlaxResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.embedder = FlaxResNetConvLayer(self.config.embedding_size, kernel_size=7, stride=2, activation=self.config.hidden_act, dtype=self.dtype)
        self.max_pool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            i = 10
            return i + 15
        num_channels = pixel_values.shape[-1]
        if num_channels != self.config.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embedding = self.embedder(pixel_values, deterministic=deterministic)
        embedding = self.max_pool(embedding)
        return embedding

class FlaxResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    out_channels: int
    stride: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.convolution = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=self.stride, use_bias=False, kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='truncated_normal'), dtype=self.dtype)
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            for i in range(10):
                print('nop')
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        return hidden_state

class FlaxResNetBasicLayerCollection(nn.Module):
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.layer = [FlaxResNetConvLayer(self.out_channels, stride=self.stride, dtype=self.dtype), FlaxResNetConvLayer(self.out_channels, activation=None, dtype=self.dtype)]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            for i in range(10):
                print('nop')
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state

class FlaxResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """
    in_channels: int
    out_channels: int
    stride: int = 1
    activation: Optional[str] = 'relu'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            return 10
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype) if should_apply_shortcut else None
        self.layer = FlaxResNetBasicLayerCollection(out_channels=self.out_channels, stride=self.stride, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation]

    def __call__(self, hidden_state, deterministic: bool=True):
        if False:
            while True:
                i = 10
        residual = hidden_state
        hidden_state = self.layer(hidden_state, deterministic=deterministic)
        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state += residual
        hidden_state = self.activation_func(hidden_state)
        return hidden_state

class FlaxResNetBottleNeckLayerCollection(nn.Module):
    out_channels: int
    stride: int = 1
    activation: Optional[str] = 'relu'
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        reduces_channels = self.out_channels // self.reduction
        self.layer = [FlaxResNetConvLayer(reduces_channels, kernel_size=1, dtype=self.dtype, name='0'), FlaxResNetConvLayer(reduces_channels, stride=self.stride, dtype=self.dtype, name='1'), FlaxResNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name='2')]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            print('Hello World!')
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state

class FlaxResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions. The first `1x1` convolution reduces the
    input by a factor of `reduction` in order to make the second `3x3` convolution faster. The last `1x1` convolution
    remaps the reduced features to `out_channels`.
    """
    in_channels: int
    out_channels: int
    stride: int = 1
    activation: Optional[str] = 'relu'
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype) if should_apply_shortcut else None
        self.layer = FlaxResNetBottleNeckLayerCollection(self.out_channels, stride=self.stride, activation=self.activation, reduction=self.reduction, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            print('Hello World!')
        residual = hidden_state
        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state = self.layer(hidden_state, deterministic)
        hidden_state += residual
        hidden_state = self.activation_func(hidden_state)
        return hidden_state

class FlaxResNetStageLayersCollection(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """
    config: ResNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        layer = FlaxResNetBottleNeckLayer if self.config.layer_type == 'bottleneck' else FlaxResNetBasicLayer
        layers = [layer(self.in_channels, self.out_channels, stride=self.stride, activation=self.config.hidden_act, dtype=self.dtype, name='0')]
        for i in range(self.depth - 1):
            layers.append(layer(self.out_channels, self.out_channels, activation=self.config.hidden_act, dtype=self.dtype, name=str(i + 1)))
        self.layers = layers

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            return 10
        hidden_state = x
        for layer in self.layers:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state

class FlaxResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """
    config: ResNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.layers = FlaxResNetStageLayersCollection(self.config, in_channels=self.in_channels, out_channels=self.out_channels, stride=self.stride, depth=self.depth, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        if False:
            print('Hello World!')
        return self.layers(x, deterministic=deterministic)

class FlaxResNetStageCollection(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        stages = [FlaxResNetStage(self.config, self.config.embedding_size, self.config.hidden_sizes[0], stride=2 if self.config.downsample_in_first_stage else 1, depth=self.config.depths[0], dtype=self.dtype, name='0')]
        for (i, ((in_channels, out_channels), depth)) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(FlaxResNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1)))
        self.stages = stages

    def __call__(self, hidden_state: jnp.ndarray, output_hidden_states: bool=False, deterministic: bool=True) -> FlaxBaseModelOutputWithNoAttention:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
            hidden_state = stage_module(hidden_state, deterministic=deterministic)
        return (hidden_state, hidden_states)

class FlaxResNetEncoder(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.stages = FlaxResNetStageCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True) -> FlaxBaseModelOutputWithNoAttention:
        if False:
            i = 10
            return i + 15
        (hidden_state, hidden_states) = self.stages(hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return FlaxBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

class FlaxResNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ResNetConfig
    base_model_prefix = 'resnet'
    main_input_name = 'pixel_values'
    module_class: nn.Module = None

    def __init__(self, config: ResNetConfig, input_shape=(1, 224, 224, 3), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            return 10
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            for i in range(10):
                print('nop')
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)
        rngs = {'params': rng}
        random_params = self.module.init(rngs, pixel_values, return_dict=False)
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    def __call__(self, pixel_values, params: dict=None, train: bool=False, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            print('Hello World!')
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        rngs = {}
        return self.module.apply({'params': params['params'] if params is not None else self.params['params'], 'batch_stats': params['batch_stats'] if params is not None else self.params['batch_stats']}, jnp.array(pixel_values, dtype=jnp.float32), not train, output_hidden_states, return_dict, rngs=rngs, mutable=['batch_stats'] if train else False)

class FlaxResNetModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            while True:
                i = 10
        self.embedder = FlaxResNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxResNetEncoder(self.config, dtype=self.dtype)
        self.pooler = partial(nn.avg_pool, padding=((0, 0), (0, 0)))

    def __call__(self, pixel_values, deterministic: bool=True, output_hidden_states: bool=False, return_dict: bool=True) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        if False:
            for i in range(10):
                print('nop')
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embedder(pixel_values, deterministic=deterministic)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state, window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]), strides=(last_hidden_state.shape[1], last_hidden_state.shape[2])).transpose(0, 3, 1, 2)
        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return FlaxBaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)

@add_start_docstrings('The bare ResNet model outputting raw features without any specific head on top.', RESNET_START_DOCSTRING)
class FlaxResNetModel(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetModule
FLAX_VISION_MODEL_DOCSTRING = '\n    Returns:\n\n    Examples:\n\n    ```python\n    >>> from transformers import AutoImageProcessor, FlaxResNetModel\n    >>> from PIL import Image\n    >>> import requests\n\n    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n    >>> image = Image.open(requests.get(url, stream=True).raw)\n    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")\n    >>> model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")\n    >>> inputs = image_processor(images=image, return_tensors="np")\n    >>> outputs = model(**inputs)\n    >>> last_hidden_states = outputs.last_hidden_state\n    ```\n'
overwrite_call_docstring(FlaxResNetModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxResNetModel, output_type=FlaxBaseModelOutputWithPoolingAndNoAttention, config_class=ResNetConfig)

class FlaxResNetClassifierCollection(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name='1')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if False:
            while True:
                i = 10
        return self.classifier(x)

class FlaxResNetForImageClassificationModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        self.resnet = FlaxResNetModule(config=self.config, dtype=self.dtype)
        if self.config.num_labels > 0:
            self.classifier = FlaxResNetClassifierCollection(self.config, dtype=self.dtype)
        else:
            self.classifier = Identity()

    def __call__(self, pixel_values=None, deterministic: bool=True, output_hidden_states=None, return_dict=None):
        if False:
            return 10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.resnet(pixel_values, deterministic=deterministic, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output[:, :, 0, 0])
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output
        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)

@add_start_docstrings('\n    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', RESNET_START_DOCSTRING)
class FlaxResNetForImageClassification(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetForImageClassificationModule
FLAX_VISION_CLASSIF_DOCSTRING = '\n    Returns:\n\n    Example:\n\n    ```python\n    >>> from transformers import AutoImageProcessor, FlaxResNetForImageClassification\n    >>> from PIL import Image\n    >>> import jax\n    >>> import requests\n\n    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n    >>> image = Image.open(requests.get(url, stream=True).raw)\n\n    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")\n    >>> model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")\n\n    >>> inputs = image_processor(images=image, return_tensors="np")\n    >>> outputs = model(**inputs)\n    >>> logits = outputs.logits\n\n    >>> # model predicts one of the 1000 ImageNet classes\n    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)\n    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])\n    ```\n'
overwrite_call_docstring(FlaxResNetForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
append_replace_return_docstrings(FlaxResNetForImageClassification, output_type=FlaxImageClassifierOutputWithNoAttention, config_class=ResNetConfig)