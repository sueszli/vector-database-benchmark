""" CLIP model configuration"""
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
logger = logging.get_logger(__name__)
CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {'openai/clip-vit-base-patch32': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json'}

class CLIPTextConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CLIPTextModel`]. It is used to instantiate a CLIP
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import CLIPTextConfig, CLIPTextModel

    >>> # Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPTextConfig()

    >>> # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'clip_text_model'

    def __init__(self, vocab_size=49408, hidden_size=512, intermediate_size=2048, projection_dim=512, num_hidden_layers=12, num_attention_heads=8, max_position_embeddings=77, hidden_act='quick_gelu', layer_norm_eps=1e-05, attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0, pad_token_id=1, bos_token_id=49406, eos_token_id=49407, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        if False:
            while True:
                i = 10
        cls._set_token_in_kwargs(kwargs)
        (config_dict, kwargs) = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'clip':
            config_dict = config_dict['text_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

class CLIPVisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'clip_vision_model'

    def __init__(self, hidden_size=768, intermediate_size=3072, projection_dim=512, num_hidden_layers=12, num_attention_heads=12, num_channels=3, image_size=224, patch_size=32, hidden_act='quick_gelu', layer_norm_eps=1e-05, attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        if False:
            while True:
                i = 10
        cls._set_token_in_kwargs(kwargs)
        (config_dict, kwargs) = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'clip':
            config_dict = config_dict['vision_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

class CLIPConfig(PretrainedConfig):
    """
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    a CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLIPConfig, CLIPModel

    >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPConfig()

    >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
    >>> from transformers import CLIPTextConfig, CLIPVisionConfig

    >>> # Initializing a CLIPText and CLIPVision configuration
    >>> config_text = CLIPTextConfig()
    >>> config_vision = CLIPVisionConfig()

    >>> config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'clip'

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        if False:
            print('Hello World!')
        text_config_dict = kwargs.pop('text_config_dict', None)
        vision_config_dict = kwargs.pop('vision_config_dict', None)
        super().__init__(**kwargs)
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}
            _text_config_dict = CLIPTextConfig(**text_config_dict).to_dict()
            for (key, value) in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key not in ['transformers_version']):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["{key}"]` will be overriden.'
                    logger.warning(message)
            text_config.update(_text_config_dict)
        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}
            _vision_config_dict = CLIPVisionConfig(**vision_config_dict).to_dict()
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {str(key): value for (key, value) in _vision_config_dict['id2label'].items()}
            for (key, value) in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and (key not in ['transformers_version']):
                    if key in vision_config_dict:
                        message = f'`{key}` is found in both `vision_config_dict` and `vision_config` but with different values. The value `vision_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. The value `vision_config["{key}"]` will be overriden.'
                    logger.warning(message)
            vision_config.update(_vision_config_dict)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `CLIPTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `CLIPVisionConfig` with default values.')
        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model\n        configuration.\n\n        Returns:\n            [`CLIPConfig`]: An instance of a configuration object\n        '
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

class CLIPOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            i = 10
            return i + 15
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'}), ('attention_mask', {0: 'batch', 1: 'sequence'})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            for i in range(10):
                print('nop')
        return OrderedDict([('logits_per_image', {0: 'batch'}), ('logits_per_text', {0: 'batch'}), ('text_embeds', {0: 'batch'}), ('image_embeds', {0: 'batch'})])

    @property
    def atol_for_validation(self) -> float:
        if False:
            i = 10
            return i + 15
        return 0.0001

    def generate_dummy_inputs(self, processor: 'ProcessorMixin', batch_size: int=-1, seq_length: int=-1, framework: Optional['TensorType']=None) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        text_input_dict = super().generate_dummy_inputs(processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework)
        image_input_dict = super().generate_dummy_inputs(processor.image_processor, batch_size=batch_size, framework=framework)
        return {**text_input_dict, **image_input_dict}

    @property
    def default_onnx_opset(self) -> int:
        if False:
            return 10
        return 14