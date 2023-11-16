""" MPLUG OWL model configuration """
import copy
import os
from typing import Union
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging
from modelscope.utils.constant import Tasks
logger = logging.get_logger()

class MplugOwlVisionConfig(PretrainedConfig):
    """
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    ```"""
    model_type = 'mplug_owl_vision_model'

    def __init__(self, hidden_size=1024, intermediate_size=4096, projection_dim=768, num_hidden_layers=24, num_attention_heads=16, num_channels=3, image_size=224, patch_size=14, hidden_act='quick_gelu', layer_norm_eps=1e-06, attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0, use_flash_attn=False, use_fp32_layernorm=True, **kwargs):
        if False:
            i = 10
            return i + 15
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
        self.use_flash_attn = use_flash_attn
        self.use_fp32_layernorm = use_fp32_layernorm

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        if False:
            i = 10
            return i + 15
        (config_dict, kwargs) = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'mplug_owl':
            config_dict = config_dict['vision_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

class MplugOwlVisualAbstractorConfig(PretrainedConfig):
    model_type = 'MPlugOwlVisualAbstractor'

    def __init__(self, hidden_size=1024, num_hidden_layers=6, num_attention_heads=16, intermediate_size=4096, attention_probs_dropout_prob=0.1, initializer_range=0.02, layer_norm_eps=1e-06, encoder_hidden_size=1024, use_fp32_layernorm=True, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.use_fp32_layernorm = use_fp32_layernorm

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        if False:
            return 10
        (config_dict, kwargs) = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'mplug_owl':
            config_dict = config_dict['abstractor_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

class MplugOwlConfig(PretrainedConfig):
    """
    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MplugOwlVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MplugOwlVisualAbstractorConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'mplug_owl'
    is_composition = True

    def __init__(self, task=Tasks.multimodal_dialogue, vision_config=None, visual_abstractor_config=None, text_config=None, num_query_tokens=64, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.task = task
        if vision_config is None:
            vision_config = MplugOwlVisionConfig().to_dict()
            logger.info('vision_config is None.')
        if visual_abstractor_config is None:
            visual_abstractor_config = {}
            logger.info('abstractor_config is None. ')
        if text_config is None:
            from transformers.models.llama.configuration_llama import LlamaConfig
            text_config = LlamaConfig(pad_token_id=2).to_dict()
            logger.info('text_config is None.')
        self.vision_config = MplugOwlVisionConfig(**vision_config)
        self.visual_abstractor_config = MplugOwlVisualAbstractorConfig(**visual_abstractor_config)
        text_model_type = text_config['model_type'] if 'model_type' in text_config else 'llama'
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.num_query_tokens = num_query_tokens
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_abstractor_text_configs(cls, vision_config: MplugOwlVisionConfig, visual_abstractor_config: MplugOwlVisualAbstractorConfig, text_config: PretrainedConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            [`MplugOwlConfig`]: An instance of a configuration object\n        '
        return cls(vision_config=vision_config.to_dict(), visual_abstractor_config=visual_abstractor_config.to_dict(), text_config=text_config.to_dict(), **kwargs)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].\n\n        Returns:\n            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,\n        '
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        tmp = self.visual_abstractor_config.to_dict()
        output['visual_abstractor_config'] = tmp
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output