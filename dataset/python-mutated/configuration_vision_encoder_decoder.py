from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
if TYPE_CHECKING:
    from ... import PreTrainedTokenizerBase, TensorType
logger = logging.get_logger(__name__)

class VisionEncoderDecoderConfig(PretrainedConfig):
    """
    [`VisionEncoderDecoderConfig`] is the configuration class to store the configuration of a
    [`VisionEncoderDecoderModel`]. It is used to instantiate a Vision-Encoder-Text-Decoder model according to the
    specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    >>> # Initializing a ViT & BERT style configuration
    >>> config_encoder = ViTConfig()
    >>> config_decoder = BertConfig()

    >>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a ViTBert model (with random weights) from a ViT & bert-base-uncased style configurations
    >>> model = VisionEncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    model_type = 'vision-encoder-decoder'
    is_composition = True

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        if 'encoder' not in kwargs or 'decoder' not in kwargs:
            raise ValueError(f'A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}')
        encoder_config = kwargs.pop('encoder')
        encoder_model_type = encoder_config.pop('model_type')
        decoder_config = kwargs.pop('decoder')
        decoder_model_type = decoder_config.pop('model_type')
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        if False:
            while True:
                i = 10
        '\n        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model\n        configuration and decoder model configuration.\n\n        Returns:\n            [`VisionEncoderDecoderConfig`]: An instance of a configuration object\n        '
        logger.info('Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config')
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.11')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            for i in range(10):
                print('nop')
        return OrderedDict([('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'})])

    @property
    def atol_for_validation(self) -> float:
        if False:
            while True:
                i = 10
        return 0.0001

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            for i in range(10):
                print('nop')
        return OrderedDict({'last_hidden_state': {0: 'batch', 1: 'encoder_sequence'}})

class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            i = 10
            return i + 15
        common_inputs = OrderedDict()
        common_inputs['input_ids'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        common_inputs['attention_mask'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        common_inputs['encoder_hidden_states'] = {0: 'batch', 1: 'encoder_sequence'}
        return common_inputs

    def generate_dummy_inputs(self, tokenizer: 'PreTrainedTokenizerBase', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None) -> Mapping[str, Any]:
        if False:
            return 10
        import torch
        common_inputs = OrderedDict()
        dummy_input = super().generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        (batch, encoder_sequence) = dummy_input['input_ids'].shape
        encoder_hidden_states_shape = (batch, encoder_sequence, self._config.encoder_hidden_size)
        common_inputs['input_ids'] = dummy_input.pop('input_ids')
        common_inputs['attention_mask'] = dummy_input.pop('attention_mask')
        common_inputs['encoder_hidden_states'] = torch.zeros(encoder_hidden_states_shape)
        return common_inputs

class VisionEncoderDecoderOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns ONNX encoder config for `VisionEncoderDecoder` model.\n\n        Args:\n            encoder_config (`PretrainedConfig`):\n                The encoder model's configuration to use when exporting to ONNX.\n\n        Returns:\n            [`VisionEncoderDecoderEncoderOnnxConfig`]: An instance of the ONNX configuration object\n        "
        return VisionEncoderDecoderEncoderOnnxConfig(encoder_config)

    def get_decoder_config(self, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, feature: str='default') -> OnnxConfig:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns ONNX decoder config for `VisionEncoderDecoder` model.\n\n        Args:\n            encoder_config (`PretrainedConfig`):\n                The encoder model's configuration to use when exporting to ONNX.\n            decoder_config (`PretrainedConfig`):\n                The decoder model's configuration to use when exporting to ONNX\n            feature (`str`, *optional*):\n                The type of feature to export the model with.\n\n        Returns:\n            [`VisionEncoderDecoderDecoderOnnxConfig`]: An instance of the ONNX configuration object.\n        "
        decoder_config.encoder_hidden_size = encoder_config.hidden_size
        return VisionEncoderDecoderDecoderOnnxConfig(decoder_config, feature)