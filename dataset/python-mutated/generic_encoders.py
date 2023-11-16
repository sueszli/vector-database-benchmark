import logging
from typing import Optional, Type
import torch
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, ENCODER_OUTPUT, NUMBER, TEXT, TIMESERIES, VECTOR
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.base import BaseEncoderConfig, DenseEncoderConfig, PassthroughEncoderConfig
logger = logging.getLogger(__name__)

@DeveloperAPI
@register_encoder('passthrough', [BINARY, NUMBER, TEXT, VECTOR])
class PassthroughEncoder(Encoder):

    def __init__(self, input_size=1, encoder_config=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = encoder_config
        logger.debug(f' {self.name}')
        self.input_size = input_size

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> EncoderOutputDict:
        if False:
            while True:
                i = 10
        '\n        :param inputs: The inputs fed into the encoder.\n               Shape: [batch x 1], type tf.float32\n        '
        return {ENCODER_OUTPUT: inputs}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        if False:
            for i in range(10):
                print('nop')
        return PassthroughEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return self.input_shape

@DeveloperAPI
@register_encoder('dense', [BINARY, NUMBER, VECTOR, TIMESERIES])
class DenseEncoder(Encoder):

    def __init__(self, input_size, layers=None, num_layers=1, output_size=256, use_bias=True, weights_initializer='xavier_uniform', bias_initializer='zeros', norm=None, norm_params=None, activation='relu', dropout=0, encoder_config=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = encoder_config
        logger.debug(f' {self.name}')
        self.input_size = input_size
        logger.debug('  FCStack')
        self.fc_stack = FCStack(first_layer_input_size=input_size, layers=layers, num_layers=num_layers, default_output_size=output_size, default_use_bias=use_bias, default_weights_initializer=weights_initializer, default_bias_initializer=bias_initializer, default_norm=norm, default_norm_params=norm_params, default_activation=activation, default_dropout=dropout)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> EncoderOutputDict:
        if False:
            print('Hello World!')
        '\n        :param inputs: The inputs fed into the encoder.\n               Shape: [batch x 1], type tf.float32\n        '
        return {ENCODER_OUTPUT: self.fc_stack(inputs)}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        if False:
            i = 10
            return i + 15
        return DenseEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.fc_stack.layers[-1]['output_size']])