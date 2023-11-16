import logging
from functools import partial
import torch
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, CATEGORY_DISTRIBUTION, LOSS, NUMBER, SET, TIMESERIES, TYPE, VECTOR
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.base import ClassifierConfig, PassthroughDecoderConfig, ProjectorConfig, RegressorConfig
from ludwig.utils.torch_utils import Dense, get_activation
logger = logging.getLogger(__name__)

@DeveloperAPI
class PassthroughDecoder(Decoder):

    def __init__(self, input_size: int=1, num_classes: int=None, decoder_config=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = decoder_config
        logger.debug(f' {self.name}')
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, inputs, **kwargs):
        if False:
            print('Hello World!')
        return inputs

    @staticmethod
    def get_schema_cls():
        if False:
            return 10
        return PassthroughDecoderConfig

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return self.input_shape

@DeveloperAPI
@register_decoder('regressor', [BINARY, NUMBER])
class Regressor(Decoder):

    def __init__(self, input_size, use_bias=True, weights_initializer='xavier_uniform', bias_initializer='zeros', decoder_config=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = decoder_config
        logger.debug(f' {self.name}')
        logger.debug('  Dense')
        self.dense = Dense(input_size=input_size, output_size=1, use_bias=use_bias, weights_initializer=weights_initializer, bias_initializer=bias_initializer)

    @staticmethod
    def get_schema_cls():
        if False:
            print('Hello World!')
        return RegressorConfig

    @property
    def input_shape(self):
        if False:
            return 10
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        if False:
            while True:
                i = 10
        return self.dense(inputs)

@DeveloperAPI
@register_decoder('projector', [VECTOR, TIMESERIES])
class Projector(Decoder):

    def __init__(self, input_size, output_size, use_bias=True, weights_initializer='xavier_uniform', bias_initializer='zeros', activation=None, multiplier=1.0, clip=None, decoder_config=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = decoder_config
        logger.debug(f' {self.name}')
        logger.debug('  Dense')
        self.dense = Dense(input_size=input_size, output_size=output_size, use_bias=use_bias, weights_initializer=weights_initializer, bias_initializer=bias_initializer)
        self.activation = get_activation(activation)
        self.multiplier = multiplier
        if clip is not None:
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                self.clip = partial(torch.clip, min=clip[0], max=clip[1])
            else:
                raise ValueError('The clip parameter of {} is {}. It must be a list or a tuple of length 2.'.format(self.feature_name, self.clip))
        else:
            self.clip = None

    @staticmethod
    def get_schema_cls():
        if False:
            i = 10
            return i + 15
        return ProjectorConfig

    @property
    def input_shape(self):
        if False:
            i = 10
            return i + 15
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        if False:
            while True:
                i = 10
        values = self.activation(self.dense(inputs)) * self.multiplier
        if self.clip:
            values = self.clip(values)
        return values

@DeveloperAPI
@register_decoder('classifier', [CATEGORY, CATEGORY_DISTRIBUTION, SET])
class Classifier(Decoder):

    def __init__(self, input_size, num_classes, use_bias=True, weights_initializer='xavier_uniform', bias_initializer='zeros', decoder_config=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = decoder_config
        logger.debug(f' {self.name}')
        logger.debug('  Dense')
        self.num_classes = num_classes
        self.dense = Dense(input_size=input_size, output_size=num_classes, use_bias=use_bias, weights_initializer=weights_initializer, bias_initializer=bias_initializer)
        self.sampled_loss = False
        if LOSS in kwargs and TYPE in kwargs[LOSS] and (kwargs[LOSS][TYPE] is not None):
            self.sampled_loss = kwargs[LOSS][TYPE].startswith('sampled')

    @staticmethod
    def get_schema_cls():
        if False:
            for i in range(10):
                print('nop')
        return ClassifierConfig

    @property
    def input_shape(self):
        if False:
            i = 10
            return i + 15
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        if False:
            print('Hello World!')
        return self.dense(inputs)