""" PyTorch RegNet model."""
from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'RegNetConfig'
_CHECKPOINT_FOR_DOC = 'facebook/regnet-y-040'
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]
_IMAGE_CLASS_CHECKPOINT = 'facebook/regnet-y-040'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'
REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/regnet-y-040']

class RegNetConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, groups: int=1, activation: Optional[str]='relu'):
        if False:
            print('Hello World!')
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        if False:
            print('Hello World!')
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

class RegNetEmbeddings(nn.Module):
    """
    RegNet Embedddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.embedder = RegNetConvLayer(config.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act)
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        if False:
            for i in range(10):
                print('nop')
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        hidden_state = self.embedder(pixel_values)
        return hidden_state

class RegNetShortCut(nn.Module):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=2):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state

class RegNetSELayer(nn.Module):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, in_channels: int, reduced_channels: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.Sequential(nn.Conv2d(in_channels, reduced_channels, kernel_size=1), nn.ReLU(), nn.Conv2d(reduced_channels, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, hidden_state):
        if False:
            return 10
        pooled = self.pooler(hidden_state)
        attention = self.attention(pooled)
        hidden_state = hidden_state * attention
        return hidden_state

class RegNetXLayer(nn.Module):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=1):
        if False:
            return 10
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.Sequential(RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act), RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act), RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None))
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        if False:
            i = 10
            return i + 15
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

class RegNetYLayer(nn.Module):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=1):
        if False:
            while True:
                i = 10
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.Sequential(RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act), RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act), RegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4))), RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None))
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        if False:
            for i in range(10):
                print('nop')
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

class RegNetStage(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=2, depth: int=2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        layer = RegNetXLayer if config.layer_type == 'x' else RegNetYLayer
        self.layers = nn.Sequential(layer(config, in_channels, out_channels, stride=stride), *[layer(config, out_channels, out_channels) for _ in range(depth - 1)])

    def forward(self, hidden_state):
        if False:
            while True:
                i = 10
        hidden_state = self.layers(hidden_state)
        return hidden_state

class RegNetEncoder(nn.Module):

    def __init__(self, config: RegNetConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.stages = nn.ModuleList([])
        self.stages.append(RegNetStage(config, config.embedding_size, config.hidden_sizes[0], stride=2 if config.downsample_in_first_stage else 1, depth=config.depths[0]))
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for ((in_channels, out_channels), depth) in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RegNetStage(config, in_channels, out_channels, depth=depth))

    def forward(self, hidden_state: Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> BaseModelOutputWithNoAttention:
        if False:
            print('Hello World!')
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

class RegNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RegNetConfig
    base_model_prefix = 'regnet'
    main_input_name = 'pixel_values'

    def _init_weights(self, module):
        if False:
            return 10
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
REGNET_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
REGNET_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`ConvNextImageProcessor.__call__`] for details.\n\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare RegNet model outputting raw features without any specific head on top.', REGNET_START_DOCSTRING)
class RegNetModel(RegNetPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.config = config
        self.embedder = RegNetEmbeddings(config)
        self.encoder = RegNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.post_init()

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> BaseModelOutputWithPoolingAndNoAttention:
        if False:
            i = 10
            return i + 15
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embedder(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)

@add_start_docstrings('\n    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', REGNET_START_DOCSTRING)
class RegNetForImageClassification(RegNetPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.regnet = RegNetModel(config)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity())
        self.post_init()

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> ImageClassifierOutputWithNoAttention:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)