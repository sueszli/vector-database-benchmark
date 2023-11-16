""" PyTorch ALIGN model."""
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPoolingAndNoAttention
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'kakaobrain/align-base'
_CONFIG_FOR_DOC = 'AlignConfig'
ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = ['kakaobrain/align-base']
ALIGN_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`AlignConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
ALIGN_TEXT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
ALIGN_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`EfficientNetImageProcessor.__call__`] for details.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
ALIGN_INPUTS_DOCSTRING = "\n    Args:\n       input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`EfficientNetImageProcessor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@dataclass
class AlignVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AlignTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AlignOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`AlignTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The output of [`AlignVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`AlignTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndNoAttention`):
            The output of the [`AlignVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            return 10
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device), label_smoothing=0.1)

def align_loss(similarity: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def round_filters(config: AlignVisionConfig, num_channels: int):
    if False:
        for i in range(10):
            print('nop')
    '\n    Round number of filters based on depth multiplier.\n    '
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)
    if new_dim < 0.9 * num_channels:
        new_dim += divisor
    return int(new_dim)

def correct_pad(kernel_size: Union[int, Tuple], adjust: bool=True):
    if False:
        while True:
            i = 10
    '\n    Utility function to get the tuple padding value for the depthwise convolution.\n\n    Args:\n        kernel_size (`int` or `tuple`):\n            Kernel size of the convolution layers.\n        adjust (`bool`, *optional*, defaults to `True`):\n            Adjusts padding value to apply to right and bottom sides of the input.\n    '
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        return (correct[1], correct[1], correct[0], correct[0])

class AlignVisionEmbeddings(nn.Module):
    """
    A module that corresponds to the stem module of the original work.
    """

    def __init__(self, config: AlignVisionConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.out_dim = round_filters(config, 32)
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.convolution = nn.Conv2d(config.num_channels, self.out_dim, kernel_size=3, stride=2, padding='valid', bias=False)
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        features = self.padding(pixel_values)
        features = self.convolution(features)
        features = self.batchnorm(features)
        features = self.activation(features)
        return features

class AlignVisionDepthwiseConv2d(nn.Conv2d):

    def __init__(self, in_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        if False:
            return 10
        out_channels = in_channels * depth_multiplier
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)

class AlignVisionExpansionLayer(nn.Module):
    """
    This corresponds to the expansion phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, padding='same', bias=False)
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)
        return hidden_states

class AlignVisionDepthwiseLayer(nn.Module):
    """
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, stride: int, kernel_size: int, adjust_padding: bool):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stride = stride
        conv_pad = 'valid' if self.stride == 2 else 'same'
        padding = correct_pad(kernel_size, adjust=adjust_padding)
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        self.depthwise_conv = AlignVisionDepthwiseConv2d(in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False)
        self.depthwise_norm = nn.BatchNorm2d(num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)
        return hidden_states

class AlignVisionSqueezeExciteLayer(nn.Module):
    """
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, expand_dim: int, expand: bool=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduce = nn.Conv2d(in_channels=self.dim, out_channels=self.dim_se, kernel_size=1, padding='same')
        self.expand = nn.Conv2d(in_channels=self.dim_se, out_channels=self.dim, kernel_size=1, padding='same')
        self.act_reduce = ACT2FN[config.hidden_act]
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)
        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = torch.mul(inputs, hidden_states)
        return hidden_states

class AlignVisionFinalBlockLayer(nn.Module):
    """
    This corresponds to the final phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool):
        if False:
            return 10
        super().__init__()
        self.apply_dropout = stride == 1 and (not id_skip)
        self.project_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, padding='same', bias=False)
        self.project_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states)
        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings
        return hidden_states

class AlignVisionBlock(nn.Module):
    """
    This corresponds to the block module of original the EfficientNet vision encoder implementation.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, expand_ratio: int, kernel_size: int, drop_rate: float, id_skip: bool, adjust_padding: bool):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.expand_ratio = expand_ratio
        self.expand = True if self.expand_ratio != 1 else False
        expand_in_dim = in_dim * expand_ratio
        if self.expand:
            self.expansion = AlignVisionExpansionLayer(config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride)
        self.depthwise_conv = AlignVisionDepthwiseLayer(config=config, in_dim=expand_in_dim if self.expand else in_dim, stride=stride, kernel_size=kernel_size, adjust_padding=adjust_padding)
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand)
        self.projection = AlignVisionFinalBlockLayer(config=config, in_dim=expand_in_dim if self.expand else in_dim, out_dim=out_dim, stride=stride, drop_rate=drop_rate, id_skip=id_skip)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if False:
            return 10
        embeddings = hidden_states
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states)
        return hidden_states

class AlignVisionEncoder(nn.Module):
    """
    Forward propogates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
    """

    def __init__(self, config: AlignVisionConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            if False:
                while True:
                    i = 10
            return int(math.ceil(self.depth_coefficient * repeats))
        num_base_blocks = len(config.in_channels)
        num_blocks = sum((round_repeats(n) for n in config.num_block_repeats))
        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]
            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks
                block = AlignVisionBlock(config=config, in_dim=in_dim, out_dim=out_dim, stride=stride, kernel_size=kernel_size, expand_ratio=expand_ratio, drop_rate=drop_rate, id_skip=id_skip, adjust_padding=adjust_padding)
                blocks.append(block)
                curr_block_num += 1
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.FloatTensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> BaseModelOutputWithPoolingAndNoAttention:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

class AlignTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values_length: int=0) -> torch.Tensor:
        if False:
            print('Hello World!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AlignTextSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            return 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            (query_length, key_length) = (query_layer.shape[2], key_layer.shape[2])
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class AlignTextSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AlignTextAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self = AlignTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = AlignTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class AlignTextIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class AlignTextOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AlignTextLayer(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AlignTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = AlignTextAttention(config, position_embedding_type='absolute')
        self.intermediate = AlignTextIntermediate(config)
        self.output = AlignTextOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            while True:
                i = 10
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class AlignTextEncoder(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AlignTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class AlignTextPooler(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AlignPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlignConfig
    base_model_prefix = 'align'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AlignModel):
            nn.init.xavier_uniform_(module.text_projection.weight)
            module.text_projection.bias.data.zero_()
            module.text_projection._is_hf_initialized = True
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

@add_start_docstrings('The text model from ALIGN without any head or projection on top.', ALIGN_START_DOCSTRING)
class AlignTextModel(AlignPreTrainedModel):
    config_class = AlignTextConfig

    def __init__(self, config: AlignTextConfig, add_pooling_layer: bool=True):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.embeddings = AlignTextEmbeddings(config)
        self.encoder = AlignTextEncoder(config)
        self.pooler = AlignTextPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=AlignTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, AlignTextModel\n\n        >>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")\n        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        (batch_size, seq_length) = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

@add_start_docstrings('The vision model from ALIGN without any head or projection on top.', ALIGN_START_DOCSTRING)
class AlignVisionModel(AlignPreTrainedModel):
    config_class = AlignVisionConfig
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = False

    def __init__(self, config: AlignVisionConfig):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.embeddings = AlignVisionEmbeddings(config)
        self.encoder = AlignVisionEncoder(config)
        if config.pooling_type == 'mean':
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == 'max':
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            while True:
                i = 10
        return self.vision_model.embeddings.convolution

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=AlignVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, AlignVisionModel\n\n        >>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")\n        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled CLS states\n        ```'
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)

@add_start_docstrings(ALIGN_START_DOCSTRING)
class AlignModel(AlignPreTrainedModel):
    config_class = AlignConfig

    def __init__(self, config: AlignConfig):
        if False:
            return 10
        super().__init__(config)
        if not isinstance(config.text_config, AlignTextConfig):
            raise ValueError(f'config.text_config is expected to be of type AlignTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, AlignVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type AlignVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.text_model = AlignTextModel(text_config)
        self.vision_model = AlignVisionModel(vision_config)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        self.temperature = nn.Parameter(torch.tensor(self.config.temperature_init_value))
        self.post_init()

    @add_start_docstrings_to_model_forward(ALIGN_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            return 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`AlignTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, AlignModel\n\n        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")\n        >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = text_outputs[0][:, 0, :]
        text_features = self.text_projection(last_hidden_state)
        return text_features

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by\n            applying the projection layer to the pooled output of [`AlignVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, AlignModel\n\n        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")\n        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_features = vision_outputs[1]
        return image_features

    @add_start_docstrings_to_model_forward(ALIGN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AlignOutput, config_class=AlignConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, AlignOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, AlignModel\n\n        >>> model = AlignModel.from_pretrained("kakaobrain/align-base")\n        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True\n        ... )\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[0][:, 0, :]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        logits_per_image = logits_per_text.t()
        loss = None
        if return_loss:
            loss = align_loss(logits_per_text)
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return AlignOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)