""" PyTorch MaskFormer model."""
import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ... import AutoBackbone
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_scipy_available, logging, replace_return_docstrings, requires_backends
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'MaskFormerConfig'
_CHECKPOINT_FOR_DOC = 'facebook/maskformer-swin-base-ade'
MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/maskformer-swin-base-ade']

@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = None

@dataclass
class MaskFormerPixelLevelModuleOutput(ModelOutput):
    """
    MaskFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a MaskFormerSwin Transformer and the `decoder` is a Feature
    Pyramid Network (FPN).

    The `encoder_last_hidden_state` are referred on the paper as **images features**, while `decoder_last_hidden_state`
    as **pixel embeddings**

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskFormerPixelDecoderOutput(ModelOutput):
    """
    MaskFormer's pixel decoder module output, practically a Feature Pyramid Network. It returns the last hidden state
    and (optionally) the hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] or or
    [`~MaskFormerImageProcessor.post_process_instance_segmentation`] or
    [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~MaskFormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output
            of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def upsample_like(pixel_values: Tensor, like: Tensor, mode: str='bilinear') -> Tensor:
    if False:
        return 10
    '\n    An utility function that upsamples `pixel_values` to match the dimension of `like`.\n\n    Args:\n        pixel_values (`torch.Tensor`):\n            The tensor we wish to upsample.\n        like (`torch.Tensor`):\n            The tensor we wish to use as size target.\n        mode (str, *optional*, defaults to `"bilinear"`):\n            The interpolation mode.\n\n    Returns:\n        `torch.Tensor`: The upsampled tensor\n    '
    (_, _, height, width) = like.shape
    upsampled = nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled

def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    if False:
        print('Hello World!')
    '\n    Compute the DICE loss, similar to generalized IOU for masks as follows:\n\n    $$ \\mathcal{L}_{\\text{dice}(x, y) = 1 - \\frac{2 * x \\cap y }{x \\cup y + 1}} $$\n\n    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow\n\n    $$ \\mathcal{L}_{\\text{dice}(x, y) = 1 - \\frac{2 * x * y }{x + y + 1}} $$\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n        num_masks (`int`):\n            The number of masks present in the current batch, used for normalization.\n\n    Returns:\n        `torch.Tensor`: The computed loss.\n    '
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss

def sigmoid_focal_loss(inputs: Tensor, labels: Tensor, num_masks: int, alpha: float=0.25, gamma: float=2) -> Tensor:
    if False:
        print('Hello World!')
    '\n    Focal loss proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) originally used in\n    RetinaNet. The loss is computed as follows:\n\n    $$ \\mathcal{L}_{\\text{focal loss} = -(1 - p_t)^{\\gamma}\\log{(p_t)} $$\n\n    where \\\\(CE(p_t) = -\\log{(p_t)}}\\\\), CE is the standard Cross Entropy Loss\n\n    Please refer to equation (1,2,3) of the paper for a better understanding.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A float tensor of arbitrary shape.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n        num_masks (`int`):\n            The number of masks present in the current batch, used for normalization.\n        alpha (float, *optional*, defaults to 0.25):\n            Weighting factor in range (0,1) to balance positive vs negative examples.\n        gamma (float, *optional*, defaults to 2.0):\n            Exponent of the modulating factor \\\\(1 - p_t\\\\) to balance easy vs hard examples.\n\n    Returns:\n        `torch.Tensor`: The computed loss.\n    '
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    probs = inputs.sigmoid()
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss
    loss = loss.mean(1).sum() / num_masks
    return loss

def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    '\n    A pair wise version of the dice loss, see `dice_loss` for usage.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n\n    Returns:\n        `torch.Tensor`: The computed loss between each pairs.\n    '
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float=0.25, gamma: float=2.0) -> Tensor:
    if False:
        print('Hello World!')
    '\n    A pair wise version of the focal loss, see `sigmoid_focal_loss` for usage.\n\n    Args:\n        inputs (`torch.Tensor`):\n            A tensor representing a mask.\n        labels (`torch.Tensor`):\n            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs\n            (0 for the negative class and 1 for the positive class).\n        alpha (float, *optional*, defaults to 0.25):\n            Weighting factor in range (0,1) to balance positive vs negative examples.\n        gamma (float, *optional*, defaults to 2.0):\n            Exponent of the modulating factor \\\\(1 - p_t\\\\) to balance easy vs hard examples.\n\n    Returns:\n        `torch.Tensor`: The computed loss between each pairs.\n    '
    if alpha < 0:
        raise ValueError('alpha must be positive')
    height_and_width = inputs.shape[1]
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    prob = inputs.sigmoid()
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    focal_pos = (1 - prob) ** gamma * cross_entropy_loss_pos
    focal_pos *= alpha
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))
    focal_neg = prob ** gamma * cross_entropy_loss_neg
    focal_neg *= 1 - alpha
    loss = torch.matmul(focal_pos, labels.T) + torch.matmul(focal_neg, (1 - labels).T)
    return loss / height_and_width

class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True):
        if False:
            return 10
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        if False:
            for i in range(10):
                print('nop')
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        if False:
            i = 10
            return i + 15
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        return tensor if object_queries is None else tensor + object_queries

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, object_queries: Optional[torch.Tensor]=None, key_value_states: Optional[torch.Tensor]=None, spatial_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            i = 10
            return i + 15
        'Input shape: Batch x Time x Channel'
        position_embeddings = kwargs.pop('position_ebmeddings', None)
        key_value_position_embeddings = kwargs.pop('key_value_position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if key_value_position_embeddings is not None and spatial_position_embeddings is not None:
            raise ValueError('Cannot specify both key_value_position_embeddings and spatial_position_embeddings. Please use just spatial_position_embeddings')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        if key_value_position_embeddings is not None:
            logger.warning_once('key_value_position_embeddings has been deprecated and will be removed in v4.34. Please use spatial_position_embeddings instead')
            spatial_position_embeddings = key_value_position_embeddings
        is_cross_attention = key_value_states is not None
        (batch_size, target_len, embed_dim) = hidden_states.size()
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)
        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(f'Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(f'Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

class DetrDecoderLayer(nn.Module):

    def __init__(self, config: DetrConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DetrAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = DetrAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, object_queries: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative\n                values.\n            object_queries (`torch.FloatTensor`, *optional*):\n                object_queries that are added to the hidden states\n            in the cross-attention layer.\n            query_position_embeddings (`torch.FloatTensor`, *optional*):\n                position embeddings that are added to the queries and keys\n            in the self-attention layer.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative\n                values.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        residual = hidden_states
        (hidden_states, self_attn_weights) = self.self_attn(hidden_states=hidden_states, object_queries=query_position_embeddings, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            (hidden_states, cross_attn_weights) = self.encoder_attn(hidden_states=hidden_states, object_queries=query_position_embeddings, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, spatial_position_embeddings=object_queries, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

class DetrDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, object_queries=None, query_position_embeddings=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                The query embeddings that are passed into the decoder.\n\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:\n\n                - 1 for queries that are **not masked**,\n                - 0 for queries that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected\n                in `[0, 1]`:\n\n                - 1 for pixels that are real (i.e. **not masked**),\n                - 0 for pixels that are padding (i.e. **masked**).\n\n            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Position embeddings that are added to the queries and keys in each cross-attention layer.\n            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):\n                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
        position_embeddings = kwargs.pop('position_embeddings', None)
        if kwargs:
            raise ValueError(f'Unexpected arguments {kwargs.keys()}')
        if position_embeddings is not None and object_queries is not None:
            raise ValueError('Cannot specify both position_embeddings and object_queries. Please use just object_queries')
        if position_embeddings is not None:
            logger.warning_once('position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead')
            object_queries = position_embeddings
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        intermediate = () if self.config.auxiliary_loss else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, None, encoder_hidden_states, encoder_attention_mask, None, output_attentions)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=None, object_queries=object_queries, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate] if v is not None))
        return DetrDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions, intermediate_hidden_states=intermediate)

class MaskFormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1.0, cost_mask: float=1.0, cost_dice: float=1.0):
        if False:
            i = 10
            return i + 15
        'Creates the matcher\n\n        Params:\n            cost_class (float, *optional*, defaults to 1.0):\n                This is the relative weight of the classification error in the matching cost.\n            cost_mask (float, *optional*,  defaults to 1.0):\n                This is the relative weight of the focal loss of the binary mask in the matching cost.\n            cost_dice (float, *optional*, defaults to 1.0):\n                This is the relative weight of the dice loss of the binary mask in the matching cost\n        '
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and (cost_dice == 0):
            raise ValueError('All costs cant be 0')
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        if False:
            while True:
                i = 10
        'Performs the matching\n\n        Params:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor` of dim `batch_size, num_queries, num_labels` with the\n                  classification logits.\n            class_queries_logits (`torch.Tensor`):\n                A tensor` of dim `batch_size, num_queries, height, width` with the\n                  predicted masks.\n\n            class_labels (`torch.Tensor`):\n                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number\n                  of ground-truth objects in the target) containing the class labels.\n            mask_labels (`torch.Tensor`):\n                A tensor` of dim `num_target_boxes, height, width` containing the target\n                  masks.\n\n        Returns:\n            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:\n                - index_i is the indices of the selected predictions (in order)\n                - index_j is the indices of the corresponding selected labels (in order)\n            For each batch element, it holds:\n                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).\n        '
        indices: List[Tuple[np.array]] = []
        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        for (pred_probs, pred_mask, target_mask, labels) in zip(preds_probs, preds_masks, mask_labels, class_labels):
            target_mask = nn.functional.interpolate(target_mask[:, None], size=pred_mask.shape[-2:], mode='nearest')
            pred_probs = pred_probs.softmax(-1)
            cost_class = -pred_probs[:, labels]
            pred_mask_flat = pred_mask.flatten(1)
            target_mask_flat = target_mask[:, 0].flatten(1)
            cost_mask = pair_wise_sigmoid_focal_loss(pred_mask_flat, target_mask_flat)
            cost_dice = pair_wise_dice_loss(pred_mask_flat, target_mask_flat)
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for (i, j) in indices]
        return matched_indices

    def __repr__(self):
        if False:
            while True:
                i = 10
        head = 'Matcher ' + self.__class__.__name__
        body = [f'cost_class: {self.cost_class}', f'cost_mask: {self.cost_mask}', f'cost_dice: {self.cost_dice}']
        _repr_indent = 4
        lines = [head] + [' ' * _repr_indent + line for line in body]
        return '\n'.join(lines)

class MaskFormerLoss(nn.Module):

    def __init__(self, num_labels: int, matcher: MaskFormerHungarianMatcher, weight_dict: Dict[str, float], eos_coef: float):
        if False:
            i = 10
            return i + 15
        '\n        The MaskFormer Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we compute\n        hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair of\n        matched ground-truth / prediction (supervise class and mask)\n\n        Args:\n            num_labels (`int`):\n                The number of classes.\n            matcher (`MaskFormerHungarianMatcher`):\n                A torch module that computes the assigments between the predictions and labels.\n            weight_dict (`Dict[str, float]`):\n                A dictionary of weights to be applied to the different losses.\n            eos_coef (`float`):\n                Weight to apply to the null class.\n        '
        super().__init__()
        requires_backends(self, ['scipy'])
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        if False:
            while True:
                i = 10
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for (index, item) in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        if False:
            print('Hello World!')
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        batch_shape = [batch_size] + max_size
        (b, _, h, w) = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for (tensor, padded_tensor, padding_mask) in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]].copy_(tensor)
            padding_mask[:tensor.shape[1], :tensor.shape[2]] = False
        return (padded_tensors, padding_masks)

    def loss_labels(self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        'Compute the losses related to the labels using cross entropy.\n\n        Args:\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, num_labels`\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n        '
        pred_logits = class_queries_logits
        (batch_size, num_queries, _) = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for (target, (_, j)) in zip(class_labels, indices)])
        target_classes = torch.full((batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {'loss_cross_entropy': loss_ce}
        return losses

    def loss_masks(self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        'Compute the losses related to the masks using focal and dice loss.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, height, width`\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            indices (`Tuple[np.array])`:\n                The indices computed by the Hungarian matcher.\n            num_masks (`int)`:\n                The number of masks, used for normalization.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:\n            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth\n              masks.\n        '
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        (target_masks, _) = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = nn.functional.interpolate(pred_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        pred_masks = pred_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        losses = {'loss_mask': sigmoid_focal_loss(pred_masks, target_masks, num_masks), 'loss_dice': dice_loss(pred_masks, target_masks, num_masks)}
        return losses

    def _get_predictions_permutation_indices(self, indices):
        if False:
            i = 10
            return i + 15
        batch_indices = torch.cat([torch.full_like(src, i) for (i, (src, _)) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return (batch_indices, predictions_indices)

    def _get_targets_permutation_indices(self, indices):
        if False:
            return 10
        batch_indices = torch.cat([torch.full_like(tgt, i) for (i, (_, tgt)) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return (batch_indices, target_indices)

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: List[Tensor], class_labels: List[Tensor], auxiliary_predictions: Optional[Dict[str, Tensor]]=None) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        "\n        This performs the loss computation.\n\n        Args:\n            masks_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, height, width`\n            class_queries_logits (`torch.Tensor`):\n                A tensor of shape `batch_size, num_queries, num_labels`\n            mask_labels (`torch.Tensor`):\n                List of mask labels of shape `(labels, height, width)`.\n            class_labels (`List[torch.Tensor]`):\n                List of class labels of shape `(labels)`.\n            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):\n                if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], then it contains the logits from the\n                inner layers of the Detr's Decoder.\n\n        Returns:\n            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:\n            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.\n            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.\n            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth\n              masks.\n            if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], the dictionary contains addional losses\n            for each auxiliary predictions.\n        "
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks: Number = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: Dict[str, Tensor] = {**self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks), **self.loss_labels(class_queries_logits, class_labels, indices)}
        if auxiliary_predictions is not None:
            for (idx, aux_outputs) in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs['masks_queries_logits']
                class_queries_logits = aux_outputs['class_queries_logits']
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f'{key}_{idx}': value for (key, value) in loss_dict.items()}
                losses.update(loss_dict)
        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Computes the average number of target masks across the batch, for normalization purposes.\n        '
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        return num_masks_pt

class MaskFormerFPNConvLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, kernel_size: int=3, padding: int=1):
        if False:
            return 10
        '\n        A basic module that executes conv - norm - in sequence used in MaskFormer.\n\n        Args:\n            in_features (`int`):\n                The number of input features (channels).\n            out_features (`int`):\n                The number of outputs features (channels).\n        '
        super().__init__()
        self.layers = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False), nn.GroupNorm(32, out_features), nn.ReLU(inplace=True)]
        for (i, layer) in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class MaskFormerFPNLayer(nn.Module):

    def __init__(self, in_features: int, lateral_features: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        A Feature Pyramid Network Layer (FPN) layer. It creates a feature map by aggregating features from the previous\n        and backbone layer. Due to the spatial mismatch, the tensor coming from the previous layer is upsampled.\n\n        Args:\n            in_features (`int`):\n                The number of input features (channels).\n            lateral_features (`int`):\n                The number of lateral features (channels).\n        '
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False), nn.GroupNorm(32, in_features))
        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        left = self.proj(left)
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode='nearest')
        down += left
        down = self.block(down)
        return down

class MaskFormerFPNModel(nn.Module):

    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int=256):
        if False:
            while True:
                i = 10
        '\n        Feature Pyramid Network, given an input tensor and a set of feature map of different feature/spatial size, it\n        creates a list of feature maps with the same feature size.\n\n        Args:\n            in_features (`int`):\n                The number of input features (channels).\n            lateral_widths (`List[int]`):\n                A list with the features (channels) size of each lateral connection.\n            feature_size (int, *optional*, defaults to 256):\n                The features (channels) of the resulting feature maps.\n        '
        super().__init__()
        self.stem = MaskFormerFPNConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(*[MaskFormerFPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        if False:
            print('Hello World!')
        fpn_features = []
        last_feature = features[-1]
        other_features = features[:-1]
        output = self.stem(last_feature)
        for (layer, left) in zip(self.layers, other_features[::-1]):
            output = layer(output, left)
            fpn_features.append(output)
        return fpn_features

class MaskFormerPixelDecoder(nn.Module):

    def __init__(self, *args, feature_size: int=256, mask_feature_size: int=256, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic\n        Segmentation](https://arxiv.org/abs/2107.06278). It first runs the backbone's features into a Feature Pyramid\n        Network creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.\n\n        Args:\n            feature_size (`int`, *optional*, defaults to 256):\n                The feature size (channel dimension) of the FPN feature maps.\n            mask_feature_size (`int`, *optional*, defaults to 256):\n                The features (channels) of the target masks size \\\\(C_{\\epsilon}\\\\) in the paper.\n        "
        super().__init__()
        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    def forward(self, features: List[Tensor], output_hidden_states: bool=False, return_dict: bool=True) -> MaskFormerPixelDecoderOutput:
        if False:
            return 10
        fpn_features = self.fpn(features)
        last_feature_projected = self.mask_projection(fpn_features[-1])
        if not return_dict:
            return (last_feature_projected, tuple(fpn_features)) if output_hidden_states else (last_feature_projected,)
        return MaskFormerPixelDecoderOutput(last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ())

class MaskFormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats: int=64, temperature: int=10000, normalize: bool=False, scale: Optional[float]=None):
        if False:
            return 10
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        if False:
            i = 10
            return i + 15
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PredictionBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for (i, layer) in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class MaskformerMLPPredictionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=3):
        if False:
            print('Hello World!')
        '\n        A classic Multi Layer Perceptron (MLP).\n\n        Args:\n            input_dim (`int`):\n                The input dimensions.\n            hidden_dim (`int`):\n                The hidden dimensions.\n            output_dim (`int`):\n                The output dimensions.\n            num_layers (int, *optional*, defaults to 3):\n                The number of layers.\n        '
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = []
        for (i, (in_dim, out_dim)) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = PredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class MaskFormerPixelLevelModule(nn.Module):

    def __init__(self, config: MaskFormerConfig):
        if False:
            while True:
                i = 10
        '\n        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic\n        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel\n        decoder, generating an image feature map and pixel embeddings.\n\n        Args:\n            config ([`MaskFormerConfig`]):\n                The configuration used to instantiate this model.\n        '
        super().__init__()
        backbone_config = config.backbone_config
        if backbone_config.model_type == 'swin':
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ['stage1', 'stage2', 'stage3', 'stage4']
        self.encoder = AutoBackbone.from_config(backbone_config)
        feature_channels = self.encoder.channels
        self.decoder = MaskFormerPixelDecoder(in_features=feature_channels[-1], feature_size=config.fpn_feature_size, mask_feature_size=config.mask_feature_size, lateral_widths=feature_channels[:-1])

    def forward(self, pixel_values: Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> MaskFormerPixelLevelModuleOutput:
        if False:
            print('Hello World!')
        features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(features, output_hidden_states, return_dict=return_dict)
        if not return_dict:
            last_hidden_state = decoder_output[0]
            outputs = (features[-1], last_hidden_state)
            if output_hidden_states:
                hidden_states = decoder_output[1]
                outputs = outputs + (tuple(features),) + (hidden_states,)
            return outputs
        return MaskFormerPixelLevelModuleOutput(encoder_last_hidden_state=features[-1], decoder_last_hidden_state=decoder_output.last_hidden_state, encoder_hidden_states=tuple(features) if output_hidden_states else (), decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else ())

class MaskFormerTransformerModule(nn.Module):
    """
    The MaskFormer's transformer module.
    """

    def __init__(self, in_features: int, config: MaskFormerConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        hidden_size = config.decoder_config.hidden_size
        should_project = in_features != hidden_size
        self.position_embedder = MaskFormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        self.input_projection = nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else None
        self.decoder = DetrDecoder(config=config.decoder_config)

    def forward(self, image_features: Tensor, output_hidden_states: bool=False, output_attentions: bool=False, return_dict: Optional[bool]=None) -> DetrDecoderOutput:
        if False:
            return 10
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)
        object_queries = self.position_embedder(image_features)
        batch_size = image_features.shape[0]
        queries_embeddings = self.queries_embedder.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.zeros_like(queries_embeddings, requires_grad=True)
        (batch_size, num_channels, height, width) = image_features.shape
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        object_queries = object_queries.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        decoder_output: DetrDecoderOutput = self.decoder(inputs_embeds=inputs_embeds, attention_mask=None, encoder_hidden_states=image_features, encoder_attention_mask=None, object_queries=object_queries, query_position_embeddings=queries_embeddings, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return decoder_output
MASKFORMER_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`MaskFormerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
MASKFORMER_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`MaskFormerImageProcessor.__call__`] for details.\n        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):\n            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:\n\n            - 1 for pixels that are real (i.e. **not masked**),\n            - 0 for pixels that are padding (i.e. **masked**).\n\n            [What are attention masks?](../glossary#attention-mask)\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of Detr's decoder attention layers.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~MaskFormerModelOutput`] instead of a plain tuple.\n"

class MaskFormerPreTrainedModel(PreTrainedModel):
    config_class = MaskFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: nn.Module):
        if False:
            while True:
                i = 10
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(module, MaskFormerTransformerModule):
            if module.input_projection is not None:
                nn.init.xavier_uniform_(module.input_projection.weight, gain=xavier_std)
                nn.init.constant_(module.input_projection.bias, 0)
        elif isinstance(module, MaskFormerFPNModel):
            nn.init.xavier_uniform_(module.stem.get_submodule('0').weight, gain=xavier_std)
        elif isinstance(module, MaskFormerFPNLayer):
            nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)
        elif isinstance(module, MaskFormerFPNConvLayer):
            nn.init.xavier_uniform_(module.get_submodule('0').weight, gain=xavier_std)
        elif isinstance(module, MaskformerMLPPredictionHead):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                    nn.init.constant_(submodule.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

@add_start_docstrings('The bare MaskFormer Model outputting raw hidden-states without any specific head on top.', MASKFORMER_START_DOCSTRING)
class MaskFormerModel(MaskFormerPreTrainedModel):

    def __init__(self, config: MaskFormerConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(in_features=self.pixel_level_module.encoder.channels[-1], config=config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> MaskFormerModelOutput:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, MaskFormerModel\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")\n        >>> model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = image_processor(image, return_tensors="pt")\n\n        >>> # forward pass\n        >>> outputs = model(**inputs)\n\n        >>> # the decoder of MaskFormer outputs hidden states of shape (batch_size, num_queries, hidden_size)\n        >>> transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state\n        >>> list(transformer_decoder_last_hidden_state.shape)\n        [1, 100, 256]\n        ```'
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, _, height, width) = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states, return_dict=return_dict)
        image_features = pixel_level_module_output[0]
        pixel_embeddings = pixel_level_module_output[1]
        transformer_module_output = self.transformer_module(image_features, output_hidden_states, output_attentions)
        queries = transformer_module_output.last_hidden_state
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output[2]
            pixel_decoder_hidden_states = pixel_level_module_output[3]
            transformer_decoder_hidden_states = transformer_module_output[1]
            hidden_states = encoder_hidden_states + pixel_decoder_hidden_states + transformer_decoder_hidden_states
        output = MaskFormerModelOutput(encoder_last_hidden_state=image_features, pixel_decoder_last_hidden_state=pixel_embeddings, transformer_decoder_last_hidden_state=queries, encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, hidden_states=hidden_states, attentions=transformer_module_output.attentions)
        if not return_dict:
            output = tuple((v for v in output.values()))
        return output

class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):

    def __init__(self, config: MaskFormerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.model = MaskFormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)
        self.matcher = MaskFormerHungarianMatcher(cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.cross_entropy_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight}
        self.criterion = MaskFormerLoss(config.num_labels, matcher=self.matcher, weight_dict=self.weight_dict, eos_coef=config.no_object_weight)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, auxiliary_logits: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits)
        for (key, weight) in self.weight_dict.items():
            for (loss_key, loss) in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        if False:
            return 10
        return sum(loss_dict.values())

    def get_logits(self, outputs: MaskFormerModelOutput) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if False:
            while True:
                i = 10
        pixel_embeddings = outputs.pixel_decoder_last_hidden_state
        auxiliary_logits: List[str, Tensor] = []
        if self.config.use_auxiliary_loss:
            stacked_transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
            classes = self.class_predictor(stacked_transformer_decoder_outputs)
            class_queries_logits = classes[-1]
            mask_embeddings = self.mask_embedder(stacked_transformer_decoder_outputs)
            (num_embeddings, batch_size, num_queries, num_channels) = mask_embeddings.shape
            (_, _, height, width) = pixel_embeddings.shape
            binaries_masks = torch.zeros((num_embeddings, batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                binaries_masks += mask_embeddings[..., c][..., None, None] * pixel_embeddings[None, :, None, c]
            masks_queries_logits = binaries_masks[-1]
            for (aux_binary_masks, aux_classes) in zip(binaries_masks[:-1], classes[:-1]):
                auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
        else:
            transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
            classes = self.class_predictor(transformer_decoder_hidden_states)
            class_queries_logits = classes
            mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)
            (batch_size, num_queries, num_channels) = mask_embeddings.shape
            (_, _, height, width) = pixel_embeddings.shape
            masks_queries_logits = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                masks_queries_logits += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
        return (class_queries_logits, masks_queries_logits, auxiliary_logits)

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskFormerForInstanceSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_auxiliary_logits: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> MaskFormerForInstanceSegmentationOutput:
        if False:
            while True:
                i = 10
        '\n        mask_labels (`List[torch.Tensor]`, *optional*):\n            List of mask labels of shape `(num_labels, height, width)` to be fed to a model\n        class_labels (`List[torch.LongTensor]`, *optional*):\n            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the\n            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.\n\n        Returns:\n\n        Examples:\n\n        Semantic segmentation example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")\n        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")\n\n        >>> url = (\n        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"\n        ... )\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # you can pass them to image_processor for postprocessing\n        >>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(\n        ...     outputs, target_sizes=[image.size[::-1]]\n        ... )[0]\n\n        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)\n        >>> list(predicted_semantic_map.shape)\n        [512, 683]\n        ```\n\n        Panoptic segmentation example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> # load MaskFormer fine-tuned on COCO panoptic segmentation\n        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")\n        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`\n        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`\n        >>> class_queries_logits = outputs.class_queries_logits\n        >>> masks_queries_logits = outputs.masks_queries_logits\n\n        >>> # you can pass them to image_processor for postprocessing\n        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]\n\n        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)\n        >>> predicted_panoptic_map = result["segmentation"]\n        >>> list(predicted_panoptic_map.shape)\n        [480, 640]\n        ```\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        raw_outputs = self.model(pixel_values, pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, return_dict=return_dict, output_attentions=output_attentions)
        outputs = MaskFormerModelOutput(encoder_last_hidden_state=raw_outputs[0], pixel_decoder_last_hidden_state=raw_outputs[1], transformer_decoder_last_hidden_state=raw_outputs[2], encoder_hidden_states=raw_outputs[3] if output_hidden_states else None, pixel_decoder_hidden_states=raw_outputs[4] if output_hidden_states else None, transformer_decoder_hidden_states=raw_outputs[5] if output_hidden_states else None, hidden_states=raw_outputs[6] if output_hidden_states else None, attentions=raw_outputs[-1] if output_attentions else None)
        (loss, loss_dict, auxiliary_logits) = (None, None, None)
        (class_queries_logits, masks_queries_logits, auxiliary_logits) = self.get_logits(outputs)
        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits)
            loss = self.get_loss(loss_dict)
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_logits = None
        if not return_dict:
            output = tuple((v for v in (loss, class_queries_logits, masks_queries_logits, auxiliary_logits, *outputs.values()) if v is not None))
            return output
        return MaskFormerForInstanceSegmentationOutput(loss=loss, **outputs, class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits, auxiliary_logits=auxiliary_logits)