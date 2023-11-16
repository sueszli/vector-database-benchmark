""" PyTorch Table Transformer model."""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_scipy_available, is_timm_available, is_vision_available, logging, replace_return_docstrings, requires_backends
from ..auto import AutoBackbone
from .configuration_table_transformer import TableTransformerConfig
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment
if is_timm_available():
    from timm import create_model
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'TableTransformerConfig'
_CHECKPOINT_FOR_DOC = 'microsoft/table-transformer-detection'
TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/table-transformer-detection']

@dataclass
class TableTransformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the TABLE_TRANSFORMER decoder. This class adds one attribute to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations, i.e. the output
    of each decoder layer, each of them gone through a layernorm. This is useful when training the model with auxiliary
    decoding losses.

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
class TableTransformerModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the TABLE_TRANSFORMER encoder-decoder model. This class adds one attribute to
    Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder
    layer, each of them gone through a layernorm. This is useful when training the model with auxiliary decoding
    losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = None

@dataclass
class TableTransformerObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TableTransformerForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~TableTransformerImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class TableTransformerFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        if False:
            while True:
                i = 10
        super().__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if False:
            print('Hello World!')
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-05
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias

def replace_batch_norm(model):
    if False:
        i = 10
        return i + 15
    '\n    Recursively replace all `torch.nn.BatchNorm2d` with `TableTransformerFrozenBatchNorm2d`.\n\n    Args:\n        model (torch.nn.Module):\n            input model\n    '
    for (name, module) in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = TableTransformerFrozenBatchNorm2d(module.num_features)
            if not module.weight.device == torch.device('meta'):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)
            model._modules[name] = new_module
        if len(list(module.children())) > 0:
            replace_batch_norm(module)

class TableTransformerConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by TableTransformerFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        if config.use_timm_backbone:
            requires_backends(self, ['timm'])
            kwargs = {}
            if config.dilation:
                kwargs['output_stride'] = 16
            backbone = create_model(config.backbone, pretrained=config.use_pretrained_backbone, features_only=True, out_indices=(1, 2, 3, 4), in_chans=config.num_channels, **kwargs)
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if 'resnet' in backbone_model_type:
            for (name, parameter) in self.model.named_parameters():
                if config.use_timm_backbone:
                    if 'layer2' not in name and 'layer3' not in name and ('layer4' not in name):
                        parameter.requires_grad_(False)
                elif 'stage.1' not in name and 'stage.2' not in name and ('stage.3' not in name):
                    parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        out = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out

class TableTransformerConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        if False:
            print('Hello World!')
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        if False:
            return 10
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for (feature_map, mask) in out:
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))
        return (out, pos)

class TableTransformerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if False:
            for i in range(10):
                print('nop')
        if pixel_mask is None:
            raise ValueError('No pixel mask provided')
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-06) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-06) * self.scale
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embedding_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class TableTransformerLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        if False:
            i = 10
            return i + 15
        (height, width) = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(config):
    if False:
        for i in range(10):
            print('nop')
    n_steps = config.d_model // 2
    if config.position_embedding_type == 'sine':
        position_embedding = TableTransformerSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == 'learned':
        position_embedding = TableTransformerLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f'Not supported {config.position_embedding_type}')
    return position_embedding

class TableTransformerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the TABLE_TRANSFORMER paper).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
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

class TableTransformerEncoderLayer(nn.Module):

    def __init__(self, config: TableTransformerConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = TableTransformerAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, object_queries: torch.Tensor=None, output_attentions: bool=False):
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative\n                values.\n            object_queries (`torch.FloatTensor`, *optional*): object queries, to be added to hidden_states.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, object_queries=object_queries, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class TableTransformerDecoderLayer(nn.Module):

    def __init__(self, config: TableTransformerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = TableTransformerAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = TableTransformerAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, object_queries: Optional[torch.Tensor]=None, query_position_embeddings: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative\n                values.\n            object_queries (`torch.FloatTensor`, *optional*):\n                object queries that are added to the queries and keys\n            in the cross-attention layer.\n            query_position_embeddings (`torch.FloatTensor`, *optional*):\n                object queries that are added to the queries and keys\n            in the self-attention layer.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative\n                values.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, self_attn_weights) = self.self_attn(hidden_states=hidden_states, object_queries=query_position_embeddings, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            (hidden_states, cross_attn_weights) = self.encoder_attn(hidden_states=hidden_states, object_queries=query_position_embeddings, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, spatial_position_embeddings=object_queries, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

class TableTransformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        if False:
            return 10
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class TableTransformerPreTrainedModel(PreTrainedModel):
    config_class = TableTransformerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'
    _no_split_modules = ['TableTransformerConvEncoder', 'TableTransformerEncoderLayer', 'TableTransformerDecoderLayer']

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        std = self.config.init_std
        if isinstance(module, TableTransformerLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
TABLE_TRANSFORMER_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`TableTransformerConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
TABLE_TRANSFORMER_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it.\n\n            Pixel values can be obtained using [`DetrImageProcessor`]. See [`DetrImageProcessor.__call__`] for details.\n\n        pixel_mask (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):\n            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:\n\n            - 1 for pixels that are real (i.e. **not masked**),\n            - 0 for pixels that are padding (i.e. **masked**).\n\n            [What are attention masks?](../glossary#attention-mask)\n\n        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):\n            Not used by default. Can be used to mask object queries.\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you\n            can choose to directly pass a flattened representation of an image.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):\n            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an\n            embedded representation.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class TableTransformerEncoder(TableTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TableTransformerEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for Table Transformer:

    - object_queries are added to the forward pass.

    Args:
        config: TableTransformerConfig
    """

    def __init__(self, config: TableTransformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.layers = nn.ModuleList([TableTransformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)
        self.post_init()

    def forward(self, inputs_embeds=None, attention_mask=None, object_queries=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.\n\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:\n\n                - 1 for pixel features that are real (i.e. **not masked**),\n                - 0 for pixel features that are padding (i.e. **masked**).\n\n                [What are attention masks?](../glossary#attention-mask)\n\n            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Position embeddings that are added to the queries and keys in each self-attention layer.\n\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, object_queries=object_queries, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        hidden_states = self.layernorm(hidden_states)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class TableTransformerDecoder(TableTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TableTransformerDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for TABLE_TRANSFORMER:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: TableTransformerConfig
    """

    def __init__(self, config: TableTransformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.layers = nn.ModuleList([TableTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, inputs_embeds=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, object_queries=None, query_position_embeddings=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if False:
            return 10
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                The query embeddings that are passed into the decoder.\n\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:\n\n                - 1 for queries that are **not masked**,\n                - 0 for queries that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected\n                in `[0, 1]`:\n\n                - 1 for pixels that are real (i.e. **not masked**),\n                - 0 for pixels that are padding (i.e. **masked**).\n\n            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Object queries that are added to the queries and keys in each cross-attention layer.\n            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):\n                , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.\n\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
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
        combined_attention_mask = None
        if attention_mask is not None and combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
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
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, combined_attention_mask, encoder_hidden_states, encoder_attention_mask, None)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=combined_attention_mask, object_queries=object_queries, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
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
        return TableTransformerDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions, intermediate_hidden_states=intermediate)

@add_start_docstrings('\n    The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) outputting raw\n    hidden-states without any specific head on top.\n    ', TABLE_TRANSFORMER_START_DOCSTRING)
class TableTransformerModel(TableTransformerPreTrainedModel):

    def __init__(self, config: TableTransformerConfig):
        if False:
            return 10
        super().__init__(config)
        backbone = TableTransformerConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = TableTransformerConvModel(backbone, object_queries)
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.encoder = TableTransformerEncoder(config)
        self.decoder = TableTransformerDecoder(config)
        self.post_init()

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.decoder

    def freeze_backbone(self):
        if False:
            for i in range(10):
                print('nop')
        for (name, param) in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        if False:
            for i in range(10):
                print('nop')
        for (name, param) in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], TableTransformerModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, TableTransformerModel\n        >>> from huggingface_hub import hf_hub_download\n        >>> from PIL import Image\n\n        >>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")\n        >>> image = Image.open(file_path).convert("RGB")\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")\n        >>> model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")\n\n        >>> # prepare image for the model\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n\n        >>> # forward pass\n        >>> outputs = model(**inputs)\n\n        >>> # the last hidden states are the final query embeddings of the Transformer decoder\n        >>> # these are of shape (batch_size, num_queries, hidden_size)\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 15, 256]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, num_channels, height, width) = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)
        (features, position_embeddings_list) = self.backbone(pixel_values, pixel_mask)
        (feature_map, mask) = features[-1]
        if mask is None:
            raise ValueError('Backbone does not return downsampled pixel mask')
        projected_feature_map = self.input_projection(feature_map)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=flattened_features, attention_mask=flattened_mask, object_queries=object_queries, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        decoder_outputs = self.decoder(inputs_embeds=queries, attention_mask=None, object_queries=object_queries, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=flattened_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return TableTransformerModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, intermediate_hidden_states=decoder_outputs.intermediate_hidden_states)

@add_start_docstrings('\n    Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on\n    top, for tasks such as COCO detection.\n    ', TABLE_TRANSFORMER_START_DOCSTRING)
class TableTransformerForObjectDetection(TableTransformerPreTrainedModel):

    def __init__(self, config: TableTransformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.model = TableTransformerModel(config)
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)
        self.bbox_predictor = TableTransformerMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3)
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        if False:
            print('Hello World!')
        return [{'logits': a, 'pred_boxes': b} for (a, b) in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[List[Dict]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], TableTransformerObjectDetectionOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`List[Dict]` of len `(batch_size,)`, *optional*):\n            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the\n            following 2 keys: \'class_labels\' and \'boxes\' (the class labels and bounding boxes of an image in the batch\n            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes\n            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from huggingface_hub import hf_hub_download\n        >>> from transformers import AutoImageProcessor, TableTransformerForObjectDetection\n        >>> import torch\n        >>> from PIL import Image\n\n        >>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")\n        >>> image = Image.open(file_path).convert("RGB")\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")\n        >>> model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> # convert outputs (bounding boxes and class logits) to COCO API\n        >>> target_sizes = torch.tensor([image.size[::-1]])\n        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[\n        ...     0\n        ... ]\n\n        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):\n        ...     box = [round(i, 2) for i in box.tolist()]\n        ...     print(\n        ...         f"Detected {model.config.id2label[label.item()]} with confidence "\n        ...         f"{round(score.item(), 3)} at location {box}"\n        ...     )\n        Detected table with confidence 1.0 at location [202.1, 210.59, 1119.22, 385.09]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        (loss, loss_dict, auxiliary_outputs) = (None, None, None)
        if labels is not None:
            matcher = TableTransformerHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality']
            criterion = TableTransformerLoss(matcher=matcher, num_classes=self.config.num_labels, eos_coef=self.config.eos_coefficient, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coefficient}
            weight_dict['loss_giou'] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for (k, v) in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum((loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return (loss, loss_dict) + output if loss is not None else output
        return TableTransformerObjectDetectionOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, auxiliary_outputs=auxiliary_outputs, last_hidden_state=outputs.last_hidden_state, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

def dice_loss(inputs, targets, num_boxes):
    if False:
        while True:
            i = 10
    '\n    Compute the DICE loss, similar to generalized IOU for masks\n\n    Args:\n        inputs: A float tensor of arbitrary shape.\n                The predictions for each example.\n        targets: A float tensor with the same shape as inputs. Stores the binary\n                 classification label for each element in inputs (0 for the negative class and 1 for the positive\n                 class).\n    '
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float=0.25, gamma: float=2):
    if False:
        return 10
    '\n    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.\n\n    Args:\n        inputs (`torch.FloatTensor` of arbitrary shape):\n            The predictions for each example.\n        targets (`torch.FloatTensor` with the same shape as `inputs`)\n            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class\n            and 1 for the positive class).\n        alpha (`float`, *optional*, defaults to `0.25`):\n            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.\n        gamma (`int`, *optional*, defaults to `2`):\n            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.\n\n    Returns:\n        Loss tensor\n    '
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes

class TableTransformerLoss(nn.Module):
    """
    This class computes the losses for TableTransformerForObjectDetection/TableTransformerForSegmentation. The process
    happens in two steps: 1) we compute hungarian assignment between ground truth boxes and the outputs of the model 2)
    we supervise each pair of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in table_transformer.py): "the naming of the
    `num_classes` parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where
    `max_obj_id` is the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass
    `num_classes` to be 91. As another example, for a dataset that has a single class with `id` 1, you should pass
    `num_classes` to be 2 (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/table_transformer/issues/108#issuecomment-650269223"


    Args:
        matcher (`TableTransformerHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        if False:
            while True:
                i = 10
        '\n        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim\n        [nb_target_boxes]\n        '
        if 'logits' not in outputs:
            raise KeyError('No logits were found in the outputs')
        source_logits = outputs['logits']
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t['class_labels'][J] for (t, (_, J)) in zip(targets, indices)])
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.\n\n        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.\n        "
        logits = outputs['logits']
        device = logits.device
        target_lengths = torch.as_tensor([len(v['class_labels']) for v in targets], device=device)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if False:
            while True:
                i = 10
        '\n        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.\n\n        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes\n        are expected in format (center_x, center_y, w, h), normalized by the image size.\n        '
        if 'pred_boxes' not in outputs:
            raise KeyError('No predicted boxes found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for (t, (_, i)) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if False:
            print('Hello World!')
        '\n        Compute the losses related to the masks: the focal loss and the dice loss.\n\n        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].\n        '
        if 'pred_masks' not in outputs:
            raise KeyError('No predicted masks found in outputs')
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs['pred_masks']
        source_masks = source_masks[source_idx]
        masks = [t['masks'] for t in targets]
        (target_masks, valid) = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]
        source_masks = nn.functional.interpolate(source_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        source_masks = source_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {'loss_mask': sigmoid_focal_loss(source_masks, target_masks, num_boxes), 'loss_dice': dice_loss(source_masks, target_masks, num_boxes)}
        return losses

    def _get_source_permutation_idx(self, indices):
        if False:
            while True:
                i = 10
        batch_idx = torch.cat([torch.full_like(source, i) for (i, (source, _)) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return (batch_idx, source_idx)

    def _get_target_permutation_idx(self, indices):
        if False:
            i = 10
            return i + 15
        batch_idx = torch.cat([torch.full_like(target, i) for (i, (_, target)) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return (batch_idx, target_idx)

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        if False:
            for i in range(10):
                print('nop')
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.loss_masks}
        if loss not in loss_map:
            raise ValueError(f'Loss {loss} not supported')
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        if False:
            for i in range(10):
                print('nop')
        "\n        This performs the loss computation.\n\n        Args:\n             outputs (`dict`, *optional*):\n                Dictionary of tensors, see the output specification of the model for the format.\n             targets (`List[dict]`, *optional*):\n                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the\n                losses applied, see each loss' doc.\n        "
        outputs_without_aux = {k: v for (k, v) in outputs.items() if k != 'auxiliary_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum((len(t['class_labels']) for t in targets))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'auxiliary_outputs' in outputs:
            for (i, auxiliary_outputs) in enumerate(outputs['auxiliary_outputs']):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for (k, v) in l_dict.items()}
                    losses.update(l_dict)
        return losses

class TableTransformerMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/table_transformer/blob/master/models/table_transformer.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList((nn.Linear(n, k) for (n, k) in zip([input_dim] + h, h + [output_dim])))

    def forward(self, x):
        if False:
            return 10
        for (i, layer) in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TableTransformerHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float=1, bbox_cost: float=1, giou_cost: float=1):
        if False:
            print('Hello World!')
        super().__init__()
        requires_backends(self, ['scipy'])
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and (giou_cost == 0):
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        if False:
            while True:
                i = 10
        '\n        Args:\n            outputs (`dict`):\n                A dictionary that contains at least these entries:\n                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits\n                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.\n            targets (`List[dict]`):\n                A list of targets (len(targets) = batch_size), where each target is a dict containing:\n                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of\n                  ground-truth\n                 objects in the target) containing the class labels\n                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.\n\n        Returns:\n            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:\n            - index_i is the indices of the selected predictions (in order)\n            - index_j is the indices of the corresponding selected targets (in order)\n            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)\n        '
        (batch_size, num_queries) = outputs['logits'].shape[:2]
        out_prob = outputs['logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        target_ids = torch.cat([v['class_labels'] for v in targets])
        target_bbox = torch.cat([v['boxes'] for v in targets])
        class_cost = -out_prob[:, target_ids]
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for (i, c) in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for (i, j) in indices]

def _upcast(t: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    '\n    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.\n\n    Args:\n        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):\n            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1\n            < x2` and `0 <= y1 < y2`.\n\n    Returns:\n        `torch.FloatTensor`: a tensor containing the area for each box.\n    '
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    if False:
        for i in range(10):
            print('nop')
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return (iou, union)

def generalized_box_iou(boxes1, boxes2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.\n\n    Returns:\n        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)\n    '
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f'boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}')
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f'boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}')
    (iou, union) = box_iou(boxes1, boxes2)
    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (bottom_right - top_left).clamp(min=0)
    area = width_height[:, :, 0] * width_height[:, :, 1]
    return iou - (area - union) / area

def _max_by_axis(the_list):
    if False:
        i = 10
        return i + 15
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for (index, item) in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        if False:
            print('Hello World!')
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        if False:
            for i in range(10):
                print('nop')
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.tensors, self.mask)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if False:
        i = 10
        return i + 15
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        (batch_size, num_channels, height, width) = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for (img, pad_img, m) in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('Only 3-dimensional tensors are supported')
    return NestedTensor(tensor, mask)