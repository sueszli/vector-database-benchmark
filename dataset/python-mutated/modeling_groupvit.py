""" PyTorch GroupViT model."""
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'nvidia/groupvit-gcc-yfcc'
GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['nvidia/groupvit-gcc-yfcc']

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def groupvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def hard_softmax(logits: torch.Tensor, dim: int):
    if False:
        i = 10
        return i + 15
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float=1, hard: bool=False, dim: int=-1) -> torch.Tensor:
    if False:
        while True:
            i = 10
    gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0, device=logits.device, dtype=logits.dtype), torch.tensor(1.0, device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def resize_attention_map(attentions, height, width, align_corners=False):
    if False:
        return 10
    '\n    Args:\n        attentions (`torch.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]\n        height (`int`): height of the output attention map\n        width (`int`): width of the output attention map\n        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.\n\n    Returns:\n        `torch.Tensor`: resized attention map of shape [batch_size, groups, height, width]\n    '
    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = attentions.shape[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = attentions.shape[2] // feat_height
    batch_size = attentions.shape[0]
    groups = attentions.shape[1]
    attentions = attentions.reshape(batch_size, groups, feat_height, feat_width)
    attentions = nn.functional.interpolate(attentions, size=(height, width), mode='bilinear', align_corners=align_corners)
    return attentions

def get_grouping_from_attentions(attentions, hw_shape):
    if False:
        for i in range(10):
            print('nop')
    '\n    Args:\n        attentions (`tuple(torch.FloatTensor)`: tuple of attention maps returned by `GroupViTVisionTransformer`\n        hw_shape (`tuple(int)`): height and width of the output attention map\n    Returns:\n        `torch.Tensor`: the attention map of shape [batch_size, groups, height, width]\n    '
    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for attn_masks in attentions:
            attn_masks = attn_masks.permute(0, 2, 1).contiguous()
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            cur_attn_map = resize_attention_map(prev_attn_masks.permute(0, 2, 1).contiguous(), *hw_shape)
            attn_maps.append(cur_attn_map)
    final_grouping = attn_maps[-1]
    return final_grouping

class GroupViTCrossAttentionLayer(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.attn = GroupViTAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.norm_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, query, key):
        if False:
            i = 10
            return i + 15
        x = query
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        x = x + self.mlp(self.norm2(x))
        x = self.norm_post(x)
        return x

class GroupViTAssignAttention(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.scale = config.hidden_size ** (-0.5)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.assign_eps = config.assign_eps

    def get_attn(self, attn, gumbel=True, hard=True):
        if False:
            print('Hello World!')
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        elif hard:
            attn = hard_softmax(attn, dim=-2)
        else:
            attn = nn.functional.softmax(attn, dim=-2)
        return attn

    def forward(self, query, key):
        if False:
            while True:
                i = 10
        value = key
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        raw_attn = query @ key.transpose(-2, -1) * self.scale
        attn = self.get_attn(raw_attn)
        soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        out = attn @ value
        out = self.proj(out)
        return (out, soft_attn)

class GroupViTTokenAssign(nn.Module):

    def __init__(self, config: GroupViTVisionConfig, num_group_token, num_output_group):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_output_group = num_output_group
        self.norm_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        assign_mlp_ratio = config.assign_mlp_ratio if isinstance(config.assign_mlp_ratio, collections.abc.Iterable) else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        (tokens_dim, channels_dim) = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        self.mlp_inter = GroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_assign_attn = GroupViTCrossAttentionLayer(config)
        self.assign = GroupViTAssignAttention(config)
        self.norm_new_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_channels = GroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size)

    def project_group_token(self, group_tokens):
        if False:
            while True:
                i = 10
        '\n        Args:\n            group_tokens (torch.Tensor): group tokens, [batch_size, num_group_tokens, channels]\n\n        Returns:\n            projected_group_tokens (torch.Tensor): [batch_size, num_output_groups, channels]\n        '
        projected_group_tokens = self.mlp_inter(group_tokens)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, image_tokens, group_tokens):
        if False:
            while True:
                i = 10
        '\n        Args:\n            image_tokens (`torch.Tensor`): image tokens, of shape [batch_size, input_length, channels]\n            group_tokens (`torch.Tensor`): group tokens, [batch_size, num_group_tokens, channels]\n        '
        group_tokens = self.norm_tokens(group_tokens)
        image_tokens = self.norm_x(image_tokens)
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        (new_image_tokens, attention) = self.assign(projected_group_tokens, image_tokens)
        new_image_tokens += projected_group_tokens
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))
        return (new_image_tokens, attention)

@dataclass
class GroupViTModelOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    segmentation_logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            return 10
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class GroupViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size: int=224, patch_size: Union[int, Tuple[int, int]]=16, num_channels: int=3, embed_dim: int=768):
        if False:
            print('Hello World!')
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool=False) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (batch_size, num_channels, height, width) = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

class GroupViTVisionEmbeddings(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.patch_embeddings = GroupViTPatchEmbeddings(image_size=config.image_size, patch_size=config.patch_size, num_channels=config.num_channels, embed_dim=config.hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if False:
            return 10
        '\n        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher\n        resolution images.\n\n        Source:\n        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174\n        '
        npatch = embeddings.shape[1]
        if npatch == self.position_embeddings.shape[1] and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        num_original_pos_embed = patch_pos_embed.shape[1]
        dim = embeddings.shape[-1]
        feat_height = height // self.config.patch_size
        feat_width = width // self.config.patch_size
        (feat_height, feat_width) = (feat_height + 0.1, feat_width + 0.1)
        original_height = original_width = math.sqrt(num_original_pos_embed)
        reshaped_patch_pos_embed = patch_pos_embed.reshape(1, int(original_height), int(original_width), dim).permute(0, 3, 1, 2)
        scale_factor = (feat_height / original_height, feat_width / original_width)
        patch_pos_embed = nn.functional.interpolate(reshaped_patch_pos_embed, scale_factor=scale_factor, mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool=False) -> torch.Tensor:
        if False:
            print('Hello World!')
        (batch_size, num_channels, height, width) = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.layernorm(embeddings)
        (batch_size, seq_len, _) = embeddings.size()
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class GroupViTTextEmbeddings(nn.Module):

    def __init__(self, config: GroupViTTextConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> torch.Tensor:
        if False:
            return 10
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings

class GroupViTStage(nn.Module):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(self, config: GroupViTVisionConfig, depth: int, num_prev_group_token: int, num_group_token: int, num_output_group: int):
        if False:
            while True:
                i = 10
        super().__init__()
        self.depth = depth
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, config.hidden_size))
        else:
            self.group_token = None
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(depth)])
        if num_group_token > 0:
            self.downsample = GroupViTTokenAssign(config=config, num_group_token=num_group_token, num_output_group=num_output_group)
        else:
            self.downsample = None
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = nn.Sequential(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps), GroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token))
        else:
            self.group_projector = None

    @property
    def with_group_token(self):
        if False:
            for i in range(10):
                print('nop')
        return self.group_token is not None

    def split_x(self, x):
        if False:
            return 10
        if self.with_group_token:
            return (x[:, :-self.num_group_token], x[:, -self.num_group_token:])
        else:
            return (x, None)

    def concat_x(self, x: torch.Tensor, group_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, hidden_states: torch.Tensor, prev_group_token: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the grouping tensors of Grouping block.\n        '
        if self.with_group_token:
            group_token = self.group_token.expand(hidden_states.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None
        x = hidden_states
        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None)
            cat_x = layer_out[0]
        (x, group_token) = self.split_x(cat_x)
        attention = None
        if self.downsample is not None:
            (x, attention) = self.downsample(x, group_token)
        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)
        return outputs

class GroupViTMLP(nn.Module):

    def __init__(self, config: GroupViTVisionConfig, hidden_size: Optional[int]=None, intermediate_size: Optional[int]=None, output_size: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        output_size = output_size if output_size is not None else hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class GroupViTMixerMLP(GroupViTMLP):

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = super().forward(x.transpose(1, 2))
        return x.transpose(1, 2)

class GroupViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            for i in range(10):
                print('nop')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            i = 10
            return i + 15
        'Input shape: Batch x Time x Channel'
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        is_cross_attention = encoder_hidden_states is not None
        query_states = self.q_proj(hidden_states) * self.scale
        if is_cross_attention:
            key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

class GroupViTEncoderLayer(nn.Module):

    def __init__(self, config: GroupViTConfig):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GroupViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class GroupViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GroupViTConfig
    base_model_prefix = 'groupvit'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        init_range = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        factor = self.config.initializer_factor
        if isinstance(module, GroupViTTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, GroupViTAttention):
            factor = self.config.initializer_factor
            in_proj_std = module.embed_dim ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            out_proj_std = module.embed_dim ** (-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, GroupViTMLP):
            factor = self.config.initializer_factor
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
GROUPVIT_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GROUPVIT_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
GROUPVIT_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
GROUPVIT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class GroupViTVisionEncoder(nn.Module):

    def __init__(self, config: GroupViTVisionConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList([GroupViTStage(config=config, depth=config.depths[i], num_group_token=config.num_group_tokens[i], num_output_group=config.num_output_groups[i], num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0) for i in range(len(config.depths))])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None
        group_tokens = None
        for (i, stage) in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)
            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings)

class GroupViTTextEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self-attention layers. Each layer is a
    [`GroupViTEncoderLayer`].

    Args:
        config: GroupViTTextConfig
    """

    def __init__(self, config: GroupViTTextConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            while True:
                i = 10
        "\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, causal_attention_mask, output_attentions)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class GroupViTTextTransformer(nn.Module):

    def __init__(self, config: GroupViTTextConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = GroupViTTextEmbeddings(config)
        self.encoder = GroupViTTextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id

    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            raise ValueError('You have to specify input_ids')
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)]
        else:
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1)]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class GroupViTTextModel(GroupViTPreTrainedModel):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.text_model = GroupViTTextTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            i = 10
            return i + 15
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import CLIPTokenizer, GroupViTTextModel\n\n        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class GroupViTVisionTransformer(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = GroupViTVisionEmbeddings(config)
        self.encoder = GroupViTVisionEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(hidden_states=hidden_states, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = last_hidden_state.mean(dim=1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class GroupViTVisionModel(GroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: GroupViTVisionConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.vision_model = GroupViTVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> GroupViTPatchEmbeddings:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, GroupViTVisionModel\n\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled CLS states\n        ```'
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class GroupViTModel(GroupViTPreTrainedModel):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(f'config.text_config is expected to be of type GroupViTTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type GroupViTVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = GroupViTTextTransformer(text_config)
        self.vision_model = GroupViTVisionTransformer(vision_config)
        self.visual_projection = nn.Sequential(nn.Linear(self.vision_embed_dim, self.projection_intermediate_dim, bias=True), nn.BatchNorm1d(self.projection_intermediate_dim), nn.ReLU(inplace=True), nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True))
        self.text_projection = nn.Sequential(nn.Linear(self.text_embed_dim, self.projection_intermediate_dim, bias=True), nn.BatchNorm1d(self.projection_intermediate_dim), nn.ReLU(inplace=True), nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True))
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.post_init()

    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`GroupViTTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import CLIPTokenizer, GroupViTModel\n\n        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            return 10
        '\n        Returns:\n            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by\n            applying the projection layer to the pooled output of [`GroupViTVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, GroupViTModel\n\n        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroupViTModelOutput, config_class=GroupViTConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_segmentation: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, GroupViTModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, GroupViTModel\n\n        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")\n        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True\n        ... )\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_segmentation = output_segmentation if output_segmentation is not None else self.config.output_segmentation
        if output_segmentation:
            output_attentions = True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        seg_logits = None
        if output_segmentation:
            image_group_embeds = vision_outputs[0]
            image_group_embeds = self.visual_projection(image_group_embeds.reshape(-1, image_group_embeds.shape[-1]))
            if output_hidden_states:
                attentions = vision_outputs[3]
            else:
                attentions = vision_outputs[2]
            grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])
            image_group_embeds = image_group_embeds / image_group_embeds.norm(dim=-1, keepdim=True)
            logits_per_image_group = torch.matmul(image_group_embeds, text_embeds.t()) * logit_scale
            logits_per_image_group = logits_per_image_group.reshape(image_embeds.shape[0], -1, text_embeds.shape[0]).permute(0, 2, 1)
            flatten_grouping = grouping.reshape(grouping.shape[0], grouping.shape[1], -1)
            seg_logits = torch.matmul(logits_per_image_group, flatten_grouping) * logit_scale
            seg_logits = seg_logits.reshape(seg_logits.shape[0], seg_logits.shape[1], grouping.shape[2], grouping.shape[3])
        loss = None
        if return_loss:
            loss = groupvit_loss(logits_per_text)
        if not return_dict:
            if seg_logits is not None:
                output = (logits_per_image, logits_per_text, seg_logits, text_embeds, image_embeds, text_outputs, vision_outputs)
            else:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return GroupViTModelOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, segmentation_logits=seg_logits, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)