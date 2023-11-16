""" PyTorch CLAP model."""
import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'laion/clap-htsat-fused'
CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = ['laion/clap-htsat-fused', 'laion/clap-htsat-unfused']

def interpolate(hidden_states, ratio):
    if False:
        print('Hello World!')
    '\n    Interpolate data in time domain. This is used to compensate the resolution reduction in downsampling of a CNN.\n\n    Args:\n        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):\n            Input hidden states\n        ratio (`int`):\n            The ratio of the length of the output to the length of the input.\n    '
    (batch_size, time_length, classes_num) = hidden_states.shape
    upsampled = hidden_states[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_length * ratio, classes_num)
    return upsampled

def window_partition(hidden_states, window_size):
    if False:
        return 10
    '\n    Returns the resized hidden states. The output shape should be `(batch_size * num_windows, window_size, window_size,\n    num_channels)`\n\n    Args:\n        hidden_states (`torch.FloatTensor` of shape `(batch_size, height, width, num_channels)`):\n            Input hidden states\n        window_size (`int`):\n            Window size\n    '
    (batch_size, height, width, num_channels) = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, height // window_size, window_size, width // window_size, window_size, num_channels)
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows

def window_reverse(windows, window_size, height, width):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        windows (`torch.FloatTensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):\n            Input windows\n        window_size (`int`):\n            Window size\n        height (`int`):\n            Height of the resized audio\n        width (`int`):\n            Width of the resized audio\n    '
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    hidden_states = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return hidden_states

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    if False:
        while True:
            i = 10
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        x: torch.Tensor x:\n\n    Returns: torch.Tensor\n    "
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)

@dataclass
class ClapTextModelOutput(ModelOutput):
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
class ClapAudioModelOutput(ModelOutput):
    """
    ClapAudio model output to mimic the output of the original implementation.

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The Audio embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ClapOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio:(`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_audio: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return tuple((self[k] if k not in ['text_model_output', 'audio_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        if False:
            return 10
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()
        output = hidden_states.div(keep_prob) * random_tensor
        return output

class ClapAudioAFFBlock(nn.Module):
    """
    ATTENTIONAL FEATURE FUSION Block from CLAP, since in CLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """

    def __init__(self, config: ClapAudioConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        channels = config.patch_embeds_hidden_size
        downsize_ratio = config.aff_block_r
        inter_channels = int(channels // downsize_ratio)
        self.local_att = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, residual):
        if False:
            i = 10
            return i + 15
        attention_input = hidden_states + residual
        fused_layer_output = self.local_att(attention_input) + self.global_att(attention_input)
        fused_layer_output = self.sigmoid(fused_layer_output)
        output = 2 * hidden_states * fused_layer_output + 2 * residual * (1 - fused_layer_output)
        return output

class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """

    def __init__(self, config: ClapAudioConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        patch_size = (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        patch_stride = (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        self.img_size = img_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = config.flatten_patch_embeds
        self.enable_fusion = config.enable_fusion
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)
        scale_factor = 4 if self.enable_fusion and config.fusion_type == 'channel_map' else 1
        self.proj = nn.Conv2d(config.patch_embed_input_channels * scale_factor, config.patch_embeds_hidden_size, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        if self.enable_fusion:
            self.fusion_model = ClapAudioAFFBlock(config)
            self.mel_conv2d = nn.Conv2d(config.patch_embed_input_channels, config.patch_embeds_hidden_size, kernel_size=(patch_size[0], patch_size[1] * 3), stride=(patch_stride[0], patch_stride[1] * 3), padding=padding)

    def forward(self, hidden_states, is_longer_idx=None):
        if False:
            print('Hello World!')
        if self.enable_fusion:
            global_hidden_states = hidden_states[:, 0:1, :, :]
            (batch_size, num_channels, height, width) = global_hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.size(-1)
            if len(is_longer_idx) > 0:
                local_hidden_states = hidden_states[is_longer_idx, 1:, :, :].contiguous()
                (batch_size, num_channels, height, width) = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size * num_channels, 1, height, width)
                local_hidden_states = self.mel_conv2d(local_hidden_states)
                (_, features, height, width) = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size, num_channels, features, height, width)
                local_hidden_states = local_hidden_states.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                local_width = local_hidden_states.size(-1)
                local_hidden_states = torch.nn.functional.pad(local_hidden_states, (0, output_width - local_width), 'constant', 0)
                global_hidden_states[is_longer_idx] = self.fusion_model(global_hidden_states[is_longer_idx], local_hidden_states)
            hidden_states = global_hidden_states
        else:
            (_, _, height, width) = hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
            hidden_states = self.proj(hidden_states)
        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ClapAudioSelfAttention(nn.Module):

    def __init__(self, config, dim, num_heads, window_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})')
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if False:
            print('Hello World!')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        (batch_size, dim, num_channels) = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class ClapAudioSelfOutput(nn.Module):

    def __init__(self, config, dim):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ClapAudioAttention(nn.Module):

    def __init__(self, config, dim, num_heads, window_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.self = ClapAudioSelfAttention(config, dim, num_heads, window_size)
        self.output = ClapAudioSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class ClapAudioIntermediate(nn.Module):

    def __init__(self, config, dim):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ClapAudioOutput(nn.Module):

    def __init__(self, config, dim):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ClapAudioLayer(nn.Module):

    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        if False:
            print('Hello World!')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = ClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        self.drop_path = ClapDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = ClapAudioIntermediate(config, dim)
        self.output = ClapAudioOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        if False:
            while True:
                i = 10
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if False:
            print('Hello World!')
        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            width_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        if False:
            for i in range(10):
                print('nop')
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return (hidden_states, pad_values)

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, always_partition: Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        (height, width) = input_dimensions
        (batch_size, _, channels) = hidden_states.size()
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        (hidden_states, pad_values) = self.maybe_pad(hidden_states, height, width)
        (_, height_pad, width_pad, _) = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
        attention_outputs = self.attention(hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()
        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = shortcut + self.drop_path(attention_windows)
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

class ClapAudioStage(nn.Module):

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList([ClapAudioLayer(config=config, dim=dim, input_resolution=input_resolution, num_heads=num_heads, shift_size=0 if i % 2 == 0 else config.window_size // 2) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
        self.pointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, always_partition: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        (height, width) = input_dimensions
        for (i, layer_module) in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition)
            hidden_states = layer_outputs[0]
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            (height_downsampled, width_downsampled) = ((height + 1) // 2, (width + 1) // 2)
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs

class ClapAudioPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module=nn.LayerNorm) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        if False:
            return 10
        should_pad = height % 2 == 1 or width % 2 == 1
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)
        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        (height, width) = input_dimensions
        (batch_size, dim, num_channels) = input_feature.shape
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)
        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)
        return input_feature

class ClapAudioEncoder(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        self.patch_embed = ClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins
        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // 2 ** i, grid_size[1] // 2 ** i) for i in range(self.num_layers)]
        self.layers = nn.ModuleList([ClapAudioStage(config=config, dim=int(config.patch_embeds_hidden_size * 2 ** i_layer), input_resolution=self.input_resolutions[i_layer], depth=config.depths[i_layer], num_heads=config.num_attention_heads[i_layer], drop_path=drop_path_rate[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=ClapAudioPatchMerging if i_layer < self.num_layers - 1 else None) for i_layer in range(self.num_layers)])
        self.gradient_checkpointing = False
        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def reshape_mel2img(self, normalized_input_features):
        if False:
            print('Hello World!')
        '\n        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel\n        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].\n        '
        (_, _, time_length, freq_length) = normalized_input_features.shape
        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio
        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError('the wav size should be less than or equal to the swin input size')
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(normalized_input_features, (spec_width, freq_length), mode='bicubic', align_corners=True)
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(normalized_input_features, (time_length, spec_heigth), mode='bicubic', align_corners=True)
        (batch, channels, time, freq) = normalized_input_features.shape
        normalized_input_features = normalized_input_features.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, freq)
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        normalized_input_features = normalized_input_features.reshape(batch, channels, freq * self.freq_ratio, time // self.freq_ratio)
        return normalized_input_features

    def forward(self, input_features, is_longer: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, always_partition: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, ClapAudioModelOutput]:
        if False:
            while True:
                i = 10
        input_features = input_features.transpose(1, 3)
        normalized_input_features = self.batch_norm(input_features)
        normalized_input_features = normalized_input_features.transpose(1, 3)
        is_longer_list_idx = None
        if self.enable_fusion:
            is_longer_list = is_longer.to(input_features.device)
            is_longer_list_idx = torch.where(is_longer_list == 1)[0]
        hidden_states = self.reshape_mel2img(normalized_input_features)
        frames_num = hidden_states.shape[2]
        hidden_states = self.patch_embed(hidden_states, is_longer_list_idx)
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        input_dimensions = self.input_resolutions[0]
        if output_hidden_states:
            (batch_size, _, hidden_size) = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for (i, layer_module) in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            input_dimensions = self.input_resolutions[i]
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition)
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_downsampling:
                (batch_size, _, hidden_size) = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and (not output_hidden_states_before_downsampling):
                (batch_size, _, hidden_size) = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        last_hidden_state = self.norm(hidden_states)
        (batch_size, _, n_channels) = last_hidden_state.shape
        freq_shape = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[0]
        temporal_shape = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[1]
        last_hidden_state = last_hidden_state.permute(0, 2, 1).contiguous().reshape(batch_size, n_channels, freq_shape, temporal_shape)
        (batch_size, n_channels, n_frequencies, n_temp) = last_hidden_state.shape
        c_freq_bin = n_frequencies // self.freq_ratio
        last_hidden_state = last_hidden_state.reshape(batch_size, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp)
        last_hidden_state = last_hidden_state.permute(0, 1, 3, 2, 4).contiguous().reshape(batch_size, n_channels, c_freq_bin, -1)
        latent_output = self.avgpool(torch.flatten(last_hidden_state, 2))
        latent_output = torch.flatten(latent_output, 1)
        if not return_dict:
            return tuple((v for v in [last_hidden_state, latent_output, all_reshaped_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=latent_output, hidden_states=all_reshaped_hidden_states, attentions=all_self_attentions)
CLAP_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`ClapConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
CLAP_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
CLAP_AUDIO_INPUTS_DOCSTRING = '\n    Args:\n        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Input audio features. This should be returnes by the [`ClapFeatureExtractor`] class that you can also\n            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.\n        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):\n            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance\n            the features.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
CLAP_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Input audio features. This should be returnes by the [`ClapFeatureExtractor`] class that you can also\n            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class ClapProjectionLayer(nn.Module):

    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim
        self.linear1 = nn.Linear(hidden_size, projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states

class ClapTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

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
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=True)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=True)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if False:
            print('Hello World!')
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
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

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        if False:
            return 10
        '\n        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: torch.Tensor\n\n        Returns: torch.Tensor\n        '
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)

class ClapTextSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            print('Hello World!')
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
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            print('Hello World!')
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

class ClapTextSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
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

class ClapTextAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            return 10
        super().__init__()
        self.self = ClapTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ClapTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class ClapTextIntermediate(nn.Module):

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
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ClapTextOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
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

class ClapTextLayer(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ClapTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = ClapTextAttention(config, position_embedding_type='absolute')
        self.intermediate = ClapTextIntermediate(config)
        self.output = ClapTextOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
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
            print('Hello World!')
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class ClapTextEncoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ClapTextLayer(config) for _ in range(config.num_hidden_layers)])
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

class ClapTextPooler(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ClapConfig
    base_model_prefix = 'clap'
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, ClapTextEmbeddings):
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            nn.init.normal_(module.logit_scale_a, std=factor * 0.02)
            nn.init.normal_(module.logit_scale_t, std=factor * 0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            in_proj_std = self.config.hidden_size ** (-0.5) * (2 * self.config.num_hidden_layers) ** (-0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()

class ClapAudioModel(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = 'input_features'

    def __init__(self, config: ClapAudioConfig):
        if False:
            return 10
        super().__init__(config)
        self.audio_encoder = ClapAudioEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        return self.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ClapAudioConfig)
    def forward(self, input_features: Optional[torch.FloatTensor]=None, is_longer: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from datasets import load_dataset\n        >>> from transformers import AutoProcessor, ClapAudioModel\n\n        >>> dataset = load_dataset("ashraq/esc50")\n        >>> audio_sample = dataset["train"]["audio"][0]["array"]\n\n        >>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")\n        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")\n\n        >>> inputs = processor(audios=audio_sample, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return self.audio_encoder(input_features=input_features, is_longer=is_longer, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class ClapTextModel(ClapPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    config_class = ClapTextConfig

    def __init__(self, config, add_pooling_layer=True):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.embeddings = ClapTextEmbeddings(config)
        self.encoder = ClapTextEncoder(config)
        self.pooler = ClapTextPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.word_embeddings = value

    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if False:
            print('Hello World!')
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
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
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

@add_start_docstrings(CLAP_START_DOCSTRING)
class ClapModel(ClapPreTrainedModel):
    config_class = ClapConfig

    def __init__(self, config: ClapConfig):
        if False:
            return 10
        super().__init__(config)
        if not isinstance(config.text_config, ClapTextConfig):
            raise ValueError(f'config.text_config is expected to be of type ClapTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.audio_config, ClapAudioConfig):
            raise ValueError(f'config.audio_config is expected to be of type ClapAudioConfig but is of type {type(config.audio_config)}.')
        text_config = config.text_config
        audio_config = config.audio_config
        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_t = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.projection_dim = config.projection_dim
        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)
        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)
        self.post_init()

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`ClapTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, ClapModel\n\n        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")\n        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")\n\n        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(self, input_features: Optional[torch.Tensor]=None, is_longer: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        '\n        Returns:\n            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by\n            applying the projection layer to the pooled output of [`ClapAudioModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoFeatureExtractor, ClapModel\n        >>> import torch\n\n        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")\n        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")\n        >>> random_audio = torch.rand((16_000))\n        >>> inputs = feature_extractor(random_audio, return_tensors="pt")\n        >>> audio_features = model.get_audio_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, return_dict=return_dict)
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_features = self.audio_projection(pooled_output)
        audio_features = F.normalize(audio_features, dim=-1)
        return audio_features

    @add_start_docstrings_to_model_forward(CLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapOutput, config_class=ClapConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, input_features: Optional[torch.FloatTensor]=None, is_longer: Optional[torch.BoolTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from datasets import load_dataset\n        >>> from transformers import AutoProcessor, ClapModel\n\n        >>> dataset = load_dataset("ashraq/esc50")\n        >>> audio_sample = dataset["train"]["audio"][0]["array"]\n\n        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")\n        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")\n\n        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]\n\n        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)\n\n        >>> outputs = model(**inputs)\n        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score\n        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)
        text_embeds = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio
        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0
        if not return_dict:
            output = (logits_per_audio, logits_per_text, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return (loss,) + output if loss is not None else output
        return ClapOutput(loss=loss, logits_per_audio=logits_per_audio, logits_per_text=logits_per_text, text_embeds=text_embeds, audio_embeds=audio_embeds, text_model_output=text_outputs, audio_model_output=audio_outputs)

@add_start_docstrings('\n    CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).\n    ', CLAP_START_DOCSTRING)
class ClapTextModelWithProjection(ClapPreTrainedModel):
    config_class = ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        if False:
            return 10
        super().__init__(config)
        self.text_model = ClapTextModel(config)
        self.text_projection = ClapProjectionLayer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.text_model.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapTextModelOutput, config_class=ClapTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapTextModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection\n\n        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")\n        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")\n\n        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> text_embeds = outputs.text_embeds\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)
        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return ClapTextModelOutput(text_embeds=text_embeds, last_hidden_state=text_outputs.last_hidden_state, hidden_states=text_outputs.hidden_states, attentions=text_outputs.attentions)

@add_start_docstrings('\n    CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).\n    ', CLAP_START_DOCSTRING)
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = 'input_features'

    def __init__(self, config: ClapAudioConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.audio_model = ClapAudioModel(config)
        self.audio_projection = ClapProjectionLayer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            print('Hello World!')
        return self.audio_model.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapAudioModelOutput, config_class=ClapAudioConfig)
    def forward(self, input_features: Optional[torch.FloatTensor]=None, is_longer: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapAudioModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from datasets import load_dataset\n        >>> from transformers import ClapAudioModelWithProjection, ClapProcessor\n\n        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")\n        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")\n\n        >>> dataset = load_dataset("ashraq/esc50")\n        >>> audio_sample = dataset["train"]["audio"][0]["array"]\n\n        >>> inputs = processor(audios=audio_sample, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> audio_embeds = outputs.audio_embeds\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(pooled_output)
        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return ClapAudioModelOutput(audio_embeds=audio_embeds, last_hidden_state=audio_outputs.last_hidden_state, attentions=audio_outputs.attentions, hidden_states=audio_outputs.hidden_states)