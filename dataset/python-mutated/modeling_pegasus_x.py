""" PyTorch PEGASUS-X model."""
import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_pegasus_x import PegasusXConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/pegasus-x-base'
_CONFIG_FOR_DOC = 'PegasusXConfig'
PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/pegasus-x-base', 'google/pegasus-x-large']

@dataclasses.dataclass
class DimensionInfo:
    """Wrapper for dimension info."""
    batch_size: int
    seq_len: int
    block_size: int
    num_heads: int
    hidden_dim: int
    dim_per_head: int
    num_blocks: int
    global_len: int
    padded_seq_len: int

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shift input ids one token to the right.\n    '
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError('self.model.config.pad_token_id has to be defined.')
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embed_dim, max_scale: int=10000.0):
        if False:
            return 10
        super().__init__()
        self.embed_dim = embed_dim
        self.max_scale = max_scale

    @torch.no_grad()
    def forward(self, input_embeds: torch.Tensor, past_key_values_length: int=0) -> torch.Tensor:
        if False:
            print('Hello World!')
        '`input_ids_shape` is expected to be [bsz x seqlen].'
        (batch_size, seq_len) = input_embeds.shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_embeds.device)[:, None]
        pe = torch.zeros((seq_len, self.embed_dim), device=input_embeds.device, dtype=input_embeds.dtype)
        half_d_feature = self.embed_dim // 2
        div_term = torch.exp(torch.arange(half_d_feature, device=input_embeds.device, dtype=input_embeds.dtype) * -(np.log(float(self.max_scale)) / (half_d_feature - 1)))
        pe[:, :half_d_feature] = torch.sin(positions * div_term)
        pe[:, half_d_feature:] = torch.cos(positions * div_term)
        return pe[None].expand(batch_size, -1, -1)

class PegasusXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[PegasusXConfig]=None):
        if False:
            return 10
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            while True:
                i = 10
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class PegasusXGlobalLocalAttention(nn.Module):
    """Global + Local attention. For use with Encoder only."""

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float=0.0, is_decoder: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            for i in range(10):
                print('nop')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, token_hidden_states: torch.Tensor, global_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if False:
            while True:
                i = 10
        'Input shape: Batch x Time x Channel'
        dim = DimensionInfo(batch_size=token_hidden_states.shape[0], seq_len=token_hidden_states.shape[1], block_size=self.block_size, num_heads=self.num_heads, hidden_dim=token_hidden_states.shape[2], dim_per_head=self.head_dim, num_blocks=token_hidden_states.shape[1] // self.block_size, global_len=global_hidden_states.shape[1], padded_seq_len=token_hidden_states.shape[1])
        local_q = self._shape(self.q_proj(token_hidden_states) * self.scaling, seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        local_k = self._shape(self.k_proj(token_hidden_states), seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        local_v = self._shape(self.v_proj(token_hidden_states), seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        global_q = self._shape(self.q_proj(global_hidden_states) * self.scaling, seq_len=dim.global_len, bsz=dim.batch_size)
        global_k = self._shape(self.k_proj(global_hidden_states), seq_len=dim.global_len, bsz=dim.batch_size)
        global_v = self._shape(self.v_proj(global_hidden_states), seq_len=dim.global_len, bsz=dim.batch_size)
        (global_attn_output, global_attn_probs) = self.compute_global_attention_representations(global_q=global_q, global_k=global_k, global_v=global_v, local_k=local_k, local_v=local_v, mask=attention_mask, dim=dim)
        (local_attn_output, local_attn_probs) = self.compute_local_attention_representations(global_k=global_k, global_v=global_v, local_q=local_q, local_k=local_k, local_v=local_v, mask=attention_mask, dim=dim)
        global_attn_output = global_attn_output.transpose(1, 2).contiguous().view(dim.batch_size, dim.global_len, dim.hidden_dim)
        global_attn_output = self.out_proj(global_attn_output)
        local_attn_output = local_attn_output.permute(0, 2, 3, 1, 4).contiguous()
        local_attn_output = local_attn_output.view(dim.batch_size, dim.padded_seq_len, dim.hidden_dim)
        local_attn_output = self.out_proj(local_attn_output)
        if output_attentions:
            attn_probs = {'global': global_attn_probs, 'local': local_attn_probs}
        else:
            attn_probs = None
        return (local_attn_output, global_attn_output, attn_probs)

    def compute_global_attention_representations(self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo):
        if False:
            for i in range(10):
                print('nop')
        'Compute attention representations for global tokens.\n\n        Global tokens will attend to both global tokens as well as all input sequence tokens. Because the input\n        sequence tokens are arranged in blocks for local attention, we unblock them and compute attention.\n\n        Args:\n            global_q (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:\n                query vectors from global tokens\n            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:\n                key vectors from global tokens\n            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:\n                value vectors from global tokens\n            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:\n                key vectors from local tokens\n            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:\n                value vectors from local tokens\n            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask\n            dim (DimensionInfo): DimensionInfo wrapper for dimensions\n\n        Returns:\n            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size\n        '
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        global_and_local_v = torch.cat([global_v, local_v], dim=2)
        extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)
        attn_weights = torch.einsum('BHGF,BHXF->BHGX', global_q, global_and_local_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_output = torch.einsum('BHGX,BHXF->BHGF', attn_probs, global_and_local_v)
        return (attn_output, attn_probs)

    def compute_local_attention_representations(self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo):
        if False:
            for i in range(10):
                print('nop')
        'Compute attention representations for local tokens.\n\n        Local tokens will attend to both global tokens as well as all other tokens within the same local block. Hence,\n        we need to tile and concatenate the global tokens to every local block\n\n        Args:\n            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:\n                key vectors from global tokens\n            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:\n                value vectors from global tokens\n            local_q (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:\n                query vectors from local tokens\n            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:\n                key vectors from local tokens\n            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:\n                value vectors from local tokens\n            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask\n            dim (DimensionInfo): DimensionInfo wrapper for dimensions\n\n        Returns:\n            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size\n        '
        blocked_local_q = local_q.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        blocked_local_k = local_k.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        blocked_local_v = local_v.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        extended_mask = nn.functional.pad(mask.view(dim.batch_size, dim.num_blocks, dim.block_size), pad=(dim.global_len, 0), value=0)
        blocked_local2global = torch.einsum('BHNKF,BHGF->BHNKG', blocked_local_q, global_k)
        blocked_local2local = torch.einsum('BHNKF,BHNXF->BHNKX', blocked_local_q, blocked_local_k)
        attn_weights = torch.cat([blocked_local2global, blocked_local2local], dim=-1)
        attn_weights = attn_weights + extended_mask[:, None, :, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        local2global_attn_probs = attn_probs[:, :, :, :, :dim.global_len]
        local2local_attn_probs = attn_probs[:, :, :, :, dim.global_len:]
        local2global_attn_output = torch.einsum('BHNKG,BHGF->BHNKF', local2global_attn_probs, global_v)
        local2local_attn_output = torch.einsum('BHNKX,BHNXF->BHNKF', local2local_attn_probs, blocked_local_v)
        attn_output = local2global_attn_output + local2local_attn_output
        return (attn_output, attn_probs)

class PegasusXEncoderLayer(nn.Module):

    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = PegasusXGlobalLocalAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, block_size=config.block_size, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.global_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.stagger_blocks_this_layer = stagger_blocks_this_layer
        self.block_size = config.block_size

    def forward(self, hidden_states: torch.Tensor, global_hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool=False) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*\n            global_hidden_states (`torch.FloatTensor`): global token hidden states\n                *(seq_len, num_global_tokens, embed_dim)*\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        global_residual = global_hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        global_hidden_states = self.global_self_attn_layer_norm(global_hidden_states)
        if self.stagger_blocks_this_layer:
            (hidden_states, attention_mask) = self.pad_local_tokens(hidden_states=hidden_states, attention_mask=attention_mask, block_size=self.block_size)
        (hidden_states, global_hidden_states, attn_weights) = self.self_attn(token_hidden_states=hidden_states, global_hidden_states=global_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        if self.stagger_blocks_this_layer:
            hidden_states = self.unpad_local_tokens(padded_hidden_states=hidden_states, block_size=self.block_size)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        global_residual = global_hidden_states
        global_hidden_states = self.final_layer_norm(global_hidden_states)
        global_hidden_states = self.activation_fn(self.fc1(global_hidden_states))
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.activation_dropout, training=self.training)
        global_hidden_states = self.fc2(global_hidden_states)
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states
        outputs = (hidden_states, global_hidden_states)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
        if False:
            for i in range(10):
                print('nop')
        pad_size = block_size // 2
        mask_min_value = torch.finfo(hidden_states.dtype).min
        padded_hidden_states = torch.nn.functional.pad(hidden_states, pad=(0, 0, pad_size, pad_size))
        padded_mask = torch.nn.functional.pad(attention_mask, pad=(pad_size, pad_size), value=mask_min_value)
        return (padded_hidden_states, padded_mask)

    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size):
        if False:
            print('Hello World!')
        pad_size = block_size // 2
        return padded_hidden_states[:, pad_size:-pad_size, :]

class PegasusXDecoderLayer(nn.Module):

    def __init__(self, config: PegasusXConfig):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = PegasusXAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True, bias=False)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = PegasusXAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True, bias=False)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape *(seq_len, batch, embed_dim)*\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            use_cache: Whether to us KV cache for decoding\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value
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
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class PegasusXPreTrainedModel(PreTrainedModel):
    config_class = PegasusXConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['PegasusXEncoderLayer', 'PegasusXDecoderLayer']

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
PEGASUS_X_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`PegasusXConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
PEGASUS_X_GENERATION_EXAMPLE = '\n    Summarization example:\n\n    ```python\n    >>> from transformers import AutoTokenizer, PegasusXForConditionalGeneration\n\n    >>> model = PegasusXForConditionalGeneration.from_pretrained("google/pegasus-x-base")\n    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")\n\n    >>> ARTICLE_TO_SUMMARIZE = (\n    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "\n    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "\n    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."\n    ... )\n    >>> inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")\n\n    >>> # Generate Summary\n    >>> summary_ids = model.generate(inputs["input_ids"])\n    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]\n    "California\'s largest electricity provider has turned off power to hundreds of thousands of customers."\n    ```\n'
PEGASUS_X_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            PEGASUS-X uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If\n            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape\n            `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you\n            can choose to directly pass an embedded representation. This is useful if you want more control over how to\n            convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class PegasusXEncoder(PegasusXPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusXEncoderLayer`].

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding]=None):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)
        self.embed_global = nn.Embedding(config.num_global_tokens, embed_dim)
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([PegasusXEncoderLayer(stagger_blocks_this_layer=i % 2 == 1 and config.stagger_local_blocks, config=config) for i in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            return 10
        '\n        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=\n        config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embeddings. If position embeddings are learned, increasing the size will add\n                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If\n                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will\n                add correct vectors at the end following the position encoding algorithm, whereas reducing the size\n                will remove vectors from the end.\n        '
        logger.info(f'Setting `config.max_position_embeddings={new_num_position_embeddings}`...')
        self.config.max_position_embeddings = new_num_position_embeddings
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(self.config.d_model)
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            print('Hello World!')
        '\n        Returns the position embeddings matrix\n        '
        return self.embed_positions

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(inputs_embeds)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        (batch_size, seq_len, _) = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(*input_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        mask_min_value = torch.finfo(hidden_states.dtype).min
        inverted_mask = 1.0 - attention_mask
        attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_min_value)
        if seq_len % self.config.block_size != 0:
            pad_len = self.config.block_size - seq_len % self.config.block_size
            hidden_states = nn.functional.pad(hidden_states, pad=(0, 0, 0, pad_len), value=0)
            attention_mask = nn.functional.pad(attention_mask, pad=(0, pad_len), value=mask_min_value)
        global_hidden_states = self.embed_global(torch.arange(self.config.num_global_tokens, device=hidden_states.device)[None].expand(batch_size, -1))
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (idx, encoder_layer) in enumerate(self.layers):
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
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, global_hidden_states, attention_mask, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, global_hidden_states, attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
                global_hidden_states = layer_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
        hidden_states = hidden_states[:, :seq_len]
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + ((hidden_states, global_hidden_states),)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class PegasusXDecoder(PegasusXPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding]=None):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(config.d_model)
        self.layers = nn.ModuleList([PegasusXDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            inputs_embeds (`torch.FloatTensor` of\n                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing\n                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more\n                control over how to convert `input_ids` indices into associated vectors than the model's internal\n                embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(inputs_embeds, past_key_values_length)
        positions = positions.to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top.', PEGASUS_X_START_DOCSTRING)
class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']

    def __init__(self, config: PegasusXConfig):
        if False:
            return 10
        super().__init__(config)
        vocab_size = config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model)
        self.encoder = PegasusXEncoder(config, self.shared)
        self.decoder = PegasusXDecoder(config, self.shared)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.shared

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            while True:
                i = 10
        '\n        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=\n        config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embeddings. If position embeddings are learned, increasing the size will add\n                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If\n                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will\n                add correct vectors at the end following the position encoding algorithm, whereas reducing the size\n                will remove vectors from the end.\n        '
        self.config.max_position_embeddings = new_num_position_embeddings
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the position embeddings matrix\n        '
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Seq2SeqModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, PegasusModel\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")\n        >>> model = PegasusModel.from_pretrained("google/pegasus-x-large")\n\n        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")\n        >>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")\n        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)\n\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 4, 1024]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

@add_start_docstrings('The PEGASUS-X for conditional generation (e.g. summarization).', PEGASUS_X_START_DOCSTRING)
class PegasusXForConditionalGeneration(PegasusXPreTrainedModel):
    base_model_prefix = 'model'
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: PegasusXConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.model = PegasusXModel(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.post_init()

    def get_encoder(self):
        if False:
            print('Hello World!')
        return self.model.get_encoder()

    def get_decoder(self):
        if False:
            print('Hello World!')
        return self.model.get_decoder()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            return 10
        '\n        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=\n        config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embeddings. If position embeddings are learned, increasing the size will add\n                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If\n                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will\n                add correct vectors at the end following the position encoding algorithm, whereas reducing the size\n                will remove vectors from the end.\n        '
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        if False:
            print('Hello World!')
        '\n        Returns the position embeddings matrix\n        '
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(PEGASUS_X_GENERATION_EXAMPLE)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Seq2SeqLMOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            while True:
                i = 10
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            while True:
                i = 10
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.decoder = PegasusXDecoder(config)

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.decoder(*args, **kwargs)