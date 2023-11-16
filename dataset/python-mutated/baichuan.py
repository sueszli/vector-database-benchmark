import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half, apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
KV_CACHE_ALLOC_BLOCK_LENGTH = 256

def baichuan_attention_forward_7b(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if False:
        for i in range(10):
            print('nop')
    (bsz, q_len, _) = hidden_states.size()
    device = hidden_states.device
    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    if query_states.device.type == 'xpu' and (not (self.training and query_states.requires_grad)):
        (query_states, key_states) = apply_rotary_pos_emb_no_cache_xpu(query_states, key_states, position_ids, 'baichuan')
    else:
        (cos, sin) = self.rotary_emb(value_states, seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, 'baichuan')
    if past_key_value is not None:
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(bsz, self.num_heads, self.head_dim, cache_k.size(2), kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_states, value_states) = append_kv_cache(cache_k, cache_v, key_states, value_states)
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(bsz, self.num_heads, self.head_dim, kv_seq_len, max_cache_length, dtype=key_states.dtype, device=device)
        new_key_states[:] = key_states
        new_value_states[:] = value_states
        key_states = new_key_states
        value_states = new_value_states
    past_key_value = (key_states, value_states) if use_cache else None
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        invalidInputError(False, f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}')
    if attention_mask is not None:
        invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len), f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}')
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    invalidInputError(attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim), f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},but is {attn_output.size()}')
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return (attn_output, attn_weights, past_key_value)

def baichuan_attention_forward_13b(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if False:
        i = 10
        return i + 15
    (bsz, q_len, _) = hidden_states.size()
    device = hidden_states.device
    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    if past_key_value is not None:
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(bsz, self.num_heads, self.head_dim, cache_k.size(2), kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_states, value_states) = append_kv_cache(cache_k, cache_v, key_states, value_states)
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(bsz, self.num_heads, self.head_dim, kv_seq_len, max_cache_length, dtype=key_states.dtype, device=device)
        new_key_states[:] = key_states
        new_value_states[:] = value_states
        key_states = new_key_states
        value_states = new_value_states
    past_key_value = (key_states, value_states) if use_cache else None
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        if q_len == 1:
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return (attn_output, attn_weights, past_key_value)