import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.utils import logging
from bigdl.llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half
KV_CACHE_ALLOC_BLOCK_LENGTH = 256

def apply_rotary_pos_emb(t, freqs):
    if False:
        while True:
            i = 10
    (cos, sin) = freqs
    rot_dim = freqs[0].shape[-1]
    (t_, t_pass_) = (t[..., :rot_dim], t[..., rot_dim:])
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = t_ * cos + rotate_half(t_) * sin
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)

def qwen_attention_forward_vl(self, hidden_states: Optional[Tuple[torch.FloatTensor]], rotary_pos_emb: Optional[List[torch.Tensor]]=None, registered_causal_mask: Optional[torch.Tensor]=None, layer_past: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=False):
    if False:
        print('Hello World!')
    mixed_x_layer = self.c_attn(hidden_states)
    (query, key, value) = mixed_x_layer.split(self.split_size, dim=2)
    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)
    kv_seq_len = hidden_states.size()[1]
    if rotary_pos_emb is not None:
        cur_len = query.shape[1]
        rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
        rotary_pos_emb = (rotary_pos_emb,) * 2
        (q_pos_emb, k_pos_emb) = rotary_pos_emb
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)
    (bsz, _, n_heads, head_dim) = key.size()
    if layer_past is not None:
        kv_seq_len += layer_past[0].shape[1]
        cache_k = layer_past[0].transpose(1, 2)
        cache_v = layer_past[1].transpose(1, 2)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(bsz, self.num_heads, self.head_dim, cache_k.size(2), kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=hidden_states.device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_states, value_states) = append_kv_cache(cache_k, cache_v, key.transpose(1, 2), value.transpose(1, 2))
        key = key_states.transpose(1, 2)
        value = value_states.transpose(1, 2)
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(bsz, self.num_heads, self.head_dim, kv_seq_len, max_cache_length, dtype=key.dtype, device=hidden_states.device)
        new_key_states[:] = key.transpose(1, 2)
        new_value_states[:] = value.transpose(1, 2)
        key = new_key_states.transpose(1, 2)
        value = new_value_states.transpose(1, 2)
    if use_cache:
        present = (key, value)
    else:
        present = None
    if self.use_logn_attn and (not self.training):
        if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
            self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
        seq_start = key.size(1) - query.size(1)
        seq_end = key.size(1)
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
        query = query * logn_tensor.expand_as(query)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    (attn_output, attn_weight) = self._attn(query, key, value, registered_causal_mask, attention_mask, head_mask)
    context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(context_layer)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weight,)
    return outputs