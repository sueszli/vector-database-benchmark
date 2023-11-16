import torch
import importlib
import torch.nn as nn
from typing import Optional, Tuple
import math
import torch.nn.functional as F
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half, apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states\n    go from (batch, num_key_value_heads, seqlen, head_dim) to\n    (batch, num_attention_heads, seqlen, head_dim)\n    '
    (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
KV_CACHE_ALLOC_BLOCK_LENGTH = 256
_ipex_version = None

def get_ipex_version():
    if False:
        for i in range(10):
            print('nop')
    global _ipex_version
    if _ipex_version is not None:
        return _ipex_version
    import intel_extension_for_pytorch as ipex
    _ipex_version = ipex.__version__
    return _ipex_version

def llama_rms_norm_forward(self, hidden_states):
    if False:
        while True:
            i = 10
    if hidden_states.device.type == 'xpu' and (not (self.training and hidden_states.requires_grad)):
        if get_ipex_version() <= '2.0.110+xpu':
            (hidden_states, _) = torch.ops.torch_ipex.rms_norm(hidden_states, [self.weight.size(0)], self.weight)
        else:
            (hidden_states, _) = torch.ops.torch_ipex.fast_rms_norm(hidden_states, [self.weight.size(0)], self.weight, None, self.variance_epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    return hidden_states

def llama_attention_forward_4_31(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False, padding_mask: Optional[torch.LongTensor]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if False:
        print('Hello World!')
    (bsz, q_len, _) = hidden_states.size()
    device = hidden_states.device
    if self.config.pretraining_tp > 1:
        key_value_slicing = self.num_key_value_heads * self.head_dim // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(self.num_heads * self.head_dim // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)
        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)
        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    use_fuse_rope = query_states.device.type == 'xpu'
    use_fuse_rope = use_fuse_rope and (not (self.training and query_states.requires_grad))
    use_fuse_rope = use_fuse_rope and self.config.rope_scaling is None
    if use_fuse_rope:
        (query_states, key_states) = apply_rotary_pos_emb_no_cache_xpu(query_states, key_states, position_ids, 'llama')
    else:
        (cos, sin) = self.rotary_emb(value_states, seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, 'llama')
    if past_key_value is not None:
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(bsz, self.num_key_value_heads, self.head_dim, cache_k.size(2), kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_states, value_states) = append_kv_cache(cache_k, cache_v, key_states, value_states)
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(bsz, self.num_key_value_heads, self.head_dim, kv_seq_len, max_cache_length, dtype=key_states.dtype, device=device)
        new_key_states[:] = key_states
        new_value_states[:] = value_states
        key_states = new_key_states
        value_states = new_value_states
    past_key_value = (key_states, value_states) if use_cache else None
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(device, dtype=hidden_states.dtype)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(device, dtype=hidden_states.dtype)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    attn_weights_size = (bsz, self.num_heads, q_len, kv_seq_len)
    if attn_weights.size() != attn_weights_size:
        invalidInputError(False, f'Attention weights should be of size {attn_weights_size}, but is {attn_weights.size()}')
    if attention_mask is not None:
        attn_mask_size = (bsz, 1, q_len, kv_seq_len)
        if attention_mask.size() != attn_mask_size:
            invalidInputError(False, f'Attention mask should be of size {attn_mask_size}, but is {attention_mask.size()}')
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False, f'`attn_output` should be of size {attn_output_size}, but is {attn_output.size()}')
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return (attn_output, attn_weights, past_key_value)