import torch
from typing import Optional, Tuple
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
KV_CACHE_ALLOC_BLOCK_LENGTH = 256

def gptneox_attention_forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, position_ids: torch.LongTensor, head_mask: Optional[torch.FloatTensor]=None, layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False):
    if False:
        return 10
    (bsz, q_len, _) = hidden_states.size()
    device = hidden_states.device
    has_layer_past = layer_past is not None
    qkv = self.query_key_value(hidden_states)
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)
    query = qkv[..., :self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size:2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)
    query_rot = query[..., :self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims:]
    key_rot = key[..., :self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims:]
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    use_fuse_rope = query.device.type == 'xpu'
    use_fuse_rope = use_fuse_rope and (not (self.training and query.requires_grad))
    if use_fuse_rope:
        (query, key) = apply_rotary_pos_emb_no_cache_xpu(query_rot, key_rot, position_ids, 'gpt_neox')
    else:
        (cos, sin) = self.rotary_emb(value, seq_len=seq_len)
        (query, key) = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids, 'gpt_neox')
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        if past_key.stride()[1] <= past_key.size(2) * past_key.size(3):
            (new_past_key, new_past_value) = extend_kv_cache(bsz, self.num_attention_heads, self.head_size, past_key.size(2), seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=past_key.dtype, device=device)
            new_past_key[:] = past_key
            new_past_value[:] = past_value
            past_key = new_past_key
            past_value = new_past_value
        (key, value) = append_kv_cache(past_key, past_value, key, value)
    elif use_cache:
        max_cache_length = seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key, new_value) = init_kv_cache(bsz, self.num_attention_heads, self.head_size, seq_len, max_cache_length, dtype=key.dtype, device=device)
        new_key[:] = key
        new_value[:] = value
        key = new_key
        value = new_value
    present = (key, value) if use_cache else None
    (attn_output, attn_weights) = self._attn(query, key, value, attention_mask, head_mask)
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    attn_output = self.dense(attn_output)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs