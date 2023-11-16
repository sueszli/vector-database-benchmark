from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb, rotate_half, GPTNeoXAttention
import types
__all__ = ['enable_gpt_neox_pos_shift_attention']

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    if False:
        while True:
            i = 10
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    x_embed = x * cos + rotate_half(x) * sin
    return x_embed

def gpt_neox_pos_shift_attention_forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, position_ids: torch.LongTensor, head_mask: Optional[torch.FloatTensor]=None, layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False):
    if False:
        print('Hello World!')
    has_layer_past = layer_past is not None
    qkv = self.query_key_value(hidden_states)
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)
    query = qkv[..., :self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size:2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)
    query_rot = query[..., :self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims:]
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    (cos, sin) = self.rotary_emb(value, seq_len=seq_len)
    query = apply_rotary_pos_emb_single(query_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    present = (key, value) if use_cache else None
    key_rot = key[..., :self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims:]
    key_position_ids = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
    key = apply_rotary_pos_emb_single(key_rot, cos, sin, key_position_ids)
    key = torch.cat((key, key_pass), dim=-1)
    (attn_output, attn_weights) = self._attn(query, key, value, attention_mask, head_mask)
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    attn_output = self.dense(attn_output)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs

def enable_gpt_neox_pos_shift_attention(model):
    if False:
        for i in range(10):
            print('nop')
    for (name, module) in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gpt_neox_pos_shift_attention(module)
        if isinstance(module, GPTNeoXAttention):
            module.forward = types.MethodType(gpt_neox_pos_shift_attention_forward, module)