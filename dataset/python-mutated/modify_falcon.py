from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import types
from transformers.models.falcon.modeling_falcon import FalconAttention, rotate_half
__all__ = ['enable_falcon_pos_shift_attention']

def falcon_pos_shift_attention_forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        for i in range(10):
            print('nop')
    fused_qkv = self.query_key_value(hidden_states)
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    (batch_size, q_length, _, _) = query_layer.shape
    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    num_kv = self.num_heads if self.num_heads == 128 else self.num_kv
    key_layer = key_layer.transpose(1, 2).reshape(batch_size * num_kv, q_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv, q_length, self.head_dim)
    past_len = 0
    if layer_past is not None:
        past_len = layer_past[0].shape[1]
    query_layer_copy = query_layer.clone()
    (query_layer, _) = self.maybe_rotary(query_layer, query_layer_copy, past_len)
    if layer_past is not None:
        (past_key, past_value) = layer_past
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)
    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None
    key_layer_copy = key_layer.clone()
    (_, key_layer) = self.maybe_rotary(key_layer_copy, key_layer, 0)
    (_, kv_length, _) = key_layer.shape
    if alibi is None:
        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv, -1, self.head_dim)
        if layer_past is not None:
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=False)
        else:
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)
        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)
        output_tensor = self.dense(attn_output)
        outputs = (output_tensor, present)
        assert not output_attentions
        return outputs
    else:
        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, -1000000000.0).to(torch.bfloat16)
        matmul_result = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        attention_probs = F.softmax((attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)) * self.inv_norm_factor + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
        context_layer = attention_probs_reshaped @ value_layer
        context_layer = self._merge_heads(context_layer)
        output_tensor = self.dense(context_layer)
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs

def enable_falcon_pos_shift_attention(model):
    if False:
        i = 10
        return i + 15
    for (name, module) in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_falcon_pos_shift_attention(module)
        if 'self_attention' == name[-14:]:
            model._modules[name].forward = types.MethodType(falcon_pos_shift_attention_forward, model._modules[name])