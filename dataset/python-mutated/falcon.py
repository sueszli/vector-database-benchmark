"""PyTorch Falcon model."""
import math
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
KV_CACHE_ALLOC_BLOCK_LENGTH = 256

def rw_attention_forward_7b(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        for i in range(10):
            print('nop')
    fused_qkv = self.query_key_value(hidden_states)
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    (batch_size, q_length, _, _) = query_layer.shape
    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_kv, q_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_kv, q_length, self.head_dim)
    (_, seq_len, _) = query_layer.shape
    if layer_past is not None:
        (_, seq_len_past, _) = layer_past[0].shape
        seq_len = seq_len + seq_len_past
    (query_layer, key_layer) = self.maybe_rotary(query_layer, key_layer, seq_len)
    (_, kv_length, _) = key_layer.shape
    if layer_past is not None:
        kv_length += layer_past[0].shape[-2]
    query_layer = query_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.view(batch_size, self.num_kv, q_length, self.head_dim)
    value_layer = value_layer.view(batch_size, self.num_kv, q_length, self.head_dim)
    device = hidden_states.device
    if layer_past is not None:
        cache_k = layer_past[0].view(batch_size, self.num_kv, -1, self.head_dim)
        cache_v = layer_past[1].view(batch_size, self.num_kv, -1, self.head_dim)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(batch_size, self.num_kv, self.head_dim, cache_k.size(2), kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_layer, value_layer) = append_kv_cache(cache_k, cache_v, key_layer, value_layer)
    elif use_cache:
        max_cache_length = kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(batch_size, self.num_kv, self.head_dim, kv_length, max_cache_length, dtype=key_layer.dtype, device=device)
        new_key_states[:] = key_layer
        new_value_states[:] = value_layer
        key_layer = new_key_states
        value_layer = new_value_states
    query_layer = query_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    key_layer = key_layer.view(batch_size * self.num_kv, -1, self.head_dim)
    value_layer = value_layer.view(batch_size * self.num_kv, -1, self.head_dim)
    (_, kv_length, _) = key_layer.shape
    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None
    if alibi is None:
        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, self.num_kv, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, self.num_kv, -1, self.head_dim)
        if layer_past is not None:
            L = query_layer_.shape[-2]
            S = key_layer_.shape[-2]
            attn_mask = torch.ones(L, S, dtype=torch.bool, device=query_layer_.device)
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, attn_mask, 0.0, is_causal=False)
        else:
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)
        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)
        output_tensor = self.dense(attn_output)
        outputs = (output_tensor, present)
        if output_attentions:
            invalidInputError(False, f"'output_attentions' are not supported yet")
        return outputs
    else:
        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, -1000000000.0).to(torch.bfloat16)
        matmul_result = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        attention_probs = F.softmax((attention_scores + alibi) * self.inv_norm_factor + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
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

def rw_attention_forward_40b(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        for i in range(10):
            print('nop')
    fused_qkv = self.query_key_value(hidden_states)
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    (batch_size, q_length, _, _) = query_layer.shape
    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    (_, seq_len, _) = query_layer.shape
    if layer_past is not None:
        (_, seq_len_past, _) = layer_past[0].shape
        seq_len = seq_len + seq_len_past
    (query_layer, key_layer) = self.maybe_rotary(query_layer, key_layer, seq_len)
    (_, kv_length, _) = key_layer.shape
    if layer_past is not None:
        kv_length += layer_past[0].shape[-2]
    query_layer = query_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    value_layer = value_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    device = hidden_states.device
    if layer_past is not None:
        cache_k = layer_past[0].view(batch_size, self.num_heads, -1, self.head_dim)
        cache_v = layer_past[1].view(batch_size, self.num_heads, -1, self.head_dim)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(batch_size, self.num_heads, self.head_dim, cache_k.size(2), kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_layer, value_layer) = append_kv_cache(cache_k, cache_v, key_layer, value_layer)
    elif use_cache:
        max_cache_length = kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(batch_size, self.num_heads, self.head_dim, kv_length, max_cache_length, dtype=key_layer.dtype, device=device)
        new_key_states[:] = key_layer
        new_value_states[:] = value_layer
        key_layer = new_key_states
        value_layer = new_value_states
    query_layer = query_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    key_layer = key_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    value_layer = value_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    (_, kv_length, _) = key_layer.shape
    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None
    if alibi is None:
        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        if present is not None:
            L = query_layer_.shape[-2]
            S = key_layer_.shape[-2]
            attn_mask = torch.ones(L, S, dtype=torch.bool, device=query_layer_.device)
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, attn_mask, 0.0, is_causal=False)
        else:
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)
        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)
        output_tensor = self.dense(attn_output)
        outputs = (output_tensor, present)
        if output_attentions:
            invalidInputError(False, f"'output_attentions' are not supported yet")
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

def falcon_attention_forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        return 10
    fused_qkv = self.query_key_value(hidden_states)
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    (batch_size, query_length, _, _) = query_layer.shape
    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)
    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    (query_layer, key_layer) = self.maybe_rotary(query_layer, key_layer, past_kv_length)
    (_, kv_length, _) = key_layer.shape
    if layer_past is not None:
        kv_length += layer_past[0].shape[-2]
    query_layer = query_layer.view(batch_size, self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.view(batch_size, self.num_heads, query_length, self.head_dim)
    value_layer = value_layer.view(batch_size, self.num_heads, query_length, self.head_dim)
    device = hidden_states.device
    if layer_past is not None:
        cache_k = layer_past[0].view(batch_size, self.num_heads, -1, self.head_dim)
        cache_v = layer_past[1].view(batch_size, self.num_heads, -1, self.head_dim)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            (new_cache_k, new_cache_v) = extend_kv_cache(batch_size, self.num_heads, self.head_dim, cache_k.size(2), kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH, dtype=cache_k.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_layer, value_layer) = append_kv_cache(cache_k, cache_v, key_layer, value_layer)
    elif use_cache:
        max_cache_length = kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH
        (new_key_states, new_value_states) = init_kv_cache(batch_size, self.num_heads, self.head_dim, kv_length, max_cache_length, dtype=key_layer.dtype, device=device)
        new_key_states[:] = key_layer
        new_value_states[:] = value_layer
        key_layer = new_key_states
        value_layer = new_value_states
    query_layer = query_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    key_layer = key_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    value_layer = value_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    (_, kv_length, _) = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None
    attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float('-1e9')).to(query_layer.dtype)
    query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
    key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
    value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
    if alibi is None:
        if output_attentions:
            attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
            attention_scores /= math.sqrt(self.head_dim)
            attention_scores = F.softmax(attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            attn_output = attention_scores @ value_layer_
        else:
            attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False)
            attention_scores = None
        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        output_tensor = self.dense(attn_output)
        if output_attentions:
            return (output_tensor, present, attention_scores)
        else:
            return (output_tensor, present)
    else:
        matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)
        attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
        attention_logits *= self.inv_norm_factor
        attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)
        context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)
        context_layer = self._merge_heads(context_layer)
        output_tensor = self.dense(context_layer)
        if output_attentions:
            return (output_tensor, present, attention_probs)
        else:
            return (output_tensor, present)