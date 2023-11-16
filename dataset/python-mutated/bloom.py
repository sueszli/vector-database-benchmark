"""PyTorch BLOOM model."""
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
KV_CACHE_ALLOC_BLOCK_LENGTH = 256

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool):
    if False:
        i = 10
        return i + 15
    '\n    Dropout add function\n\n    Args:\n        x (`torch.tensor`, *required*):\n            input tensor\n        residual (`torch.tensor`, *required*):\n            residual tensor\n        prob (`float`, *required*):\n            dropout probability\n        training (`bool`, *required*):\n            training mode\n    '
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def bloom_attention_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        i = 10
        return i + 15
    fused_qkv = self.query_key_value(hidden_states)
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    (batch_size, q_length, _, _) = query_layer.shape
    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    (_, _, kv_length) = key_layer.shape
    if layer_past is not None:
        kv_length += layer_past[0].shape[-1]
    query_layer = query_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).view(batch_size, self.num_heads, q_length, self.head_dim)
    value_layer = value_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    device = hidden_states.device
    if layer_past is not None:
        cache_k = layer_past[0].transpose(1, 2).view(batch_size, self.num_heads, -1, self.head_dim)
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
    key_layer = key_layer.view(batch_size * self.num_heads, -1, self.head_dim).transpose(1, 2)
    value_layer = value_layer.view(batch_size * self.num_heads, -1, self.head_dim)
    (_, _, kv_length) = key_layer.shape
    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None
    matmul_result = alibi.baddbmm(batch1=query_layer, batch2=key_layer, beta=self.beta, alpha=self.inv_norm_factor)
    attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
    input_dtype = attention_scores.dtype
    if input_dtype == torch.float16:
        attention_scores = attention_scores.to(torch.float)
    attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
    attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
    attention_probs = self.attention_dropout(attention_probs)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
    context_layer = torch.bmm(attention_probs_reshaped, value_layer)
    context_layer = self._merge_heads(context_layer)
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(context_layer[:, :, int(i * slices):int((i + 1) * slices)], self.dense.weight[:, int(i * slices):int((i + 1) * slices)])
    else:
        output_tensor = self.dense(context_layer)
    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
    outputs = (output_tensor, present)
    if output_attentions:
        outputs += (attention_probs,)
    return outputs