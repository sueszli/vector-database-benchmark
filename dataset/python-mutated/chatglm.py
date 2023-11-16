import math
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from typing import Optional, Tuple
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache

def rotate_half(x):
    if False:
        for i in range(10):
            print('nop')
    (x1, x2) = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    if False:
        i = 10
        return i + 15
    (cos, sin) = (F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(position_id, sin.squeeze(1)).unsqueeze(2))
    (q, k) = (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)
    return (q, k)
KV_CACHE_ALLOC_BLOCK_LENGTH = 256
KV_CACHE_ALLOC_MIN_LENGTH = 512

def attention_fn(self, query_layer, key_layer, value_layer, attention_mask, hidden_size_per_partition, layer_id, layer_past=None, scaling_attention_score=True, use_cache=False):
    if False:
        return 10
    key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
    value_layer = value_layer.permute(1, 2, 0, 3).contiguous()
    (cur_length, batch_size) = (query_layer.shape[0], query_layer.shape[1])
    device = query_layer.device
    if layer_past is not None:
        (cache_k, cache_v) = (layer_past[0], layer_past[1])
        cache_k = cache_k.permute(1, 2, 0, 3)
        cache_v = cache_v.permute(1, 2, 0, 3)
        past_length = cache_k.size(2)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            max_cache_length = past_length + cur_length + KV_CACHE_ALLOC_BLOCK_LENGTH
            (new_cache_k, new_cache_v) = extend_kv_cache(batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, past_length, max_cache_length, dtype=query_layer.dtype, device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        (key_layer, value_layer) = append_kv_cache(cache_k, cache_v, key_layer, value_layer)
    elif use_cache:
        max_cache_length = max(KV_CACHE_ALLOC_MIN_LENGTH, cur_length) + KV_CACHE_ALLOC_BLOCK_LENGTH
        (key_cache, value_cache) = init_kv_cache(batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, cur_length, max_cache_length, dtype=query_layer.dtype, device=device)
        key_cache[:] = key_layer
        value_cache[:] = value_layer
        key_layer = key_cache
        value_layer = value_cache
    (b, nh, seq_len, hidden_size) = key_layer.shape
    if use_cache:
        present = (key_layer.permute(2, 0, 1, 3), value_layer.permute(2, 0, 1, 3))
    else:
        present = None
    pytorch_major_version = int(torch.__version__.split('.')[0])
    if query_layer.size(0) > 1 and pytorch_major_version >= 2:
        query_layer = query_layer.permute(1, 2, 0, 3)
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            if torch.is_autocast_cpu_enabled():
                attention_mask = torch.ones(query_layer.shape[2], key_layer.shape[2], dtype=torch.bool).tril(diagonal=0)
                attention_mask = attention_mask.masked_fill(~attention_mask, -float('inf'))
                attention_mask = attention_mask.to(torch.get_autocast_cpu_dtype())
                query_layer = query_layer.to(torch.get_autocast_cpu_dtype())
                key_layer = key_layer.to(torch.get_autocast_cpu_dtype())
                value_layer = value_layer.to(torch.get_autocast_cpu_dtype())
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, is_causal=False)
            else:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, is_causal=True)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            attention_mask = attention_mask.masked_fill(~attention_mask, -float('inf'))
            if torch.is_autocast_cpu_enabled():
                query_layer = query_layer.to(torch.get_autocast_cpu_dtype())
                key_layer = key_layer.to(torch.get_autocast_cpu_dtype())
                value_layer = value_layer.to(torch.get_autocast_cpu_dtype())
                attention_mask = attention_mask.to(torch.get_autocast_cpu_dtype())
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask)
        context_layer = context_layer.permute(2, 0, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        attention_probs = None
    else:
        query_key_layer_scaling_coeff = float(layer_id + 1)
        if scaling_attention_score:
            query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(2))
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)
        matmul_input_buffer = torch.empty(output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype, device=query_layer.device)
        matmul_result = torch.empty(output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype, device=query_layer.device)
        torch.baddbmm(matmul_input_buffer, query_layer.transpose(0, 1), key_layer.transpose(1, 2), beta=0.0, alpha=1.0, out=matmul_result)
        attention_scores = matmul_result.view(*output_size)
        if self.scale_mask_softmax:
            self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
        else:
            if not (attention_mask == 0).all():
                attention_scores.masked_fill_(attention_mask, -10000.0)
            dtype = attention_scores.dtype
            attention_scores = attention_scores.float()
            attention_scores = attention_scores * query_key_layer_scaling_coeff
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type(dtype)
        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(0), value_layer.size(3))
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.empty(output_size[0] * output_size[1], output_size[2], value_layer.size(-1), dtype=value_layer.dtype, device=query_layer.device)
        torch.bmm(attention_probs, value_layer, out=context_layer)
        context_layer = context_layer.view(*output_size)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
    outputs = (context_layer, present, attention_probs)
    return outputs

def chatglm_attention_forward(self, hidden_states: torch.Tensor, position_ids, attention_mask: torch.Tensor, layer_id, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: bool=False, output_attentions: bool=False):
    if False:
        i = 10
        return i + 15
    '\n    hidden_states: [seq_len, batch, hidden_size]\n    attention_mask: [(1, 1), seq_len, seq_len]\n    '
    mixed_raw_layer = self.query_key_value(hidden_states)
    new_tensor_shape = mixed_raw_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
    mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
    (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)
    if self.position_encoding_2d:
        (q1, q2) = query_layer.chunk(2, dim=query_layer.ndim - 1)
        (k1, k2) = key_layer.chunk(2, dim=key_layer.ndim - 1)
        (cos, sin) = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        (position_ids, block_position_ids) = (position_ids[:, 0, :].transpose(0, 1).contiguous(), position_ids[:, 1, :].transpose(0, 1).contiguous())
        (q1, k1) = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        (q2, k2) = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        query_layer = torch.concat([q1, q2], dim=q1.ndim - 1)
        key_layer = torch.concat([k1, k2], dim=k1.ndim - 1)
    else:
        position_ids = position_ids.transpose(0, 1)
        (cos, sin) = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
        (query_layer, key_layer) = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)
    (context_layer, present, attention_probs) = attention_fn(self=self, query_layer=query_layer, key_layer=key_layer, value_layer=value_layer, attention_mask=attention_mask, hidden_size_per_partition=self.hidden_size_per_partition, layer_id=layer_id, layer_past=layer_past, use_cache=use_cache)
    output = self.dense(context_layer)
    outputs = (output, present)
    if output_attentions:
        outputs += (attention_probs,)
    return outputs