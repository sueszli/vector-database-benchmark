import torch
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import torch.nn.functional as F
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
KV_CACHE_ALLOC_BLOCK_LENGTH = 256
KV_CACHE_ALLOC_MIN_LENGTH = 512

def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool=False) -> List[torch.Tensor]:
    if False:
        return 10
    'Split a tensor along its last dimension.\n    Arguments:\n        tensor: input tensor.\n        num_partitions: number of partitions to split the tensor\n        contiguous_split_chunks: If True, make each chunk contiguous\n                                 in memory.\n    Returns:\n        A list of Tensors\n    '
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple((chunk.contiguous() for chunk in tensor_list))
    return tensor_list

@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    (sq, b, np, hn) = (x.size(0), x.size(1), x.size(2), x.size(3))
    rot_dim = rope_cache.shape[-2] * 2
    (x, x_pass) = (x[..., :rot_dim], x[..., rot_dim:])
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

def chatglm2_32k_attention_forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
    if False:
        for i in range(10):
            print('nop')
    device = hidden_states.device
    mixed_x_layer = self.query_key_value(hidden_states)
    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split([self.num_attention_heads_per_partition * self.hidden_size_per_attention_head, self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head, self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head], dim=-1)
        query_layer = query_layer.view(query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
        key_layer = key_layer.view(key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head))
        value_layer = value_layer.view(value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head))
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
    (cur_length, batch_size) = (query_layer.shape[0], query_layer.shape[1])
    if self.multi_query_attention:
        key_length = key_layer.size(0)
        query_group_size = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
        key_layer = key_layer.permute(1, 2, 0, 3).unsqueeze(-3)
        key_layer = key_layer.expand(-1, -1, query_group_size, -1, -1)
        key_layer = key_layer.contiguous().view((batch_size, self.num_attention_heads_per_partition, key_length, self.hidden_size_per_attention_head))
        value_layer = value_layer.permute(1, 2, 0, 3).unsqueeze(-3)
        value_layer = value_layer.expand(-1, -1, query_group_size, -1, -1)
        value_layer = value_layer.contiguous().view((batch_size, self.num_attention_heads_per_partition, key_length, self.hidden_size_per_attention_head))
    if kv_cache is not None:
        (cache_k, cache_v) = kv_cache
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
    key_layer = key_layer.permute(2, 0, 1, 3)
    value_layer = value_layer.permute(2, 0, 1, 3)
    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None
    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    output = self.dense(context_layer)
    return (output, kv_cache)