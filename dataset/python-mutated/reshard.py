import copy
from typing import List, Tuple
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec._internals import get_split_size, get_chunked_dim_size
from torch.distributed.nn.functional import all_to_all, all_to_all_single
from torch.distributed._shard.metadata import ShardMetadata
from .shard import Shard

def get_idx_from_placements(placements, current_rank) -> int:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the position of the current rank in the given placements.\n\n    Args:\n        placements(List[Union[_remote_device, str]]):\n            Specifies the placement of each shard of the Tensor. The size of\n            the list represents the number of shards to be created. This could\n            be a list of\n            :class:`torch.distributed._remote_device`'s. This list\n            could also contain a string which represents remote\n            device as accepted by\n            :class:`torch.distributed._remote_device`\n        current_rank (int): number of current device.\n\n    Returns:\n        A int which contains the position of current device in the placement list.\n    "
    for (idx, placement) in enumerate(placements):
        if current_rank == placement.rank():
            return idx
    raise RuntimeError('current_rank not in the placement.')

def build_reshard_metadata(st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, world_size: int) -> Tuple[List[ShardMetadata], List[int]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Based the given sharding spec, we calculate the offset and local shard size.\n    We then build a ShardMetadata on top of the calculation result.\n\n    Args:\n        st_size (torch.Size): The size of the sharded tensor.\n        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The\n            specification describing how the tensor is sharded.\n        world_size (int): number of ranks.\n\n    Returns:\n        A Tuple of the followings:\n            A List[`ShardMetadata`] which contains the metadata for the shard, including\n                offsets, lengths and device placement.\n            A List[int] which contains the ranks in the order of placement.\n    '
    shard_dim = int(sharding_spec.dim)
    shards_metadata = [None] * world_size
    ranks = []
    offsets = [0] * len(st_size)
    split_size = get_split_size(st_size[shard_dim], world_size)
    for (idx, placement) in enumerate(sharding_spec.placements):
        ranks.append(placement.rank())
        sharded_dim_size = get_chunked_dim_size(st_size[shard_dim], split_size, idx)
        local_tensor_size = list(st_size)
        local_tensor_size[shard_dim] = sharded_dim_size
        shards_metadata[placement.rank()] = ShardMetadata(shard_offsets=copy.deepcopy(offsets), shard_sizes=local_tensor_size, placement=placement)
        offsets[shard_dim] += sharded_dim_size
    return (shards_metadata, ranks)

def reshuffle_local_shard(local_shard: torch.Tensor, st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, resharding_spec: shard_spec.ShardingSpec, pg: ProcessGroup) -> Tuple[List[Shard], List[ShardMetadata]]:
    if False:
        return 10
    '\n    Reshuffle the local shard directly when the reshard dim is same as the original\n    sharding dim. Logically we do this in two step:\n    1. To collect all shards based on original sharding spec.\n    2. Reshard the tensor based on the given resharding spec.\n\n    In reality, we consolidate the two steps into one by sending the local tensor to\n    the new shard directly based on the resharding spec.\n\n    Args:\n        local_shard (Tensor): Local tensor stored in the current rank.\n        st_size (torch.Size): The size of the sharded tensor.\n        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The\n            specification describing how the tensor is sharded originally.\n        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The\n            specification describing how the tensor will be resharded.\n        pg (ProcessGroup): The process group to aggregate on.\n\n    Returns:\n        A Tuple of the followings:\n            A List[`Shard`] which contains the local tensor and its metadata.\n            A List[`ShardMetadata`] which contains the metadata for the shard, including\n                offsets, lengths and device placement.\n    '
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    (shards_metadata, ranks) = build_reshard_metadata(st_size, resharding_spec, world_size)
    reshard_dim = int(resharding_spec.dim)
    split_size = get_split_size(st_size[reshard_dim], world_size)
    input_split_sizes = [0] * world_size
    idx = get_idx_from_placements(sharding_spec.placements, current_rank)
    new_rank = resharding_spec.placements[idx].rank()
    input_split_sizes[new_rank] = local_shard.size(reshard_dim)
    output_split_sizes = [0] * world_size
    new_idx = ranks.index(current_rank)
    sharded_dim_size = get_chunked_dim_size(st_size[reshard_dim], split_size, new_idx)
    output_split_sizes[new_rank] = sharded_dim_size
    local_shard = local_shard.transpose(0, reshard_dim).contiguous()
    gathered_input_size = list(local_shard.size())
    gathered_input_size[0] = sharded_dim_size
    gathered_input = torch.empty(gathered_input_size, device=local_shard.device, dtype=local_shard.dtype)
    local_shard = all_to_all_single(gathered_input, local_shard, input_split_sizes=input_split_sizes, output_split_sizes=output_split_sizes, group=pg)
    local_tensor = local_shard.transpose(0, reshard_dim).contiguous()
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return (local_shards, shards_metadata)

def reshard_local_shard(local_tensor: torch.Tensor, st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, resharding_spec: shard_spec.ShardingSpec, pg: ProcessGroup) -> Tuple[List[Shard], List[ShardMetadata]]:
    if False:
        while True:
            i = 10
    '\n    Reshard a sharded tensor given the ``resharding_spec``. When the reshard dim is\n    different from the original sharding dim, we need to do two steps logically:\n    1. To collect all shards based on original sharding spec.\n    2. Reshard the tensor based on the given resharding spec.\n\n    In reality, we consolidate the two steps into one by sending each rank the new\n    shard based on the resharding spec.\n\n    Args:\n        local_tensor (Tensor): Local tensor stored in the current rank.\n        st_size (torch.Size): The size of the sharded tensor.\n        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The\n            specification describing how the tensor is sharded originally.\n        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The\n            specification describing how the tensor will be resharded.\n        pg (ProcessGroup): The process group to aggregate on.\n\n    Returns:\n        A Tuple of the followings:\n            A List[`Shard`] which contains the local tensor and its metadata.\n            A List[`ShardMetadata`] which contains the metadata for the shard, including\n                offsets, lengths and device placement.\n    '
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    current_sharding_dim = int(sharding_spec.dim)
    reshard_dim = int(resharding_spec.dim)
    (shards_metadata, ranks) = build_reshard_metadata(st_size, resharding_spec, world_size)
    input_split_sizes = []
    for metadata in shards_metadata:
        input_split_sizes.append(metadata.shard_sizes[reshard_dim])
    rearrange_input = any((ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1)))
    if rearrange_input:
        indices: List[int] = []
        for metadata in shards_metadata:
            offset_start_idx = metadata.shard_offsets[reshard_dim]
            split_size = metadata.shard_sizes[reshard_dim]
            indices += range(offset_start_idx, offset_start_idx + split_size)
        local_tensor = local_tensor.index_select(reshard_dim, torch.tensor(indices, device=local_tensor.device))
    output_tensor_list = [torch.tensor(1)] * world_size
    split_size = get_split_size(st_size[current_sharding_dim], world_size)
    rearrange_output_list = False
    indices = []
    for (idx, placement) in enumerate(sharding_spec.placements):
        sharded_dim_size = get_chunked_dim_size(st_size[current_sharding_dim], split_size, idx)
        output_tensor_size = list(st_size)
        output_tensor_size[current_sharding_dim] = sharded_dim_size
        output_tensor_size[reshard_dim] = input_split_sizes[current_rank]
        output_tensor_list[placement.rank()] = torch.empty(output_tensor_size, device=local_tensor.device, dtype=local_tensor.dtype)
        indices.append(placement.rank())
        if idx != placement.rank():
            rearrange_output_list = True
    input_tensor_tuple = torch.split(local_tensor, input_split_sizes, dim=reshard_dim)
    input_tensor_list = [tensor.contiguous() for tensor in input_tensor_tuple]
    output_tensor_list = all_to_all(output_tensor_list, input_tensor_list, group=pg)
    if rearrange_output_list:
        output_tensor_list = [output_tensor_list[idx] for idx in indices]
    local_tensor = torch.cat(output_tensor_list, dim=current_sharding_dim)
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return (local_shards, shards_metadata)