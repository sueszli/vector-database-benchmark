from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata

def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks if two shards overlap.\n    '
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_sizes[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_sizes[i]:
            return False
    return True

def _find_nd_overlapping_shards(shards: List[ShardMetadata], sharded_dims: List[int]) -> Optional[Tuple[int, int]]:
    if False:
        return 10
    shard_intervals = [[(s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1) for dim in sharded_dims] for s in shards]
    for i in range(len(shards)):
        shard_i = shard_intervals[i]
        for j in range(i + 1, len(shards)):
            shard_j = shard_intervals[j]
            overlap = True
            for (interval_i, interval_j) in zip(shard_i, shard_j):
                if interval_i[0] > interval_j[1] or interval_j[0] > interval_i[1]:
                    overlap = False
                    break
            if overlap:
                return (i, j)
    return None

def _find_1d_overlapping_shards(shards: List[ShardMetadata], dim: int) -> Optional[Tuple[int, int]]:
    if False:
        return 10
    intervals = [(s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i) for (i, s) in enumerate(shards)]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None

def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    if False:
        return 10
    "\n    Ensures none of the shards overlap with each other.\n\n    Args:\n        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing\n            each shard.\n    Raises:\n        ``ValueError`` if there's overlap in any two shards.\n    "
    if not shards or len(shards) == 1:
        return
    sharded_dims: List[int] = []
    for dim in range(len(shards[0].shard_offsets)):
        for i in range(1, len(shards)):
            if shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim] or shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]:
                sharded_dims.append(dim)
                break
    pair: Optional[Tuple[int, int]] = None
    if len(sharded_dims) == 0:
        pair = (0, 1)
    elif len(sharded_dims) == 1:
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    else:
        pair = _find_nd_overlapping_shards(shards, sharded_dims)
    if pair:
        raise ValueError(f'Shards {shards[pair[0]]} and {shards[pair[1]]} overlap')

def check_tensor(shards_metadata, tensor_dims) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Checks if the shards_metadata is compatible with the provided tensor dims.\n\n    Args:\n        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`\n            objects representing each shard of the tensor.\n        tensor_dims(Sequence of int): Dimensions of tensor to verify\n    Raises:\n        ``ValueError`` if not compatible.\n    '
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(f'Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}')
    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for (i, shard_length) in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(f'Shard offset {shard.shard_offsets[i]} and length {shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}')
        total_shard_volume += shard_volume
    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size
    if total_shard_volume != tensor_volume:
        raise ValueError(f'Total volume of shards: {total_shard_volume} does not match tensor volume: {tensor_volume}, in other words all the individual shards do not cover the entire tensor')

def get_split_size(dim_size, chunks):
    if False:
        while True:
            i = 10
    '\n    Computes the split size inline with ``torch.chunk``\n\n    Args:\n        dim_size(int): Size of the dimension being chunked.\n        chunks(int): Number of chunks to create for ``dim_size``.\n\n    Returns:\n        An int indicating the split size to use.\n    '
    return (dim_size + chunks - 1) // chunks

def get_chunked_dim_size(dim_size, split_size, idx):
    if False:
        print('Hello World!')
    '\n    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``\n    and ``split_size``.\n\n    Args:\n        dim_size(int): Size of the dimension being chunked.\n        split_size(int): The chunk size for each chunk of ``dim_size``.\n        idx(int): The index of chunk whose dim size is being requested.\n\n    Returns:\n        An int indicating the dim size of the chunk.\n    '
    return max(min(dim_size, split_size * (idx + 1)) - split_size * idx, 0)

def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    if False:
        print('Hello World!')
    '\n    Generate the start pos and offset length for the current rank for\n    chunk sharding.\n\n    Args:\n        sharding_dim_size(int): The dimension length which we shard on.\n        world_size(int): number of ranks.\n        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):\n            sharding spec.\n        rank(int): # of cuda process.\n\n    Returns:\n        start_pos(int): start position of sharded tensor on the given rank.\n        chunk_size(int): chunk size of sharded tensor on the given rank.\n    '
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    for (idx, placement) in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return (start_pos, chunk_size)