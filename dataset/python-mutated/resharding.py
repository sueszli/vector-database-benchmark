from typing import List, Tuple
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
__all__: List[str] = []

def _check_shard_metadata_pair_overlap(shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata):
    if False:
        while True:
            i = 10
    'Check if two shards overlap.'
    ndims = len(shard1.offsets)
    for i in range(ndims):
        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:
            return False
        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:
            return False
    return True

def _shards_get_overlap_region_wrt_saved_tensor(saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata) -> List[Tuple[int, int, int, int]]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the overlapping region between saved_shard and current_shard.\n\n    There returned list has the same number of elements as the tensor's dimension.\n    For each element, we produce a tuple with the following contents:\n        (dimension, `saved_shard` offset, `current_shard` offset, length)\n\n    Offsets are relative to each shard.\n    "
    narrows = []
    for (dim, (saved_shard_offset, current_shard_offset, saved_shard_size, current_shard_size)) in enumerate(zip(saved_shard.offsets, current_shard.offsets, saved_shard.sizes, current_shard.sizes)):
        min_range_end = min(saved_shard_offset + saved_shard_size, current_shard_offset + current_shard_size)
        length = min_range_end - max(current_shard_offset, saved_shard_offset)
        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0
        narrows.append((dim, offset_for_saved_tensor, offset_for_current_tensor, length))
    return narrows