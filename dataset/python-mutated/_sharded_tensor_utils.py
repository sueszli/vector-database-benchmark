import copy
import torch.distributed as dist
from torch.distributed.remote_device import _remote_device
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata, ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
from ._traverse import OBJ_PATH, traverse_state_dict, set_element, STATE_DICT_ITEM
from .utils import _element_wise_add, _normalize_device_info

def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    if False:
        return 10
    "\n    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.\n\n    The resulting ShardedTensor instances are only correct regarding the local shard and\n    MUST not be used for any other purpose but checkpointing, as no operator will work with them.\n\n    This function should be used in conjunction with a state_dict produced by FSDP's\n    StateDictType.SHARDED_STATE_DICT methods.\n    "
    new_state_dict: STATE_DICT_TYPE = {}

    def rewrite_dict(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        shards = value.local_shards()
        if len(shards) == 0:
            return
        if len(shards) != 1:
            set_element(new_state_dict, path, value)
            return
        outer_shard = shards[0]
        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        if len(inner_st.local_shards()) != 1:
            raise ValueError('Cannot handle inner tensor with more than 1 shard')
        inner_shard = inner_st.local_shards()[0]
        local_shards = [Shard(tensor=inner_shard.tensor, metadata=ShardMetadata(shard_offsets=_element_wise_add(outer_shard.metadata.shard_offsets, inner_shard.metadata.shard_offsets), shard_sizes=inner_shard.metadata.shard_sizes, placement=f'rank:{dist.get_rank()}/{inner_shard.tensor.device}'))]
        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        device_info = _normalize_device_info(inner_shard.tensor.device.type, 0)
        for (i, shard_md) in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break
        for shard_md in st_meta.shards_metadata:
            shard_md.placement = _remote_device(f'rank:{other_rank}/{device_info}')
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(ShardMetadata(shard_offsets=_element_wise_add(outer_shard.metadata.shard_offsets, inner_md.shard_offsets), shard_sizes=inner_md.shard_sizes, placement=f'rank:{other_rank}/{device_info}'))
        st_meta.shards_metadata.append(local_shards[0].metadata)
        st = ShardedTensor._init_from_local_shards_and_global_metadata(local_shards=local_shards, sharded_tensor_metadata=st_meta)
        set_element(new_state_dict, path, st)
    traverse_state_dict(state_dict, rewrite_dict)
    return new_state_dict