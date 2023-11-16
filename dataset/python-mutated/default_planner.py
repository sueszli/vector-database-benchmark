import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import SavePlanner, LoadPlanner, SavePlan, LoadPlan, ReadItem, WriteItem, WriteItemType
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, ChunkStorageMetadata, TensorStorageMetadata, MetadataIndex, Metadata, STATE_DICT_TYPE, STORAGE_TYPES
from torch.distributed.checkpoint.planner_helpers import _create_read_items, _create_write_items, _create_default_metadata_only_plan
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING, flatten_state_dict
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
logger: logging.Logger = logging.getLogger(__name__)
__all__ = ['DefaultSavePlanner', 'DefaultLoadPlanner', 'create_default_local_load_plan', 'create_default_global_load_plan', 'create_default_local_save_plan', 'create_default_global_save_plan']

class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING

    def __init__(self, flatten_state_dict: bool=True, flatten_sharded_tensors: bool=True, dedup_replicated_tensors: bool=True) -> None:
        if False:
            while True:
                i = 10
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = dedup_replicated_tensors
        self.mappings = {}

    def set_up_planner(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        if False:
            while True:
                i = 10
        if self.flatten_state_dict:
            (state_dict, self.mappings) = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        if False:
            for i in range(10):
                print('nop')
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan
        return self.plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        if False:
            return 10
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)
        (global_plan, metadata) = create_default_global_save_plan(all_plans)
        if self.flatten_state_dict:
            planner_data_dict = [p.planner_data for p in global_plan]
            merged_mappings = dict(ChainMap(*planner_data_dict))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)
        if not _validate_global_plan(global_plan, metadata):
            raise ValueError('Failed to validate global plan')
        self.global_plan = global_plan
        self.metadata = metadata
        return (self.global_plan, self.metadata)

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        if False:
            i = 10
            return i + 15
        self.plan = new_plan
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        if False:
            while True:
                i = 10
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        if False:
            print('Hello World!')
        'Extension from the planner interface to make it easy to extend the default planner.'
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        if False:
            for i in range(10):
                print('nop')
        'Extension from the planner interface to make it easy to extend the default planner.'
        if write_item.type == WriteItemType.BYTE_IO:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object

class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    """
    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(self, flatten_state_dict: bool=True, flatten_sharded_tensors: bool=True) -> None:
        if False:
            while True:
                i = 10
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}

    def set_up_planner(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.original_state_dict = state_dict
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        if self.flatten_state_dict:
            (state_dict, self.mappings) = flatten_state_dict(state_dict)
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        if False:
            i = 10
            return i + 15
        return create_default_local_load_plan(self.state_dict, self.metadata)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        if False:
            for i in range(10):
                print('nop')
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        if False:
            i = 10
            return i + 15
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.flatten_state_dict:
            set_element(self.original_state_dict, self.mappings[read_item.dest_index.fqn], torch.load(value))
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        if False:
            i = 10
            return i + 15
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'Extension from the planner interface to make it easy to extend the default planner.'
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        if False:
            i = 10
            return i + 15
        'Extension from the planner interface to make it easy to extend the default planner.'
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)

def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata) -> LoadPlan:
    if False:
        return 10
    requests = []
    '\n    Create the ``LoadPlan`` used by DefaultLoadPlanner.\n\n    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.\n\n    The default behavior is to match key exactly between state_dict and metadata.\n    It handles resharding by issuing multiple read requests against storage in order to match\n    load requirements.\n    '
    for (fqn, obj) in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)
    return LoadPlan(requests)

def create_default_global_load_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    if False:
        return 10
    "\n    Create global load plan used by DefaultLoadPlanner.\n\n    The default load behavior involved no global coordination and this function\n    currently doesn't change the local plans.\n    "
    return all_plans

def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create the ``SavePlan`` used by DefaultSavePlanner.\n\n    On non-coordinator ranks, this function ignores tensors and non-tensor objects,\n    only producing writes for ShardedTensor objects.\n\n    On the coordinator rank, produce writes for all values.\n    '
    requests = []
    for (fqn, obj) in state_dict.items():
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, torch.Tensor) or is_coordinator:
            requests += _create_write_items(fqn, obj)
    return SavePlan(requests)

def create_default_global_save_plan(all_plans: List[SavePlan], rewrite_index_hints: bool=True) -> Tuple[List[SavePlan], Metadata]:
    if False:
        print('Hello World!')
    '\n    Create the global plan and metadata used by DefaultSavePlanner.\n\n    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.\n\n    The only global planning change is to update index hints in all ``MetadataIndex`` objects if\n    ``rewrite_index_hints`` is True.\n    '
    md: Dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md
            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(TensorStorageMetadata, md.setdefault(item.index.fqn, TensorStorageMetadata(properties=item.tensor_data.properties, size=item.tensor_data.size, chunks=[])))
                new_item = item
                if rewrite_index_hints:
                    new_index = dataclasses.replace(item.index, index=len(tensor_md.chunks))
                    new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)
                assert item.tensor_data.chunk is not None, f'\n                    Cannot create MD for tensor without bounds.\n                    FQN: {item.index.fqn}\n                '
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))

def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    if False:
        for i in range(10):
            print('nop')
    'Return the ``Metadata`` if DefaultSavePlanner was used to checkpoint ``state_dict``.'
    plan = _create_default_metadata_only_plan(state_dict)
    (_, md) = create_default_global_save_plan([plan])
    return md

def _check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if two boxes overlap. Tuples are (offset, lengths).'
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False
    return True

def _check_box_bounds(outer_box_size: torch.Size, inner_box: ChunkStorageMetadata) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False
    return True

def _validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    if False:
        return 10
    all_good = True
    for (key, value) in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for (chunk_idx, chunk0) in enumerate(value.chunks):
            if not _check_box_bounds(value.size, chunk0):
                logger.warning('\n                        key:%s has out of bounds chunk:\n                        tensor-size:%s chunk: %s\n                    ', key, value.size, chunk0)
                all_good = False
            chunks_volume += reduce(operator.mul, chunk0.sizes, 1)
            for chunk1 in value.chunks[chunk_idx + 1:]:
                if _check_box_overlap(chunk0, chunk1):
                    logger.warning('key:%s has overlapping chunks: %s %s', key, chunk0, chunk1)
                    all_good = False
        tensor_volume = reduce(operator.mul, value.size, 1)
        if chunks_volume != tensor_volume:
            logger.warning('\n                    key:%s invalid fill tensor-volume:\n                    %s chunks-volume: %s\n                ', key, tensor_volume, chunks_volume)
            all_good = False
    return all_good