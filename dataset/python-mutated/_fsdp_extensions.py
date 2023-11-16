from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._shard_utils import _all_gather_dtensor, _create_chunk_dtensor, _create_chunk_sharded_tensor

class FSDPExtensions(ABC):
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """

    @abstractmethod
    def pre_flatten_transform(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[Any]]:
        if False:
            i = 10
            return i + 15
        'E.g. converting ``DistributedTensor`` to local tensor.'
        ...

    @abstractmethod
    def post_unflatten_transform(self, tensor: torch.Tensor, param_extension: Any) -> torch.Tensor:
        if False:
            print('Hello World!')
        'E.g. converting local tensor to ``DistributedTensor``.'
        ...

    @abstractmethod
    def chunk_tensor(self, tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, device: Optional[torch.device]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Shards a tensor to chunks and returns the local chunk.'
        ...

    @abstractmethod
    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Shards a tensor/DTensor to DTensor and returns the local DTensor.'
        ...

    @abstractmethod
    def pre_load_state_dict_transform(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[Shard]]:
        if False:
            return 10
        '\n        This is to be called before loading a *sharded* model state dict and\n        should return the tensor and list of shards from which to load data.\n        '
        ...

    @abstractmethod
    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: Optional[DeviceMesh]) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        This is to be called before loading a *sharded* DTensor state dict.\n        This gathers tensor in FSDP dimension and returns local tensor of\n        TP DTensor.\n        '
        ...
_extensions: Optional[FSDPExtensions] = None

def _set_fsdp_extensions(flattener: FSDPExtensions) -> None:
    if False:
        print('Hello World!')
    global _extensions
    _extensions = flattener

def _ext_pre_flatten_transform(tensor: torch.Tensor, fsdp_extension: Optional[FSDPExtensions]=None) -> Tuple[torch.Tensor, Optional[Any]]:
    if False:
        i = 10
        return i + 15
    if fsdp_extension is not None:
        (new_tensor, param_extension) = fsdp_extension.pre_flatten_transform(tensor)
        if param_extension is not None:
            return (new_tensor, param_extension)
    return (tensor, None)

def _ext_post_unflatten_transform(tensor: torch.Tensor, param_extension: Any, fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    if False:
        print('Hello World!')
    if fsdp_extension is not None and param_extension is not None:
        return fsdp_extension.post_unflatten_transform(tensor, param_extension)
    return tensor

def _ext_chunk_tensor(tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    if False:
        while True:
            i = 10
    chunk_tensor_fn = fsdp_extension.chunk_tensor if fsdp_extension is not None else _create_chunk_sharded_tensor
    return chunk_tensor_fn(tensor, rank, world_size, num_devices_per_node, pg)

def _ext_chunk_dtensor(tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh, fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    chunk_dtensor_fn = fsdp_extension.chunk_dtensor if fsdp_extension is not None else _create_chunk_dtensor
    return chunk_dtensor_fn(tensor, rank, device_mesh)

def _ext_pre_load_state_dict_transform(tensor: torch.Tensor, fsdp_extension: Optional[FSDPExtensions]=None) -> Tuple[torch.Tensor, List[Shard]]:
    if False:
        print('Hello World!')
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)
    assert type(tensor) is ShardedTensor
    shards = tensor.local_shards()
    return (tensor, shards)

def _ext_all_gather_dtensor(tensor: DTensor, parent_mesh: Optional[DeviceMesh], fsdp_extension: Optional[FSDPExtensions]=None) -> torch.Tensor:
    if False:
        return 10
    all_gather_dtensor_fn = fsdp_extension.all_gather_dtensor if fsdp_extension is not None else _all_gather_dtensor
    return all_gather_dtensor_fn(tensor, parent_mesh)