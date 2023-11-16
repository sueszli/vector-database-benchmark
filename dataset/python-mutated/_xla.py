import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Placement, Replicate
log = logging.getLogger(__name__)
TORCH_XLA_INITIALIZED = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
    from torch_xla.experimental.xla_sharding import mark_sharding, Mesh, ShardingType
    TORCH_XLA_INITIALIZED = True
except ImportError as e:
    log.warning(e.msg)

def with_xla(func: Callable) -> Callable:
    if False:
        print('Hello World!')
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if TORCH_XLA_INITIALIZED:
            os.environ['XLA_USE_SPMD'] = '1'
            return func(self, *args, **kwargs)
        else:
            raise ImportError('torch.distributed._tensor._xla API requires torch_xla package installation.')
    return wrapper

@with_xla
def convert_to_xla_mesh(dt_mesh: DeviceMesh) -> 'Mesh':
    if False:
        print('Hello World!')
    '\n    Convert DTensor `dt_mesh` to XLAShardedTensor `partition_spec`.\n\n    Example (1x4 logical device mesh topology):\n      ```\n      dt_mesh = DeviceMesh("xla", [[1, 2, 3, 4]])\n      dt_mesh.shape\n      >> torch.Size([1, 4])\n\n      mesh = convert_to_xla_mesh(dt_mesh)\n      mesh_shape\n      >> [1, 4]\n      ```\n    '
    assert dt_mesh.size() == xr.global_runtime_device_count()
    return Mesh(dt_mesh.mesh.flatten(), tuple(dt_mesh.mesh.size()), dt_mesh.mesh_dim_names)

@with_xla
def convert_to_xla_partition_spec(tensor: torch.Tensor, placements: Sequence[Placement]) -> Tuple[Union[Tuple, int, None]]:
    if False:
        i = 10
        return i + 15
    '\n    Convert DTensor `placements` to XLAShardedTensor `partitoin_spec`.\n    This supports Shard and Replicate Placement types.\n\n    Example:\n      ```\n      # Mesh partitioning, 1/4-th of the input with replicated overlaps.\n      # The first input tensor dimension is sharded across the second mesh\n      # dimension, and the rest is replicated over the first mesh dimension.\n      t = torch.randn(4, 8, 8)\n      dt_mesh = DeviceMesh("xla", torch.arange(8).reshape(2,4))\n      placements = [Replicate(), Shard(0)]\n      my_dtensor = distribute_tensor(t, dt_mesh, placements)\n\n      # `placements = [Replicate(), Shard(0)]` describes sharding per mesh dim,\n      # and this is equivalent to `partition_spec = (1, None, None)` which is\n      # sharding per input tensor dimension.\n      partition_spec = convert_to_xla_partition_spec(t, placements)\n      >> (1, None, None)\n      ```\n    '
    sharding_spec = [None] * len(tensor.shape)
    for (mesh_idx, spec) in enumerate(placements):
        if spec.is_shard():
            tensor_idx = spec.dim
            sharding_spec[tensor_idx] = mesh_idx
        elif spec.is_replicate():
            continue
        else:
            raise ValueError(f'Unsupported placement type: {type(spec).__name__}')
    return tuple(sharding_spec)

@with_xla
def xla_distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh, placements: Optional[Sequence[Placement]]=None) -> 'XLAShardedTensor':
    if False:
        print('Hello World!')
    '\n    Distribute a torch.Tensor to the `device_mesh` according to the `placements`\n    specified. The rank of `device_mesh` and `placements` must be the same.\n\n    Args:\n        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you\n            want to shard a tensor on a dimension that is not evenly divisible by\n            the number of devices in that mesh dimension, we use `torch.chunk`\n            semantic to shard the tensor and scatter the shards.\n        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the\n            tensor, if not specified, must be called under a DeviceMesh context\n            manager, default: None\n        placements (List[:class:`Placement`], optional): the placements that\n            describes how to place the tensor on DeviceMesh, must have the same\n            number of elements as `device_mesh.ndim`. If not specified, we will\n            by default replicate the tensor across the `device_mesh` from the\n            first rank of each dimension of the `device_mesh`.\n\n    Returns:\n        A :class:`XLAShardedTensor` object\n\n    .. note:: We return a XLAShardedTensor with a global view and access to local shards.\n    The successive ops would be programmed as if on a single-device and without calling\n    any explicit collective ops. The actual sharded computation on the sharding annotated tensor\n    happens lazily, is transparent to the user. In the future, we will introduce\n    a new DTensor type for this kind of programming-mode (single-controller) and return.\n    '
    dt_mesh = device_mesh
    assert dt_mesh.device_type == 'xla'
    xla_mesh = convert_to_xla_mesh(dt_mesh)
    assert xla_mesh.mesh_shape == tuple(dt_mesh.mesh.size())
    if not tensor.is_meta:
        tensor = tensor.to(dt_mesh.device_type)
    if placements is None:
        placements = [Replicate() for _ in range(dt_mesh.ndim)]
    assert len(placements) == dt_mesh.ndim, '`placements` must have the same length as `device_mesh.ndim`! '
    f'Found placements length: {len(placements)}, and device_mesh.ndim: {dt_mesh.ndim}.'
    partition_spec = convert_to_xla_partition_spec(tensor, placements)
    assert len(tensor.shape) == len(partition_spec), '`partition_spec` from `placements` must have the same length as `tensor.length`! '
    f'Found tensor shape length: {len(tensor.shape)}, and partition_spec length: {len(partition_spec)}.'
    global_tensor = tensor
    if type(tensor).__name__ == 'DTensor':
        raise ValueError('Cannot distribute a DTensor with local tensor on xla devices.The input tensor must be global.')
    if type(tensor).__name__ == 'XLAShardedTensor':
        sharding_type = tensor.sharding_type
        assert sharding_type is None or sharding_type == ShardingType.REPLICATED, 'XLAShardedTensor `tensor` is already annotated with non-replication sharding. '
        'Clear the existing sharding annotation first, by callling torch_xla.experimental.xla_sharding.clear_sharding API.'
        global_tensor = tensor.global_tensor
    assert global_tensor is not None, 'distributing a tensor should not be None'
    xla_tensor = mark_sharding(global_tensor, xla_mesh, partition_spec)
    return xla_tensor

@with_xla
def xla_distribute_module(module: nn.Module, device_mesh: Optional[DeviceMesh]=None, partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]]=None, input_fn: Optional[Callable[..., None]]=None, output_fn: Optional[Callable[..., None]]=None) -> nn.Module:
    if False:
        return 10
    raise NotImplementedError