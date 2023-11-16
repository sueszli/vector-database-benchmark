import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed._tensor.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import all_to_all, broadcast, get_global_rank, get_rank, get_world_size, GroupMember, ProcessGroup, scatter, Work
logger = logging.getLogger(__name__)

def mesh_scatter(output: torch.Tensor, scatter_list: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int=0, async_op: bool=False) -> Optional[Work]:
    if False:
        while True:
            i = 10
    '\n    scatter a list of tensors to a device mesh dimension. We by default\n    use the first rank of the mesh dimension as the source of truth, i.e\n    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will\n    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank\n    2 to rank 2/3.\n\n    Args:\n        output (torch.Tensor): the tensor to receive the scattered list.\n        scatter_list (List[torch.Tensor]): the tensor list to be scattered.\n        mesh_dim (int, optional): indicate which mesh dimension we want\n            to scatter on, we by default choose the first rank on the\n            mesh dimension as source of truth.\n\n    Returns:\n        A :class:`Work` object\n    '
    if output.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)
    if src_for_dim == get_rank():
        fut = scatter(output, scatter_list=scatter_list, src=src_for_dim, group=dim_group, async_op=async_op)
    else:
        fut = scatter(output, scatter_list=None, src=src_for_dim, group=dim_group, async_op=async_op)
    return fut

def mesh_broadcast(tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int=0, async_op: bool=False) -> Optional[Work]:
    if False:
        print('Hello World!')
    '\n    broadcast the tensor to a device mesh dimension. We by default\n    use the first rank of the mesh dimension as the source of truth, i.e\n    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will\n    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2\n    to rank 2/3.\n\n    Args:\n        tensor (torch.Tensor): tensor to broadcast.\n        mesh_dim (int, optional): indicate which mesh dimension we want\n            to scatter on, we by default choose the first rank on the\n            mesh dimension as source of truth.\n\n    Returns:\n        A :class:`Work` object\n    '
    if tensor.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)
    return broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)

def mesh_all_to_all(output_tensor_list: List[torch.Tensor], input_tensor_list: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int=0, async_op: bool=False) -> Optional[Work]:
    if False:
        print('Hello World!')
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    work = None
    if mesh.device_type == 'cpu':
        logger.warning('ProcessGroupGloo does not support all_to_all, falling back with scatters!')
        dim_group_size = get_world_size(dim_group)
        for i in range(dim_group_size):
            src_for_dim = i
            if dim_group is not GroupMember.WORLD:
                src_for_dim = get_global_rank(dim_group, i)
            work = scatter(output_tensor_list[i], input_tensor_list if mesh.get_rank() == src_for_dim else [], group=dim_group, src=src_for_dim, async_op=async_op)
    else:
        work = all_to_all(output_tensor_list, input_tensor_list, dim_group, async_op=async_op)
    return work

def spec_to_bytes(spec: 'placement_types.DTensorSpec') -> int:
    if False:
        print('Hello World!')
    assert spec.tensor_meta is not None, 'spec should have tensor meta defined!'
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)

def get_bandwidth_factor(mesh: DeviceMesh) -> List[float]:
    if False:
        return 10
    factors = [1.0] * mesh.ndim
    num_devices_per_host = _mesh_resources.num_devices_per_host(mesh.device_type)
    num_devices = 1
    for mesh_dim in reversed(range(mesh.ndim)):
        num_devices *= mesh.size(mesh_dim)
        if num_devices <= num_devices_per_host:
            factors[mesh_dim] = 0.2
    return factors

def allgather_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    if False:
        for i in range(10):
            print('nop')
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    return 1 + bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim

def allreduce_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    if False:
        i = 10
        return i + 15
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    return 1 + 2 * bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim

def reduce_scatter_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    if False:
        return 10
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    return 1 + bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim

def redistribute_cost(current_spec: 'placement_types.DTensorSpec', target_spec: 'placement_types.DTensorSpec') -> float:
    if False:
        print('Hello World!')
    '\n    This function returns the cost of redistribute from current to target DTensorSpec.\n\n    NOTE:\n    1. Only consider communication cost here, since computation costs for redistribute\n       are quite trival (i.e. we only need to narrow or simple division)\n    2. Only consider redistribute cost on same mesh, cross mesh communication cost is\n       not quite needed for operator strategy estimation/selection.\n    '
    if current_spec.mesh != target_spec.mesh:
        return float('inf')
    if current_spec.is_replicated():
        return 0.0
    mesh = current_spec.mesh
    cost = 0.0
    comm_bytes = spec_to_bytes(current_spec) / current_spec.num_shards
    for (i, (current, target)) in enumerate(zip(current_spec.placements, target_spec.placements)):
        if current == target:
            continue
        if current.is_shard() and target.is_replicate():
            comm_bytes *= mesh.size(i)
            cost += allgather_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_shard() and target.is_shard():
            cost += allgather_cost(comm_bytes, current_spec.mesh, i) + 1.0
        elif current.is_partial() and target.is_replicate():
            cost += allreduce_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_partial() and target.is_shard():
            cost += reduce_scatter_cost(comm_bytes, current_spec.mesh, i)
            comm_bytes /= mesh.size(i)
        elif current.is_shard() and target.is_partial():
            return float('inf')
    return cost