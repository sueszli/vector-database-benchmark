from typing import cast, Dict, List, Tuple
import torch
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement, Replicate, Shard
_PlacementItem = Tuple[int, Tuple[Placement, Placement]]

def _replicate_then_shard(val: _PlacementItem) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Replicate from inner to outer dimension.\n    Shard from outer to inner dimension.\n    '
    (i, (current, target)) = val
    if (target.is_replicate() or target.is_partial()) and current.is_shard():
        return -i
    elif (current.is_replicate() or current.is_partial()) and target.is_shard():
        return i
    else:
        return 0

def _decompose_reshard(val: List[_PlacementItem]) -> List[_PlacementItem]:
    if False:
        i = 10
        return i + 15
    '\n    Decompose Si -> Sj into Si -> R -> Sj\n    There\'s 2 ways a shardings can differ within a mesh dimension:\n      1) sharding on different tensor dimensions, e.g. Shard(0) -> Shard(1)\n      2) different sub-shards of a repeated shard ("mis-aligned sharding")\n          (Shard(0), Shard(0)) -> (Replicate(), Shard(0))\n          Here the Shard(0) -> Shard(0) for mesh dimension 2 is actually\n          a reshard, because in the first case it\'s a sub-sharding of an already tensor dimension 0,\n          and in the second case, it\'s the first sharding on tensor dimension 0.\n    '
    from collections import defaultdict
    repeat_dim_current: Dict[int, int] = defaultdict(int)
    repeat_dim_target: Dict[int, int] = defaultdict(int)
    output: List[_PlacementItem] = []
    for (i, (current, target)) in val:
        if current.is_shard():
            repeat_dim_current[cast(Shard, current).dim] += 1
        if target.is_shard():
            repeat_dim_target[cast(Shard, target).dim] += 1
        if isinstance(current, Shard) and isinstance(target, Shard) and (current.dim != target.dim or repeat_dim_current[current.dim] != repeat_dim_target[target.dim]):
            output.append((i, (current, Replicate())))
            output.append((i, (Replicate(), target)))
        else:
            output.append((i, (current, target)))
    return output

def redistribute_local_tensor(local_tensor: torch.Tensor, current_spec: DTensorSpec, target_spec: DTensorSpec) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to\n    the target DTensorSpec, which involves the necessary collective calls to transform\n    the local shard of the DTensor from its current spec to the target spec.\n    '
    if current_spec.mesh != target_spec.mesh:
        raise NotImplementedError('Cross device mesh comm not supported yet!')
    new_local_tensor = None
    current_placements = current_spec.placements
    target_placements = target_spec.placements
    sorted_placements = list(enumerate(zip(current_placements, target_placements)))
    sorted_placements = _decompose_reshard(sorted_placements)
    sorted_placements.sort(key=_replicate_then_shard)
    device_mesh = current_spec.mesh
    for (i, (current, target)) in sorted_placements:
        my_coordinate = device_mesh.get_coordinate()
        num_chunks = device_mesh.size(dim=i)
        if my_coordinate is None:
            return local_tensor
        if current == target:
            new_local_tensor = local_tensor
            continue
        if target.is_replicate():
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                new_local_tensor = partial_spec._to_replicate(local_tensor, device_mesh, i)
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(local_tensor, current_spec.shape, device_mesh, i)
            else:
                raise RuntimeError(f'redistribute from {current_placements} to {target_placements} not supported yet')
        elif target.is_shard():
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                new_local_tensor = partial_spec._to_shard(local_tensor, device_mesh, i, target_placement)
            elif current.is_replicate():
                (shards, _) = target_placement._split_tensor(local_tensor, num_chunks, with_padding=False, contiguous=False)
                new_local_tensor = shards[my_coordinate[i]].clone()
            else:
                assert current.is_shard(), f'Current placement should be shard but found {current}'
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    raise NotImplementedError('Changing sharding dim is not supported yet!')
        elif target.is_partial():
            if current.is_replicate():
                new_local_tensor = local_tensor / num_chunks
            else:
                raise RuntimeError(f'redistribute from {current_placements} to {target_placements} not supported yet')
        assert new_local_tensor is not None
        local_tensor = new_local_tensor
    assert new_local_tensor is not None, 'redistribute failed!'
    return new_local_tensor

class Redistribute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: 'dtensor.DTensor', device_mesh: DeviceMesh, placements: List[Placement]):
        if False:
            while True:
                i = 10
        current_spec = input._spec
        ctx.current_spec = current_spec
        target_spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=input._spec.tensor_meta)
        local_tensor = input._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec)
        return dtensor.DTensor(output, device_mesh, target_spec.placements, shape=input.shape, dtype=input.dtype, requires_grad=input.requires_grad, stride=input.stride())

    @staticmethod
    def backward(ctx, grad_output: 'dtensor.DTensor'):
        if False:
            i = 10
            return i + 15
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec
        target_placements: List[Placement] = []
        for (current, target) in zip(current_spec.placements, previous_spec.placements):
            if not current.is_partial() and target.is_partial():
                target_placements.append(Replicate())
            else:
                target_placements.append(target)
        target_spec = DTensorSpec(previous_spec.mesh, tuple(target_placements), tensor_meta=previous_spec.tensor_meta)
        local_tensor = grad_output._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec)
        output_dtensor = dtensor.DTensor(output, target_spec.mesh, target_spec.placements, shape=grad_output.shape, dtype=grad_output.dtype, requires_grad=grad_output.requires_grad, stride=grad_output.stride())
        return (output_dtensor, None, None)