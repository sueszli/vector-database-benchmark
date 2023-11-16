from typing import Tuple, Union, Sequence, cast
import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor import DTensor as DT
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import _Partial, Placement, Replicate, Shard

def _view_with_sharding_dim_change(tensor: Union[torch.Tensor, DT], sharding_dim: int, shape: Tuple[int, ...]) -> Union[torch.Tensor, DT]:
    if False:
        print('Hello World!')
    "\n    Change the implicit sharding dim for a distributed tensor without comms.\n\n    Because if we don't change sharding dim, we will ended up having more comms that are not necessary.\n    Note that this op will produce invalid DTensor, you will need to call this op in pair to recover\n    it back to a valid DTensor.\n\n    This should only be used when implicitly changing sharding dim doesn't have semantic issue.\n    "
    if isinstance(tensor, DT):
        return _ViewAndRedistribute.apply(tensor, sharding_dim, shape)
    else:
        return tensor.view(shape)

def _infer_dtensor_stride(local_tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]) -> Tuple[int, ...]:
    if False:
        print('Hello World!')
    'Infer the dtensor stride from a local tensor.'
    tensor_stride = list(local_tensor.stride())
    for (idx, placement) in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    tensor_stride[i] = tensor_stride[i] * mesh.size(idx)
        elif not isinstance(placement, (Replicate, _Partial)):
            raise RuntimeError(f'placement type {type(placement)} not supported!')
    return tuple(tensor_stride)

class _ViewAndRedistribute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, self: DT, sharding_dim: int, shape: Tuple[int, ...]) -> DT:
        if False:
            for i in range(10):
                print('nop')
        ctx.previous_placement = self.placements
        ctx.previous_device_mesh = self.device_mesh
        ctx.previous_local_shape = self.to_local().size()
        ctx.previous_global_shape = self.size()
        assert self.device_mesh.ndim == 1, 'Only support 1D Device Mesh for _ViewAndRedistribute.'
        if self.placements[0].is_shard(dim=sharding_dim) or self.placements[0].is_replicate() or self.placements[0].is_partial():
            return self.view(shape)
        else:
            if sharding_dim < 0:
                sharding_dim += self.dim()
            device_mesh = self.device_mesh
            world_size = device_mesh.size(dim=0)
            new_sharding_placement = [Shard(sharding_dim)]
            try:
                infer_idx = shape.index(-1)
            except ValueError:
                infer_idx = None
            if infer_idx is not None:
                st_size = prod(self.size())
                shape_size = -1 * prod(shape)
                shape = (*shape[:infer_idx], st_size // shape_size, *shape[infer_idx + 1:])
            new_local_tensor_size = (*shape[:sharding_dim], shape[sharding_dim] // world_size, *shape[sharding_dim + 1:])
            new_local_tensor = self.to_local().view(*new_local_tensor_size)
            return DT(new_local_tensor, device_mesh, tuple(new_sharding_placement), shape=torch.Size(shape), dtype=new_local_tensor.dtype, requires_grad=new_local_tensor.requires_grad, stride=_infer_dtensor_stride(new_local_tensor, device_mesh, new_sharding_placement))

    @staticmethod
    def backward(ctx, grad_output: DT) -> Tuple[DT, None, None]:
        if False:
            i = 10
            return i + 15
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        previous_local_tensor_size = ctx.previous_local_shape
        previous_global_shape = ctx.previous_global_shape
        new_local_tensor = grad_output.to_local().view(*previous_local_tensor_size)
        return (DT(new_local_tensor, previous_device_mesh, previous_placement, shape=previous_global_shape, dtype=grad_output.dtype, requires_grad=grad_output.requires_grad, stride=_infer_dtensor_stride(new_local_tensor, previous_device_mesh, previous_placement)), None, None)