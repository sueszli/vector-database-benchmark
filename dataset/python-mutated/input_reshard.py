from functools import partial
from typing import Any, Optional, Tuple
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
__all__ = ['input_reshard']

def input_reshard(module: torch.nn.Module, tp_device_mesh: DeviceMesh, input_reshard_dim: Optional[int]=None) -> torch.nn.Module:
    if False:
        for i in range(10):
            print('nop')
    '\n    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.\n\n    Register hooks to an nn.Module with input resharding so that we can shard\n    per the given `tp_device_mesh` and `input_reshard_dim` and restore the\n    input back when recomputing the activations in the backward. The reason\n    why we can do this is that for Tensor Parallel(TP), the input are same\n    across all TP ranks.\n\n    Args:\n        module (:class:`nn.Module`):\n            Module to be registered with input resharding.\n        tp_device_mesh (:class:`DeviceMesh`):\n            Object which describes the mesh topology\n            of devices for Tensor Parallel.\n        input_reshard_dim (Optional[int]):\n            The dimension of where we perform the sharding\n            of input. If set None, there is no sharding of input.\n            Default: None\n\n    Return:\n        A :class:`nn.Module` object registered with TP input resharding.\n    '
    cx: Optional[torch.autograd.graph.saved_tensors_hooks] = None

    def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: Tuple[Any, ...]) -> None:
        if False:
            return 10
        saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim), partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim))
        saved_tensor_hooks.__enter__()
        nonlocal cx
        cx = saved_tensor_hooks

    def input_reshard_backward_hook(_: torch.nn.Module, _i: Tuple[Any, ...], _o: Any) -> Any:
        if False:
            i = 10
            return i + 15
        nonlocal cx
        cx.__exit__()
    if input_reshard_dim is None:
        return module
    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module

def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:
    if False:
        i = 10
        return i + 15
    'Hook function called after FWD to shard input.'
    if isinstance(x, DTensor) and all((p.is_replicate() for p in x._spec.placements)):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif not isinstance(x, DTensor) and isinstance(x, torch.Tensor) and (x.numel() >= mesh.size()):
        return DTensor.from_local(x, device_mesh=mesh).redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)]).to_local()
    else:
        return x

def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Hook function called before activation recomputing in BWD to restore input.'
    if isinstance(x, DTensor) and len(x._spec.placements) == 1 and x._spec.placements[0].is_shard():
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif not isinstance(x, DTensor) and isinstance(x, torch.Tensor) and (x.numel() >= mesh.size()):
        return DTensor.from_local(x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]).redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
    else:
        return x