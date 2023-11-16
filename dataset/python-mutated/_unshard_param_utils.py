import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _has_fsdp_params, _module_handle, HandleTrainingState, TrainingState
from torch.distributed.fsdp._runtime_utils import _get_fsdp_root_states_with_modules, _lazy_init, _reset_flat_param_grad_info_if_needed, _reshard, _reshard_grads, _unshard, _unshard_grads
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
FLAT_PARAM = '_flat_param'

@torch.no_grad()
def _writeback_to_local_shard(handle: FlatParamHandle, writeback_grad: bool):
    if False:
        return 10
    "\n    For the handle, writes back the this rank's shard of the unsharded\n    flattened parameter to the sharded flattened parameter. If\n    ``writeback_grad=True``, then writes back to the sharded gradient as\n    well.\n\n    Precondition: The handle's ``FlatParameter`` 's data points to the\n    padded unsharded flattened parameter.\n    "

    def _get_shard(flat_param_or_grad: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        if handle.uses_sharded_strategy:
            (shard, _) = FlatParamHandle._get_unpadded_shard(flat_param_or_grad, handle.rank, handle.world_size)
            return shard
        return flat_param_or_grad
    param_shard = _get_shard(handle.flat_param)
    handle.flat_param._local_shard[:param_shard.numel()].copy_(param_shard)
    if writeback_grad:
        existing_grad = handle.sharded_grad
        if existing_grad is not None:
            assert handle.flat_param.grad is not None
            grad_shard = _get_shard(handle.flat_param.grad)
            existing_grad[:grad_shard.numel()].copy_(grad_shard)

def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    if False:
        print('Hello World!')
    '\n    De-registers the flattened parameter from the wrapped module, hiding it\n    from ``nn.Module`` methods.\n\n    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an\n    attribute but dynamically change whether it is visible to ``nn.Module``\n    methods.\n    '
    if _has_fsdp_params(state, module):
        cast(nn.Module, module.module)._parameters.pop(FLAT_PARAM, None)

def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Registers the flattened parameter to the wrapped module, making it\n    visible to ``nn.Module`` methods.\n\n    We do not use :meth:`nn.Module.register_parameter` because we want\n    ``FLAT_PARAM`` to always be an attribute but dynamically change whether\n    it is visible to ``nn.Module`` methods.\n    '
    handle = _module_handle(state, module)
    if _has_fsdp_params(state, module):
        cast(nn.Module, module.module)._parameters[FLAT_PARAM] = handle.flat_param

@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    if False:
        print('Hello World!')
    '\n    Assumes that the flattened parameter is unsharded. When in the context,\n    de-registers the flattened parameter and unflattens the original\n    parameters as ``nn.Parameter`` views into the flattened parameter.\n    After the context, re-registers the flattened parameter and restores\n    the original parameters as ``Tensor`` views into the flattened\n    parameter.\n    '
    handle = _module_handle(state, module)
    if not handle:
        yield
    else:
        _deregister_flat_param(state, module)
        try:
            with handle.unflatten_as_params():
                yield
        finally:
            if not handle._use_orig_params:
                _register_flat_param(state, module)

def _validate_unshard_params_args(state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    if False:
        while True:
            i = 10
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        raise NotImplementedError(f'with_grads={with_grads}, use_orig_params={state._use_orig_params}, offload_to_cpu={offload_to_cpu} is not supported yet')
    if offload_to_cpu and state._handle and (not state._handle.uses_sharded_strategy):
        raise NotImplementedError('offload_to_cpu=True and NO_SHARD is not supported yet')
    if writeback and rank0_only:
        raise NotImplementedError('writeback=True and rank0_only=True is not supported yet')
    if offload_to_cpu and (not rank0_only):
        warnings.warn('offload_to_cpu=True and rank0_only=False may result in theunsharded parameters being redundantly copied to CPU memory for GPUs sharing the same CPU memory, which risks CPU OOM. We recommend using offload_to_cpu=True with rank0_only=True.')

@contextlib.contextmanager
def _unshard_fsdp_state_params(module: nn.Module, state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    if False:
        for i in range(10):
            print('nop')
    '\n    This unshards the parameters for a single FSDP state ``state`` that\n    corresponds to ``module``.\n    '
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    state._device_handle.synchronize()
    maybe_handle = _module_handle(state, module)
    handle = None
    if maybe_handle and maybe_handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS:
        handle = maybe_handle
    if not handle:
        yield
        return
    assert handle._training_state == HandleTrainingState.IDLE, f'Expects the handle training to be IDLE but got {handle._training_state}'
    handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS
    _reset_flat_param_grad_info_if_needed(handle)
    free_unsharded_flat_param = handle.needs_unshard()
    computation_stream = state._device_handle.current_stream()
    _unshard(state, handle, computation_stream, computation_stream)
    if with_grads:
        _unshard_grads(handle)
    if rank0_only and state.rank != 0:
        _reshard(state, handle, free_unsharded_flat_param)
        if with_grads:
            _reshard_grads(handle)
        try:
            yield
        finally:
            handle._training_state = HandleTrainingState.IDLE
    else:
        with contextlib.ExitStack() as stack:
            if offload_to_cpu and handle.uses_sharded_strategy:
                stack.enter_context(handle.to_cpu())
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE

@contextlib.contextmanager
def _unshard_params_recurse(module: nn.Module, state: _FSDPState, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    if False:
        print('Hello World!')
    '\n    This is a helper for :func:`_unshard_params` that recursively calls\n    :func:`_unshard_fsdp_state_params` on FSDP states if ``recurse=True``.\n    NOTE: This runs lazy initialization.\n    '
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    if recurse:
        with contextlib.ExitStack() as stack:
            for (state, fsdp_module) in zip(*traversal_utils._get_fsdp_states_with_modules(module)):
                stack.enter_context(_unshard_params_recurse(module=fsdp_module, state=state, recurse=False, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads))
            yield
        return
    _lazy_init(state, module)
    if state.training_state == TrainingState.FORWARD_BACKWARD:
        raise AssertionError('Cannot manually unshard parameters during forward/backward')
    elif state.training_state == TrainingState.SUMMON_FULL_PARAMS:
        raise AssertionError('Cannot manually unshard parameters when already unsharding parameters')
    with _unshard_fsdp_state_params(module=module, state=state, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads):
        try:
            state.training_state = TrainingState.SUMMON_FULL_PARAMS
            yield
        finally:
            state.training_state = TrainingState.IDLE

@contextlib.contextmanager
def _unshard_params(module: nn.Module, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    if False:
        return 10
    '\n    This unshards FSDP-managed parameters for all modules with FSDP applied in\n    the module tree rooted at ``module``.\n    '
    (root_fsdp_states, root_fsdp_modules) = _get_fsdp_root_states_with_modules(module)
    with contextlib.ExitStack() as stack:
        for (root_fsdp_state, root_fsdp_module) in zip(root_fsdp_states, root_fsdp_modules):
            stack.enter_context(_unshard_params_recurse(module=root_fsdp_module, state=root_fsdp_state, recurse=recurse, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads))
        yield
    return

def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Deregisters the original parameters; registers the ``FlatParameter``.\n    '
    handle = _module_handle(state, module)
    if not handle:
        return
    _p_assert(handle._use_orig_params, f'Inconsistent `_use_orig_params` -- FSDP: {state._use_orig_params} handle: {handle._use_orig_params}')
    handle._deregister_orig_params()
    _register_flat_param(state, module)

def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    if False:
        return 10
    '\n    Deregisters the ``FlatParameter``; registers the original parameters.\n    '
    handle = _module_handle(state, module)
    if not handle:
        return
    _deregister_flat_param(state, module)
    if handle.is_sharded(handle.flat_param):
        handle._use_sharded_views()
        handle._use_sharded_grad_views()
    else:
        handle._use_unsharded_views(as_params=True)