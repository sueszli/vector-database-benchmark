import functools
import logging
from enum import auto, Enum
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import _assert_in_training_states, _FSDPState, _get_module_fsdp_state, _is_composable, _log_post_backward_hook, _no_dispatch_record_stream, clean_tensor_name, TrainingState
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle, HandleShardingStrategy, HandleTrainingState, RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import _apply_to_tensors, _cast_forward_inputs, _p_assert, _to_kwargs
from torch.utils import _pytree as pytree
log = logging.getLogger(__name__)
HOMOGENEOUS_ATTR_NAMES = ('_use_orig_params', 'limit_all_gathers', '_use_full_prec_in_eval')

class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()

def _get_fsdp_root_states_with_modules(module: nn.Module) -> Tuple[List[_FSDPState], List[nn.Module]]:
    if False:
        while True:
            i = 10
    '\n    Returns a tuple containing:\n    1. A list of the root ``_FSDPState`` instances in the module tree rooted at\n    ``module`` without any duplicates and following the ``module.modules()``\n    traversal order (which is assumed to be depth-first).\n    2. A corresponding list of the root modules owning the states in the first\n    list.\n\n    This is similar to :func:`_get_fsdp_states_with_modules` except that we\n    must call :func:`_is_fsdp_root` to force a lazy initialization to determine\n    the FSDP root in case lazy initialization has not yet happened.\n    '
    fsdp_root_states: List[_FSDPState] = []
    fsdp_root_modules: List[nn.Module] = []
    visited_fsdp_states: Set[_FSDPState] = set()
    for submodule in module.modules():
        optional_state = _get_module_fsdp_state(submodule)
        if optional_state is not None and optional_state not in visited_fsdp_states and _is_fsdp_root(optional_state, submodule):
            visited_fsdp_states.add(optional_state)
            fsdp_root_states.append(optional_state)
            fsdp_root_modules.append(submodule)
    return (fsdp_root_states, fsdp_root_modules)

def _get_fsdp_root_states(module: nn.Module) -> List[_FSDPState]:
    if False:
        while True:
            i = 10
    'See :func:`_get_fsdp_root_states_with_modules`.'
    (fsdp_root_states, _) = _get_fsdp_root_states_with_modules(module)
    return fsdp_root_states

def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    if False:
        print('Hello World!')
    "\n    Returns if ``state`` corresponds to that of an FSDP root.\n\n    For the wrapper code path, ``state`` and ``module`` should be the same. For\n    the non-wrapper code path, ``state`` should be ``module`` 's state.\n    "
    _lazy_init(state, module)
    assert state._is_root is not None
    return state._is_root

@no_type_check
def _validate_and_get_hybrid_shard_state(root_module: nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Precondition: ``root_module`` is a ``FullyShardedDataParallel`` instance.\n\n    This checks that all instances using a hybrid sharding strategy have the\n    same intra- and inter-node process groups.\n    '
    intra_node_pgs: Set[dist.ProcessGroup] = set()
    inter_node_pgs: Set[dist.ProcessGroup] = set()
    for fsdp_state in traversal_utils._get_fsdp_states(root_module):
        if fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            intra_node_pgs.add(fsdp_state.process_group)
            inter_node_pgs.add(fsdp_state._inter_node_pg)
    if len(intra_node_pgs) == 0 and len(inter_node_pgs) == 0:
        return
    error_prefix = 'At least one instance uses a hybrid sharding strategy but has no '
    if len(intra_node_pgs) > 0 and len(inter_node_pgs) == 0:
        raise AssertionError(error_prefix + 'inter-node process group set')
    if len(intra_node_pgs) == 0 and len(inter_node_pgs) > 0:
        raise AssertionError(error_prefix + 'intra-node process group set')
    error_prefix = 'Some instances use a hybrid sharding strategy, but '
    if len(intra_node_pgs) != 1:
        raise ValueError(error_prefix + 'intra-node process groups do not match')
    if len(inter_node_pgs) != 1:
        raise ValueError(error_prefix + 'inter-node process groups do not match')

@no_type_check
def _lazy_init(state: _FSDPState, root_module: nn.Module) -> _FSDPState:
    if False:
        while True:
            i = 10
    "\n    Performs initialization lazily, typically right before the first forward\n    pass. The laziness is needed to ensure that the parameter device/dtype and\n    the FSDP hierarchy have finalized. This method's actual logic only runs on\n    the root FSDP instance, which performs initialization for all non-root FSDP\n    instances to avoid partial initialization.\n\n    For the non-composable code path, ``state`` and ``root_module`` should be\n    the same, namely the FSDP instance itself.\n    "
    if state._is_root is not None:
        return
    if not state._device_handle.is_available():
        raise RuntimeError('FSDP does not support CPU only execution')
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    _init_streams(state)
    (buffers, buffer_dtypes) = _get_buffers_and_dtypes_for_computation(state, root_module)
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    state._exec_order_data.init(state, root_module, state.process_group)
    _share_state_and_init_handle_attrs(state, root_module)
    return state

def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    if False:
        return 10
    "\n    Checks that all ``FlatParameter``s in ``module`` 's tree managed by\n    ``state`` are on the expected device for *lazy initialization*.\n    "
    cpu_device = torch.device('cpu')
    for handle in traversal_utils._get_fsdp_handles(module):
        if not handle._offload_params and handle.flat_param.device != state.compute_device:
            raise RuntimeError(f'An FSDP-managed module unexpectedly has parameters on {handle.flat_param.device}. Make sure to move the module to {state.compute_device} before training.')
        elif handle._offload_params and handle.flat_param.device != cpu_device:
            raise RuntimeError(f'An FSDP-managed module with parameter CPU offloading enabled has parameters on {handle.flat_param.device}. Make sure to not move the module from CPU when offloading parameters.')

@no_type_check
def _share_state_and_init_handle_attrs(root_state: _FSDPState, root_module: nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Shares data structure state from the ``root_state`` to all FSDP states in\n    ``root_module`` 's module tree, and initializes handle attributes. These\n    are done together to require a single loop over the states.\n    "
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    _validate_and_get_hybrid_shard_state(root_module)
    attr_name_to_values: Dict[str, Set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_handles = root_state._exec_order_data.all_handles
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        if hasattr(flat_param, '_in_backward_optimizers'):
            raise RuntimeError('FSDP optimizer in backward only supported with use_orig_params=True!')
        handle._has_optim_in_backward = flat_param._params is not None and any((hasattr(param, '_in_backward_optimizers') for param in flat_param._params))
        if handle._has_optim_in_backward:
            torch._C._log_api_usage_once('fsdp.optimizer_in_backward')
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(hasattr(fsdp_state, attr_name), f'FSDP state missing attribute {attr_name}')
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        _p_assert(fsdp_state._is_root is None or not fsdp_state._is_root, "Non-root FSDP instance's `_is_root` should not have been set yet or should have been set to `False`")
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
        if hasattr(root_state, '_device_mesh'):
            fsdp_state._device_mesh = root_state._device_mesh
            fsdp_state._fsdp_extension = root_state._fsdp_extension
    for (attr_name, attr_values) in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(f'Expects one homogeneous value for {attr_name} but got {attr_values}')

@no_type_check
def _init_streams(state: _FSDPState) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Initializes CUDA streams for overlapping communication, computation, and\n    data transfers. The streams should be shared across FSDP instances.\n    '
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any((fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES for fsdp_state in state._all_fsdp_states))
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    state._default_stream = state._device_handle.current_stream()
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    state._all_reduce_stream = state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream

@no_type_check
def _unshard(state: _FSDPState, handle: FlatParamHandle, unshard_stream: torch.Stream, pre_unshard_stream: torch.Stream) -> None:
    if False:
        print('Hello World!')
    "\n    Unshards the handles in ``handles``. If the handles are in\n    :meth:`summon_full_params` and are using mixed precision, then they are\n    forced to full precision.\n\n    Postcondition: handle's ``FlatParameter`` 's data is the padded\n    unsharded flat parameter on the compute device.\n    "
    if not handle:
        return
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function('FullyShardedDataParallel.rate_limiter'):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()

@no_type_check
def _reshard(state: _FSDPState, handle: FlatParamHandle, free_unsharded_flat_param: bool):
    if False:
        while True:
            i = 10
    "\n    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to\n    free the handle's padded unsharded flat parameter.\n    "
    handle.reshard(free_unsharded_flat_param)
    if state.limit_all_gathers and free_unsharded_flat_param:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            free_event = state._device_handle.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
    handle.post_reshard()
    handle._prefetched = False

def _unshard_grads(handle: Optional[FlatParamHandle]) -> None:
    if False:
        print('Hello World!')
    if handle:
        handle.unshard_grad()

def _reshard_grads(handle: Optional[FlatParamHandle]) -> None:
    if False:
        i = 10
        return i + 15
    if handle:
        handle.reshard_grad()

@no_type_check
def _pre_forward(state: _FSDPState, handle: Optional[FlatParamHandle], unshard_fn: Callable, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if False:
        return 10
    '\n    Runs the pre-forward logic. This includes an opportunity to unshard\n    currently sharded parameters such as those for the current forward and\n    registering post-backward hooks for these current parameters. This function\n    also converts forward ``args`` and ``kwargs`` to the given precision.\n\n    Args:\n        handles (List[FlatParamHandle]): Handles giving the parameters used in\n            the current forward.\n        unshard_fn (Optional[Callable]): A callable to unshard any currently\n            sharded parameters or ``None`` to not do any unsharding.\n        module (nn.Module): Module whose forward this method runs right before;\n            expected by the hook signature.\n        args (Tuple[Any, ...]): Module forward ``args``.\n        kwargs (Dict[str, Any]): Module forward ``kwargs``.\n    '
    with torch.profiler.record_function('FullyShardedDataParallel._pre_forward'):
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return (args, kwargs)
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD
        if unshard_fn is not None:
            unshard_fn(state, handle)
        _register_post_backward_hook(state, handle)
        if handle and handle._offload_params and (handle.flat_param._cpu_grad is None):
            handle.flat_param._cpu_grad = torch.zeros_like(handle.flat_param._local_shard, device=torch.device('cpu')).pin_memory()
        should_cast_forward_inputs = state._handle and (not state._handle._force_full_precision)
        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            (args, kwargs) = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return (args, kwargs)

@no_type_check
def _pre_forward_unshard(state: _FSDPState, handle: Optional[FlatParamHandle]) -> None:
    if False:
        while True:
            i = 10
    'Unshards parameters in the pre-forward.'
    if not handle:
        return
    if not handle._prefetched:
        _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._needs_pre_forward_unshard = False
    state._device_handle.current_stream().wait_stream(state._unshard_stream)
    with torch.profiler.record_function('FullyShardedDataParallel._pre_forward_prefetch'):
        _prefetch_handle(state, handle, _PrefetchMode.FORWARD)

@no_type_check
def _post_forward(state: _FSDPState, handle: Optional[FlatParamHandle], reshard_fn: Callable, module: nn.Module, input: Any, output: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "\n    Runs the post-forward logic. This includes an opportunity to reshard\n    currently unsharded parameters such as those used in the current forward\n    and registering pre-backward hooks on the forward outputs.\n\n    Args:\n        handles (List[FlatParamHandle]): Handles giving the parameters used in\n            the current forward.\n        reshard_fn (Optional[Callable]): A callable to reshard any currently\n            unsharded parameters (e.g. from the current forward) or ``None`` to\n            not do any resharding.\n        module (nn.Module): Module whose forward just ran, which should be a\n            fully sharded module (see [Note: Fully Sharded Module]); expected\n            by the hook signature.\n        input (Any): Unused; expected by the hook signature.\n        output (Any): Forward pass output; pre-backward hooks are registered on\n            the tensors that require gradients in this output.\n\n    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat\n    parameter.\n    "
    with torch.profiler.record_function('FullyShardedDataParallel._post_forward'):
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return output
        state._exec_order_data.record_post_forward(handle)
        if reshard_fn is not None:
            reshard_fn(state, handle)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output

@no_type_check
def _post_forward_reshard(state: _FSDPState, handle: FlatParamHandle) -> None:
    if False:
        print('Hello World!')
    'Reshards parameters in the post-forward.'
    if not handle:
        return
    free_unsharded_flat_param = not state._is_root and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    _reshard(state, handle, free_unsharded_flat_param)

@no_type_check
def _root_pre_forward(state: _FSDPState, module: nn.Module, args, kwargs) -> None:
    if False:
        while True:
            i = 10
    "\n    Runs pre-forward logic specific to the root FSDP instance, which should run\n    before any individual module's pre-forward. This starts with an attempt at\n    lazy initialization (which only runs non-vacuously once). Otherwise, if\n    this is called on a non-root FSDP instance, then it returns directly.\n\n    Args:\n        module (nn.Module): Module for which this logic tries to run. It may or\n            may not be the root. If not, then this method does not do anything.\n    "
    with torch.profiler.record_function('FullyShardedDataParallel._root_pre_forward'):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, 'Expects a root FSDP to have been set')
        if not state._is_root:
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return (args, kwargs)
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            should_cast_buffers_to_full_prec = True
        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(buffers=dict(module.named_buffers()).values(), buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()), device=state.compute_device)
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, '_needs_buffer_dtype_restore_check', False):
            (buffers, buffer_dtypes_for_computation) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any((buffer.dtype != buffer_dtype_for_computation for (buffer, buffer_dtype_for_computation) in zip(buffers, buffer_dtypes_for_computation))):
                    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes_for_computation, state.compute_device)
            state._needs_buffer_dtype_restore_check = False
        if state.forward_prefetch:
            handles = []
            for fsdp_state in state._all_fsdp_states:
                if fsdp_state._handle:
                    handles.append(fsdp_state._handle)
            for handle in handles:
                handle._needs_pre_forward_unshard = True
                handle._prefetched = False
        _wait_for_computation_stream(state._device_handle.current_stream(), state._unshard_stream, state._pre_unshard_stream)
        _reset_flat_param_grad_info_if_needed(state._all_handles)
        with torch.profiler.record_function('FullyShardedDataParallel._to_kwargs'):
            (args_tuple, kwargs_tuple) = _to_kwargs(args, kwargs, state.compute_device, False)
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]
        return _root_cast_forward_input(state, module, args, kwargs)

@no_type_check
def _root_cast_forward_input(state: _FSDPState, module: torch.nn.Module, args, kwargs) -> Tuple[Any, Any]:
    if False:
        while True:
            i = 10
    if state._handle:
        force_full_precision = not state._handle._force_full_precision
    else:
        force_full_precision = True
    should_cast_forward_inputs = ((module.training or not state._use_full_prec_in_eval) and force_full_precision) and state.mixed_precision.cast_root_forward_inputs
    if should_cast_forward_inputs:
        input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
        (args, kwargs) = _cast_forward_inputs(input_dtype, *args, **kwargs)
    return (args, kwargs)

@no_type_check
def _pre_backward_hook(state: _FSDPState, module: nn.Module, handle: FlatParamHandle, *unused: Any) -> Any:
    if False:
        print('Hello World!')
    "\n    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.\n\n    Args:\n        module (nn.Module): Fully sharded module (see [Note: Fully Sharded\n            Module]).\n    "
    if handle and handle._ran_pre_backward_hook:
        return
    with torch.profiler.record_function('FullyShardedDataParallel._pre_backward_hook'):
        if state._is_root and (not state._post_backward_callback_queued):
            _register_post_backward_final_callback(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        elif handle:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        state.training_state = TrainingState.FORWARD_BACKWARD
        if not handle:
            return
        handle._training_state = HandleTrainingState.BACKWARD_PRE
        if handle._needs_pre_backward_unshard:
            if not handle._prefetched:
                _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
            state._device_handle.current_stream().wait_stream(state._unshard_stream)
        handle._needs_pre_backward_unshard = False
        with torch.profiler.record_function('FullyShardedDataParallel._pre_backward_prefetch'):
            _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
        handle.prepare_gradient_for_backward()
        handle._ran_pre_backward_hook = True

@no_type_check
@torch.no_grad()
def _post_backward_hook(state: _FSDPState, handle: FlatParamHandle, *unused: Any):
    if False:
        return 10
    "\n    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.\n\n    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the\n    unsharded gradient for the local batch.\n\n    Postcondition:\n    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced\n    unsharded gradient.\n    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded\n    gradient (accumulating with any existing gradient).\n    "
    _log_post_backward_hook(state, handle, log)
    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function('FullyShardedDataParallel._post_backward_hook'):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        _p_assert(handle._training_state in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST), f'Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}')
        handle._training_state = HandleTrainingState.BACKWARD_POST
        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError('FSDP does not support gradients of gradients')
        _post_backward_reshard(state, handle)
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return
        state._post_backward_stream.wait_stream(state._device_handle.current_stream())
        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if not _low_precision_hook_enabled(state) and flat_param.grad.dtype != handle._reduce_dtype and (not handle._force_full_precision):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                _reduce_grad_no_shard(state, handle)
            _no_dispatch_record_stream(autograd_computed_grad, state._post_backward_stream)

def _post_backward_reshard(state: _FSDPState, handle: FlatParamHandle, *unused: Any) -> None:
    if False:
        print('Hello World!')
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    _reshard(state, handle, free_unsharded_flat_param)
    with torch.profiler.record_function('FullyShardedDataParallel._post_backward_prefetch'):
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)

@no_type_check
def _should_free_in_backward(state: _FSDPState, handle: FlatParamHandle) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Returns whether FSDP should free the unsharded flat parameter in the\n    post-backward or not.\n    '
    if not handle.uses_sharded_strategy:
        return False
    return state._sync_gradients or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES

@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    if False:
        print('Hello World!')
    '\n    For sharded strategies, this runs gradient reduction, sharded gradient\n    accumulation if needed, and the post-reduction callback.\n    '
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (HandleShardingStrategy.HYBRID_SHARD, HandleShardingStrategy._HYBRID_SHARD_ZERO2)
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    (padded_unsharded_grad, new_sharded_grad) = _get_reduce_scatter_tensors(state, unsharded_grad)
    if state._comm_hook is None:
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        dist.reduce_scatter_tensor(new_sharded_grad, padded_unsharded_grad, group=state.process_group)
        if uses_hybrid_sharded_strategy:
            state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)

@no_type_check
def _get_reduce_scatter_tensors(state: _FSDPState, unsharded_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        while True:
            i = 10
    '\n    Returns the input and output tensors to reduce-scatter, respectively.\n    '
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    new_sharded_grad = torch.empty_like(chunks[0])
    return (padded_unsharded_grad, new_sharded_grad)

@no_type_check
def _accumulate_sharded_grad(state: _FSDPState, handle: FlatParamHandle, sharded_grad: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Accumulates the reduce-scattered sharded gradient with any existing sharded\n    gradient if needed, returning the gradient to offload (if CPU offloading is\n    enabled).\n    '
    flat_param = handle.flat_param
    _cast_grad_to_param_dtype(state, sharded_grad, flat_param)
    accumulate_grad = hasattr(flat_param, '_saved_grad_shard')
    if accumulate_grad:
        _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        flat_param._saved_grad_shard += sharded_grad
    else:
        flat_param._saved_grad_shard = sharded_grad
    grad_to_offload = flat_param._saved_grad_shard
    return grad_to_offload

@no_type_check
def _reduce_grad_no_shard(state: _FSDPState, handle: FlatParamHandle) -> None:
    if False:
        return 10
    '\n    For no-shard, this runs gradient reduction (which directly covers any\n    gradient accumulation implicitly) and the post-reduction callback.\n    '
    flat_param = handle.flat_param
    if state._comm_hook is None:
        _div_if_needed(flat_param.grad, state._gradient_predivide_factor)
        dist.all_reduce(flat_param.grad, group=state.process_group)
        _div_if_needed(flat_param.grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, flat_param.grad)
    if not handle._keep_low_precision_grads:
        _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
    grad_to_offload = flat_param.grad.data
    _post_reduce_grad_callback(state, handle, grad_to_offload)

@no_type_check
def _post_reduce_grad_callback(state: _FSDPState, handle: FlatParamHandle, grad_to_offload: torch.Tensor):
    if False:
        print('Hello World!')
    '\n    This callback captures any logic to run after the gradient reduction\n    finishes. Currently, this offloads the gradient to CPU if CPU offloading is\n    enabled and uses sharded gradient views if ``use_orig_params=True``.\n    '
    _offload_grad(state, handle, grad_to_offload)
    _post_backward_use_sharded_grad_views(handle)

@no_type_check
def _offload_grad(state: _FSDPState, handle: FlatParamHandle, grad_to_offload: torch.Tensor):
    if False:
        print('Hello World!')
    if not handle._offload_params:
        return
    non_blocking = handle.uses_sharded_strategy and (not handle._has_optim_in_backward)
    handle.flat_param._cpu_grad.copy_(grad_to_offload.detach(), non_blocking=non_blocking)
    _no_dispatch_record_stream(grad_to_offload.data, state._post_backward_stream)

@no_type_check
def _post_backward_use_sharded_grad_views(handle: FlatParamHandle):
    if False:
        i = 10
        return i + 15
    if not handle._use_orig_params:
        return
    handle._reset_is_grad_none()
    handle._use_sharded_grad_views()
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            if orig_param.grad is not None and hasattr(orig_param, '_in_backward_optimizers'):
                for optim in orig_param._in_backward_optimizers:
                    optim.step()
                optim.zero_grad(set_to_none=True)
        handle._reset_flat_param_grad_info_if_needed()
        if handle._offload_params:
            handle.flat_param._cpu_grad = None

def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if False:
        for i in range(10):
            print('nop')
    if div_factor > 1:
        tensor.div_(div_factor)

@no_type_check
def _cast_grad_to_param_dtype(state: _FSDPState, sharded_grad: torch.Tensor, param: FlatParameter):
    if False:
        while True:
            i = 10
    '\n    Casts ``sharded_grad`` back to the full parameter dtype so that the\n    optimizer step runs with that dtype. This performs an actual cast if\n    1. parameters were in reduced precision during the forward since then\n    gradients would be in that reduced precision, or\n    2. parameters were not in reduced precision but gradients were in\n    reduced precision for communication.\n    However, if a low precision communication hook is registered, then this\n    dtype cast happens in the hook instead.\n    '
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        _no_dispatch_record_stream(low_prec_grad_data, state._device_handle.current_stream())

def _check_grad_to_accumulate(new_sharded_grad: torch.Tensor, accumulated_grad: torch.Tensor) -> None:
    if False:
        print('Hello World!')
    _p_assert(accumulated_grad.shape == new_sharded_grad.shape, f'Shape mismatch when accumulating gradients: existing gradient shape={accumulated_grad.shape} new gradient shape={new_sharded_grad.shape}')
    _p_assert(accumulated_grad.device == new_sharded_grad.device, f'Device mismatch when accumulating gradients: existing gradient device={accumulated_grad.device} new gradient device={new_sharded_grad.device}')

@no_type_check
def _low_precision_hook_enabled(state: _FSDPState) -> bool:
    if False:
        print('Hello World!')
    return state._comm_hook in LOW_PRECISION_HOOKS

@no_type_check
@torch.no_grad()
def _post_backward_final_callback(state: _FSDPState, module: nn.Module):
    if False:
        i = 10
        return i + 15
    '\n    This waits for the post-backward to finish and performs some final cleanup.\n    This runs at the end of the entire backward pass and should only be called\n    on the root FSDP instance.\n    '
    _p_assert(state._is_root, 'The post-backward callback should only be called on the root FSDP instance')
    root_state = state
    if root_state._sync_gradients:
        current_stream = state._device_handle.current_stream()
        current_stream.wait_stream(root_state._post_backward_stream)
        if root_state._all_reduce_stream is not current_stream:
            current_stream.wait_stream(root_state._all_reduce_stream)
        if root_state.cpu_offload.offload_params:
            state._device_handle.current_stream().synchronize()
    root_state._exec_order_data.next_iter()
    for fsdp_state in state._all_fsdp_states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state.training_state = TrainingState.IDLE
        handle = fsdp_state._handle
        if handle:
            handle._ran_pre_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
    root_state._post_backward_callback_queued = False

@no_type_check
def _catch_all_reshard(state: _FSDPState) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Reshards the parameters that may not have been resharded in the\n    post-backward hook. This can happen when a module's output is used in the\n    forward pass, meaning that its pre-backward hook runs (unsharding the\n    parameter), but the post-backward hook does not run because the output was\n    not jused in the loss computation corresponding to this backward pass.\n    "
    try:
        if state._handle:
            already_resharded = state._handle.flat_param.data_ptr() == state._handle.flat_param._local_shard.data_ptr() and (not state._handle._skipped_use_sharded_views)
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(False, f'Got exception in the catch-all reshard for {state}: {str(e)}', raise_assertion_error=False)
        raise e

@no_type_check
def _finalize_params(state: _FSDPState) -> None:
    if False:
        print('Hello World!')
    'Finalizes the parameters before the next iteration.'
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if hasattr(flat_param, '_post_backward_hook_state'):
        post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
        expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
        _p_assert(post_backward_hook_state_len == expected_post_backward_hook_state_len, f'Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}')
        flat_param._post_backward_hook_state[-1].remove()
        delattr(flat_param, '_post_backward_hook_state')
    if flat_param.requires_grad:
        if not state._sync_gradients:
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        _p_assert(hasattr(flat_param, '_post_backward_called'), 'Expects `_post_backward_called` to be set on the `FlatParameter`')
        flat_param._post_backward_called = False

@no_type_check
def _prefetch_handle(state: _FSDPState, current_handle: Optional[FlatParamHandle], prefetch_mode: _PrefetchMode) -> None:
    if False:
        return 10
    '\n    Prefetches the next handles if needed (without synchronization). An empty\n    handles key cannot prefetch.\n    '
    if not current_handle:
        return
    handle = _get_handle_to_prefetch(state, current_handle)
    if not handle:
        return
    prev_training_state = handle._training_state
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        handle._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError(f'Invalid prefetch mode on rank {state.rank}: {prefetch_mode}')
    _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True

@no_type_check
def _get_handle_to_prefetch(state: _FSDPState, current_handle: FlatParamHandle) -> FlatParamHandle:
    if False:
        print('Hello World!')
    '\n    Returns a :class:`list` of the handles keys to prefetch for the next\n    module(s), where ``current_handle`` represents the current module.\n\n    "Prefetching" refers to running the unshard logic early (without\n    synchronization), and the "next" modules depend on the recorded execution\n    order and the current training state.\n    '
    training_state = _get_training_state(current_handle)
    valid_training_states = (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST, HandleTrainingState.FORWARD)
    _p_assert(training_state in valid_training_states, f'Prefetching is only supported in {valid_training_states} but currently in {training_state}')
    eod = state._exec_order_data
    target_handle: Optional[FlatParamHandle] = None
    if training_state == HandleTrainingState.BACKWARD_PRE and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE or (training_state == HandleTrainingState.BACKWARD_POST and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        if target_handle_candidate and target_handle_candidate._needs_pre_backward_unshard and (not target_handle_candidate._prefetched):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        if target_handle_candidate and target_handle_candidate._needs_pre_forward_unshard and (not target_handle_candidate._prefetched):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    return target_handle

def _get_training_state(handle: FlatParamHandle) -> HandleTrainingState:
    if False:
        i = 10
        return i + 15
    'Returns the training state of the handles in ``handle``.'
    _p_assert(handle, 'Expects a non-empty handle')
    return handle._training_state

@no_type_check
def _register_pre_forward_hook(state: _FSDPState, module: nn.Module) -> None:
    if False:
        while True:
            i = 10
    '\n    Registers a pre-forward hook on ``module``.\n    '
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(_pre_forward, state, module_param_handle, _pre_forward_unshard)
    state._pre_forward_handles.append(module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True))

@no_type_check
def _register_post_forward_hook(state: _FSDPState, module: nn.Module) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Registers a post-forward hook on ``module``. Even if the module has no\n    handles, we should register the hook since it will register the module's\n    pre-backward hook.\n    "
    for forward_handle in state._post_forward_handles:
        forward_handle.remove()
    state._post_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(_post_forward, state, module_param_handle, _post_forward_reshard)
    state._post_forward_handles.append(module.register_forward_hook(hook))

@no_type_check
def _register_root_pre_forward_hook(state: _FSDPState, module: nn.Module):
    if False:
        while True:
            i = 10
    '\n    Registers root pre-forward hook on ``module``, which should be the local\n    FSDP root.\n\n    NOTE: For the current composable FSDP design, we have each application of\n    ``fully_shard()`` to a module to indicate that that module is the local\n    FSDP root. We may remove this assumption in the future, in which case we\n    will need to register this root pre-forward hook on any candidate module\n    that may be the local FSDP root.\n    '
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    state._root_pre_forward_handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    state._root_pre_forward_handles.append(module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True))

@no_type_check
def _register_pre_backward_hooks(state: _FSDPState, module: nn.Module, outputs: Any, handle: FlatParamHandle) -> None:
    if False:
        print('Hello World!')
    '\n    Registers pre-backward hooks on the tensors that require gradients in the\n    forward pass outputs ``outputs``, which were computed using the\n    ``FlatParameter`` s of ``handles``.\n\n    Args:\n        module (nn.Module): Fully sharded module (see [Note: Fully Sharded\n            Module]).\n\n    Returns:\n        Forward pass outputs with pre-backward hooks registered to tensors that\n        require gradients.\n    '
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False
    if handle:
        handle._needs_pre_backward_unshard = False
        handle._ran_pre_backward_hook = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        if t.requires_grad:
            t.register_hook(functools.partial(_pre_backward_hook, state, module, handle))
            if handle:
                handle._needs_pre_backward_unshard = True
        return t
    return _apply_to_tensors(_register_hook, outputs)

def _register_post_backward_hook(state: _FSDPState, handle: Optional[FlatParamHandle]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Registers post-backward hooks on the ``FlatParameter`` s'\n    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.\n\n    The ``AccumulateGrad`` object represents the last function that finalizes\n    the ``FlatParameter`` 's gradient, so it only runs after its entire\n    gradient computation has finished.\n\n    We register the post-backward hook only once in the *first* forward that a\n    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``\n    object being preserved through multiple forwards.\n\n    NOTE: We follow this heuristic to prefer the *first* forward to target the\n    parameter mixed precision case, where there are *separate*\n    ``AccumulateGrad`` objects across the different forwards. (Without\n    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If\n    we instead prefer the *last* forward, then the hook runs early.\n    "
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, '_post_backward_hook_state')
    if already_registered or not flat_param.requires_grad:
        return
    temp_flat_param = flat_param.expand_as(flat_param)
    _p_assert(temp_flat_param.grad_fn is not None, 'The `grad_fn` is needed to access the `AccumulateGrad` and register the post-backward hook')
    acc_grad = temp_flat_param.grad_fn.next_functions[0][0]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(functools.partial(_post_backward_hook, state, handle))
    flat_param._post_backward_hook_state = (acc_grad, hook_handle)

def _register_post_backward_reshard_only_hook(state: _FSDPState, handle: Optional[FlatParamHandle], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Registers post-backward hooks to reshard flat parameters that do not\n    require gradient. We register these using multi-post-grad hooks on the\n    input activations to ensure that all gradients that may depend on the\n    parameters have been computed before resharding.\n    '
    if not torch.is_grad_enabled():
        return
    inp_tensors: Optional[List[torch.Tensor]] = None
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, '_post_backward_hook_state')
    if already_registered or flat_param.requires_grad:
        return
    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad]
    assert inp_tensors is not None
    hook_handle = register_multi_grad_hook(inp_tensors, functools.partial(_post_backward_reshard, state, handle))
    flat_param._post_backward_hook_state = (hook_handle,)

@no_type_check
def _register_post_backward_final_callback(state: _FSDPState, module: nn.Module) -> None:
    if False:
        return 10
    '\n    Registers the post-backward final callback that runs at the end of the\n    backward pass. This should be called from the root FSDP instance at the\n    beginning of the pre-backward.\n    '
    _p_assert(state._is_root, 'Only the root FSDP instance should register the post-backward callback')
    if state._post_backward_callback_queued:
        return
    _assert_in_training_states(state, [TrainingState.IDLE])
    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(functools.partial(_post_backward_final_callback, state, module))

def _wait_for_computation_stream(computation_stream: torch.Stream, unshard_stream: torch.Stream, pre_unshard_stream: torch.Stream):
    if False:
        while True:
            i = 10
    "\n    Has the unshard and pre-unshard streams wait for the computation stream.\n    For example, this should be called in the FSDP root's pre-forward to\n    respect optimizer step computation.\n    "
    unshard_stream.wait_stream(computation_stream)
    pre_unshard_stream.wait_stream(computation_stream)

def _reset_flat_param_grad_info_if_needed(handles: List[FlatParamHandle]):
    if False:
        i = 10
        return i + 15
    "\n    Clears the original parameters' gradients if needed. This method's CPU\n    overhead is minimal, so we may call it throughout FSDP methods, which serve\n    as callsites to free the gradient memory earlier.\n    "
    if not isinstance(handles, list):
        handles = [handles]
    for handle in handles:
        if handle._use_orig_params:
            handle._reset_flat_param_grad_info_if_needed()

@no_type_check
def _get_buffers_and_dtypes_for_computation(state: _FSDPState, root_module: nn.Module) -> Tuple[List[torch.Tensor], List[Optional[torch.dtype]]]:
    if False:
        return 10
    '\n    Returns all buffers in the module tree rooted at ``root_module`` and a\n    corresponding list of the buffer dtypes for computation. Each buffer dtype\n    is either ``None`` if buffer mixed precision is not enabled or the buffer\n    low precision dtype otherwise.\n    '
    _p_assert(state._is_root, 'Expects the root to cast buffers')
    buffers: List[torch.Tensor] = []
    buffer_dtypes: List[Optional[torch.dtype]] = []
    visited_buffers: Set[torch.Tensor] = set()
    (fsdp_states, fsdp_modules) = traversal_utils._get_fsdp_states_with_modules(root_module)
    for (fsdp_state, fsdp_module) in zip(reversed(fsdp_states), reversed(fsdp_modules)):
        for (buffer_name, buffer) in fsdp_module.named_buffers():
            if buffer in visited_buffers:
                continue
            visited_buffers.add(buffer)
            if clean_tensor_name(buffer_name) in fsdp_state._ignored_buffer_names:
                continue
            buffers.append(buffer)
            buffer_dtypes.append(fsdp_state.mixed_precision.buffer_dtype)
    assert len(buffers) == len(buffer_dtypes), f'{len(buffers)} {len(buffer_dtypes)}'
    return (buffers, buffer_dtypes)

@no_type_check
def _get_orig_buffer_dtypes(state: _FSDPState, buffer_names: List[str]) -> List[torch.dtype]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the original buffer types of the given buffer names.\n    '
    buffer_dtypes: List[torch.dtype] = []
    for buffer_name in buffer_names:
        _p_assert(buffer_name in state._buffer_name_to_orig_dtype, f'{buffer_name} is missing from pre-computed dict on rank {state.rank}, which only has keys {state._buffer_name_to_orig_dtype.keys()}')
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    return buffer_dtypes

def _cast_buffers_to_dtype_and_device(buffers: List[torch.Tensor], buffer_dtypes: List[Optional[torch.dtype]], device: torch.device) -> None:
    if False:
        return 10
    '\n    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them\n    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the\n    corresponding buffer is only moved to ``device``.\n    '
    _p_assert(buffer_dtypes is None or len(buffers) == len(buffer_dtypes), f'Expects `buffers` and `buffer_dtypes` to have the same length if `buffer_dtypes` is specified but got {len(buffers)} and {len(buffer_dtypes)}')
    for (buffer, buffer_dtype) in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)