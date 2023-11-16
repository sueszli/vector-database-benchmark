"""
This file includes private common utilities for FSDP.
"""
import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, Generator, Iterable, List, no_type_check, Optional, Set, Tuple, Type
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import FullOptimStateDictConfig, FullStateDictConfig, OptimStateDictConfig, ShardingStrategy, StateDictConfig, StateDictType
FSDP_WRAPPED_MODULE = '_fsdp_wrapped_module'
FSDP_PREFIX = FSDP_WRAPPED_MODULE + '.'
FSDP_FLATTENED = '_fsdp_flattened'
_MODULE_TO_INP_DTYPE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any=None):
        if False:
            print('Hello World!')
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError as exc:
                raise AttributeError(f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'.") from exc
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> '_FSDPDeviceHandle':
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an device handle corresponding to the device, and through this handle,\n        operations with the same semantics as CUDA can be performed on the device.\n        Just return torch.cuda if the device is cuda to make attribute-access faster.\n        Custom backend must first register a module with the same name with {device.type} on torch.\n        '
        if device.type == 'cuda':
            return cast(_FSDPDeviceHandle, torch.cuda)
        return cls(device)

    def __getattr__(self, __name: str) -> Any:
        if False:
            while True:
                i = 10
        try:
            return getattr(self.__backend, __name)
        except AttributeError as exc:
            raise AttributeError(f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{__name}'") from exc

class _UninitializedDeviceHandle(_FSDPDeviceHandle):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def __getattribute__(self, __name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Trying to use an uninitialized device handle.')

class _FSDPState(_State):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._ignored_modules: Set[nn.Module] = set()
        self._ignored_params: Set[nn.Parameter] = set()
        self._ignored_buffer_names: Set[str] = set()
        self.process_group: Optional[dist.ProcessGroup] = None
        self.rank: int = -1
        self.world_size: int = -1
        self._device_mesh: Optional[DeviceMesh] = None
        self.sharding_strategy = ShardingStrategy.FULL_SHARD
        self._use_orig_params: bool = False
        self.training_state = TrainingState.IDLE
        self._unshard_params_ctx: Dict[nn.Module, Generator] = {}
        self._state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._is_root: Optional[bool] = None
        self._handle: Optional[flat_param_file.FlatParamHandle] = None
        self._fully_sharded_module_to_handle: Dict[nn.Module, Optional[flat_param_file.FlatParamHandle]] = {}
        self.compute_device: Optional[torch.device] = None
        self._gradient_predivide_factor: int = 0
        self._gradient_postdivide_factor: int = 0
        self._comm_hook: Optional[Callable] = None
        self._comm_hook_state: Optional[Any] = None
        self._device_handle: _FSDPDeviceHandle = _UninitializedDeviceHandle()
        self._all_fsdp_states: List[_FSDPState] = []
        self._all_handles: List[flat_param_file.FlatParamHandle] = []
        self._fsdp_extension: Optional[FSDPExtensions] = None

def _get_module_fsdp_state(module: nn.Module) -> Optional[_FSDPState]:
    if False:
        for i in range(10):
            print('nop')
    state = _get_module_state(module)
    if state is None or not isinstance(state, _FSDPState):
        return None
    return state

def _get_module_fsdp_state_if_fully_sharded_module(module: nn.Module) -> Optional[_FSDPState]:
    if False:
        print('Hello World!')
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state == module:
        return state
    if module in state._fully_sharded_module_to_handle:
        return state
    return None

class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """
    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()

class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()

def _is_composable(state: _FSDPState):
    if False:
        print('Hello World!')
    return not isinstance(state, nn.Module)

@no_type_check
def _module_handle(state: _FSDPState, module: nn.Module) -> Optional['FlatParamHandle']:
    if False:
        while True:
            i = 10
    '\n    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is\n    the handle that contains some parameter in ``module``.\n    '
    if _is_composable(state):
        if state._handle is None:
            return None
        assert module in state._fully_sharded_module_to_handle, f'Expects a fully sharded module but got {module} on rank {state.rank}'
        return state._fully_sharded_module_to_handle[module]
    else:
        return module._handle

@no_type_check
def _has_fsdp_params(state: _FSDPState, module: nn.Module) -> bool:
    if False:
        print('Hello World!')
    'Returns if ``module`` has parameters managed by FSDP.'
    return _module_handle(state, module) is not None

def _get_sharding_strategy(handle):
    if False:
        print('Hello World!')
    '\n    Returns the sharding strategy of the handle.\n    '
    return handle._sharding_strategy if handle else None

def clean_tensor_name(tensor_name: str) -> str:
    if False:
        return 10
    '\n    Cleans the parameter or buffer name by removing any module wrapper\n    prefixes.\n    '
    tensor_name = tensor_name.replace(FSDP_PREFIX, '')
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX, '')
    return tensor_name

def _set_fsdp_flattened(tensor: torch.Tensor) -> None:
    if False:
        while True:
            i = 10
    '\n    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to\n    avoid re-flattening it during nested construction.\n    '
    setattr(tensor, FSDP_FLATTENED, True)

def _is_fsdp_flattened(tensor: torch.Tensor) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns if ``tensor`` has been marked as flattened by FSDP.'
    return getattr(tensor, FSDP_FLATTENED, False)

def _named_parameters_with_duplicates(module: nn.Module, **kwargs: Any) -> List[Tuple[str, nn.Parameter]]:
    if False:
        while True:
            i = 10
    '\n    This API is required as some modules overwrite `named_parameters()` but do not support\n    `remove_duplicate`.\n    '
    assert 'remove_duplicate' not in kwargs, '_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument.'
    kwargs['remove_duplicate'] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError as e:
        kwargs.pop('remove_duplicate')
        ret = list(module.named_parameters(**kwargs))
    return ret

def _get_param_to_fqns(model: torch.nn.Module, dedup_shared_params: bool=True) -> Dict[nn.Parameter, List[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Constructs a mapping from parameter to a list of its "canonical" FQNs. Here,\n    we use canonical to mean the fully-qualified name assigned to the parameter\n    based on its position in the original nn.Module hierarchy before any wrapper\n    or parallelism has been applied to it. This is in contrast to FQNs that may be\n    generated after parallelisms or wrappers have been applied to the model.\n\n    Each normal parameter maps to a singleton list containing its FQN, while each\n    ``FlatParameter`` maps to a list of its original parameter FQNs, which may\n    have length greater than one.  All FQNs are prefixed starting from ``model``.\n\n    In the case where FSDP was applied with ``use_orig_params=True``, there should be no\n    ``FlatParameter`` s registered to the model\'s modules and this mapping will only\n    contain mappings from ``nn.Parameter`` s to singleton FQN lists.\n\n    It is only in the case where FSDP was applied with ``use_orig_params=False`` where\n    a ``FlatParameter`` will be registered in place of the original parameters and there\n    will be mappings from each ``FlatParameter`` to lists of FQNs corresponding to the\n    original parameters.\n\n    Args:\n        model (torch.nn.Module): Root module (which may or may not be a\n            :class:`FullyShardedDataParallel` instance).\n        dedup_shared_params (bool): For shared parameters, if ``True``, only\n            includes the FQNs corresponding to the first encounter of the\n            shared parameter in the module traversal; if ``False``, then\n            includes the FQNs across all encounters. (Default: ``True``)\n    '

    def module_fn(module, prefix, tree_level, param_to_fqns):
        if False:
            i = 10
            return i + 15
        for (param_name, param) in _named_parameters_with_duplicates(module, recurse=False):
            local_fqns = param._fqns if isinstance(param, flat_param_file.FlatParameter) else [param_name]
            global_fqns = [clean_tensor_name(prefix + name) for name in local_fqns]
            is_shared_param = param in param_to_fqns
            if not is_shared_param:
                param_to_fqns[param] = global_fqns
            elif isinstance(param, flat_param_file.FlatParameter):
                warnings.warn('FlatParameter is being traversed more than once. This case should only happen when using DistributedModelParallel with FullyShardedDataParallel.')
                param_to_fqns[param] = global_fqns
            elif not dedup_shared_params:
                param_to_fqns[param].extend(global_fqns)

    def return_fn(param_to_fqns):
        if False:
            i = 10
            return i + 15
        return param_to_fqns
    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    return _apply_to_modules(model, module_fn, return_fn, [key for (key, _) in _named_parameters_with_duplicates(model)], param_to_unflat_param_names)

@no_type_check
def _log_post_backward_hook(state: _FSDPState, handle: 'FlatParamHandle', log: logging.Logger) -> None:
    if False:
        for i in range(10):
            print('nop')
    if state._use_orig_params and handle._debug_level == dist.DebugLevel.INFO:
        param_fqns = _get_handle_fqns_from_root(state, handle)
        log.warning('FSDP firing post-backward hooks for parameters %s', param_fqns)

@no_type_check
def _get_handle_fqns_from_root(state: _FSDPState, handle: 'FlatParamHandle') -> Optional[List[str]]:
    if False:
        while True:
            i = 10
    if handle is None:
        return None
    param_to_fqn = state._exec_order_data.param_to_fqn
    handle_params = handle.flat_param._params
    param_fqns = [fqn for fqn_list in [param_to_fqn[p] for p in handle_params] for fqn in fqn_list]
    return param_fqns

def _apply_to_modules(root_module: torch.nn.Module, module_fn: Callable, return_fn: Callable, filter_fqns: Optional[List[str]]=None, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Performs a pre-order traversal of the modules in the hierarchy rooted at\n    ``root_module``, applying ``module_fn`` at each module and finally\n    returning a value using ``return_fn``. The traversal constructs the full\n    module prefix name (e.g. "module.submodule." just like in model state dict)\n    and makes that available to ``module_fn``.\n\n    ``filter_fqns`` is used because some module may have its own prefix similar\n    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten\n    to remove the prefix.\n    '

    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for (submodule_name, submodule) in module.named_children():
            if submodule is None:
                continue
            new_prefix = prefix + submodule_name + '.'
            new_tree_level = tree_level + 1
            if filter_fqns is not None:
                for fqn in filter_fqns:
                    if fqn.startswith(new_prefix):
                        break
                else:
                    if submodule_name == '_fsdp_wrapped_module' or submodule_name == '_dmp_wrapped_module':
                        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                            warnings.warn(f'An unexpected prefix is detected. This case  should only happen when using DMP with FSDP. prefix = {prefix}, submodule_name = {submodule_name}')
                        new_prefix = prefix
                    elif submodule_name == 'module':
                        warnings.warn(f'An unexpected prefix is detected. This case  should only happen when DDP wraps the outer  modules while FSDP wraps the inner ones.prefix = {prefix}, submodule_name = {submodule_name}')
                        new_prefix = prefix
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)
    f(root_module, '', 0, *args, **kwargs)
    return return_fn(*args, **kwargs)

@no_type_check
def _assert_in_training_states(state: _FSDPState, training_states: List[TrainingState]) -> None:
    if False:
        while True:
            i = 10
    'Asserts that FSDP is in the states ``_training_states``.'
    if state.training_state not in training_states:
        msg = f'expected to be in states {training_states} but current state is {state.training_state}'
        if state.rank == 0:
            if isinstance(state, nn.Module):
                print(f'Asserting FSDP instance is: {state}')
            print(f'ERROR: {msg}')
            traceback.print_stack()
        raise ValueError(msg)

def _get_root_modules(modules: Set[nn.Module]) -> Set[nn.Module]:
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.\n        parent-less) with respect to the modules in the set itself. In other\n        words, these are the modules in ``modules`` that are not the child of\n        any other module in ``modules``.\n    '
    root_modules: Set[nn.Module] = set()
    module_to_submodules = {module: set(module.modules()) for module in modules}
    for candidate_module in modules:
        is_root_module = True
        for (module, submodules) in module_to_submodules.items():
            is_child_module = candidate_module is not module and candidate_module in submodules
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.add(candidate_module)
    return root_modules

def _override_module_mixed_precision(root: torch.nn.Module, module_classes_to_override: Iterable[Type[nn.Module]], wrap_override_dict: Dict[str, Any]={'mixed_precision': None}) -> Set[Type[nn.Module]]:
    if False:
        for i in range(10):
            print('nop')
    module_classes_to_override = tuple(set(module_classes_to_override))
    overridden_module_classes: Set[Type[nn.Module]] = set()
    for mod in root.modules():
        if isinstance(mod, module_classes_to_override):
            overridden_module_classes.add(type(mod))
            mod._wrap_overrides = wrap_override_dict

            def cast_fn(dtype: torch.dtype, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                _MODULE_TO_INP_DTYPE[module] = x.dtype
                return x.to(dtype)

            def forward_pre_hook(module, args):
                if False:
                    while True:
                        i = 10
                return _apply_to_tensors(partial(cast_fn, torch.float32, module), args)

            def forward_post_hook(module, args, output):
                if False:
                    for i in range(10):
                        print('nop')
                if module in _MODULE_TO_INP_DTYPE:
                    old_dtype = _MODULE_TO_INP_DTYPE[module]
                    return _apply_to_tensors(partial(cast_fn, old_dtype, module), output)
            mod.register_forward_pre_hook(forward_pre_hook, prepend=False)
            mod.register_forward_hook(forward_post_hook, prepend=False)
    return overridden_module_classes

def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    if False:
        for i in range(10):
            print('nop')
    if tensor.device.type not in ['cuda', torch._C._get_privateuse1_backend_name()]:
        return
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        with no_dispatch():
            tensor.record_stream(stream)
    else:
        tensor.record_stream(stream)

def _same_storage_as_data_ptr(x: torch.Tensor, data_ptr: int) -> bool:
    if False:
        i = 10
        return i + 15
    return x._typed_storage()._data_ptr() == data_ptr