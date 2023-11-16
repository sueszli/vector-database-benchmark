import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, Callable, cast, Dict, Iterable, List, no_type_check, Optional, Set, Tuple, Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict, _offload_state_dict_to_cpu
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP, OptimStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictConfig, StateDictType
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module, FSDP_WRAPPED_MODULE
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
FLAT_PARAM = '_flat_param'
PG = 'param_groups'
PG_PREFIX = f'{PG}.'
STATE = 'state'
STATE_PREFIX = f'{STATE}.'
PARAMS = 'params'
FQNS_T = Set[str]
_patched_state_dict: Set[Callable] = set()
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[PrimitiveType, List[PrimitiveType], Tuple[PrimitiveType], Dict[str, 'ValueType']]
DictValueType = Dict[str, ValueType]
ListDictValueType = List[DictValueType]
OptimizerStateType = Dict[str, Union[DictValueType, ListDictValueType]]

@contextlib.contextmanager
def gc_context():
    if False:
        print('Hello World!')
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if is_enabled:
            gc.enable()

@dataclass
class StateDictOptions:
    """
    This dataclass specifies how get_state_dict/set_state_dict will work.

    - ``full_state_dict``: if this is set to True, all the tensors in the
      returned state_dict will be gathered. No ShardedTensor and DTensor
      will be in the returned state_dict.

    - ``cpu_offload``: offload all the tensors to cpu. To prevent CPU OOM, if
      ``full_state_dict`` is also true, then only the rank0 will get the
      state_dict and all other ranks will get empty state_dict.

    - ``ignore_frozen_params``: if the value is True, the returned state_dict
      won't contain any frozen parameters -- the ``requires_grad`` is False.
      The default value is False.

    - ``keep_submodule_prefixes``: when ``submodules`` is not None, this option
      indicates whether to keep the submodule prefixes from the state_dict keys.
      or example, if the submodule is ``module.pretrain`` and the full FQN of
      the parameter is ``pretrain.layer1.weight`` of the param. When this option
      is True, the parameter's key in the returned state_dict will be
      ``pretrain.layer1.weight``. If the options is False, the key will be
      ``layer1.weight``.
      Note that if ``keep_submodule_prefixes`` is False, there may be conflicted
      FQNs, hence there should be only one submodule in ``submodules``.

    - ``strict``: the ``strict`` option when ``set_state_dict`` calls
      model.load_state_dict().
      The default value is False.
    """
    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True

@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: Dict[Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]] = field(default_factory=dict)
    all_fqns: Set[str] = field(default_factory=set)
    submodule_prefixes: Set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)

def _get_fqns(model: nn.Module, name: str, skip_ddp_prefix: bool=True) -> FQNS_T:
    if False:
        for i in range(10):
            print('nop')
    "\n    This API is used to convert the name of a parameter to the FQNs. For FSDP\n    without `use_orig_params`, the name of FlatParameter can be mapped to\n    multiple original parameters. As a result, the return type of this function\n    is `Set[str]`.\n\n    Args:\n        module (nn.Module): the root model.\n        name (str): the name\n        skip_ddp_prefix (bool): whether to skip DDP's `module` prefix\n\n    Returns:\n        The canonical FQNs based on the model traversal.\n    "
    if '.' not in name:
        return {name}
    obj_names = name.split('.')
    fqn_obj_names = []
    curr_obj = model
    for (i, curr_obj_name) in enumerate(obj_names):
        if isinstance(curr_obj, DDP):
            assert curr_obj_name == 'module'
            curr_obj = curr_obj.module
            if not skip_ddp_prefix:
                fqn_obj_names.append(curr_obj_name)
        elif isinstance(curr_obj, FSDP):
            if obj_names[i + 1] == FLAT_PARAM:
                prefix = '.'.join(fqn_obj_names)
                flat_param = getattr(curr_obj, FLAT_PARAM)
                if prefix:
                    prefix = f'{prefix}.'
                return {f'{prefix}{fqn}' for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        else:
            fqn_obj_names.append(curr_obj_name)
            curr_obj = getattr(curr_obj, curr_obj_name)
    return {'.'.join(fqn_obj_names)}

def _verify_options(model: nn.Module, optims: Tuple[torch.optim.Optimizer, ...], optim_only: bool, *, submodules: Optional[Set[nn.Module]]=None, options: Optional[StateDictOptions]=None) -> _StateDictInfo:
    if False:
        i = 10
        return i + 15
    '\n    Verify the model and options passed by the user and generates _StateDictInfo.\n    '
    if optim_only and (not optims):
        raise RuntimeError('Optimizers are not passed in but optim_only is set to True.')
    options = options or StateDictOptions()
    fqn_param_mapping: Dict[Union[str, torch.Tensor], Union[Set[str], torch.Tensor]] = {}
    all_fqns = set()
    for (name, param) in model.named_parameters():
        fqns = _get_fqns(model, name)
        fqn_param_mapping[param] = fqns
        for fqn in fqns:
            fqn_param_mapping[fqn] = param
            all_fqns.add(fqn)
    submodule_prefixes = set()
    if submodules:
        submodules = set(submodules)
        for (name, module) in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            assert len(fqns) == 1, 'Submodule FQN should only have 1 instance'
            for fqn in fqns:
                submodule_prefixes.add(f'{fqn}.')
    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        if options.full_state_dict:
            state_dict_config = FullStateDictConfig(offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload)
            optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload)
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            state_dict_config = ShardedStateDictConfig()
            optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=options.cpu_offload)
            state_dict_type = StateDictType.SHARDED_STATE_DICT
        fsdp_context = functools.partial(FSDP.state_dict_type, module=model, state_dict_type=state_dict_type, state_dict_config=state_dict_config, optim_state_dict_config=optim_state_dict_config)
    else:
        fsdp_context = contextlib.nullcontext
    return _StateDictInfo(**asdict(options), fqn_param_mapping=fqn_param_mapping, all_fqns=all_fqns, submodule_prefixes=submodule_prefixes, fsdp_context=fsdp_context, fsdp_modules=cast(List[nn.Module], fsdp_modules), handle_model=not optim_only, handle_optim=len(optims) > 0)

def _verify_state_dict(model_state_dict: Dict[str, ValueType], optim_state_dict: OptimizerStateType, info: _StateDictInfo) -> None:
    if False:
        return 10
    has_fsdp_root = False
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, 'Expected a fsdp_state with a fsdp module.'
        if fsdp_state._is_root:
            has_fsdp_root = True
            break
    if info.fsdp_modules and (not has_fsdp_root):
        raise RuntimeError('The model has FSDP modules but no FSDP root module exists.')
    if info.handle_model and (not model_state_dict) and (not info.submodule_prefixes) and (not info.ignore_frozen_params) and (not (info.cpu_offload and info.full_state_dict)) and info.strict:
        raise RuntimeError(f'The option indicates that model state_dict is required to save or load, but model state_dict is empty.rank = dist.get_rank()={dist.get_rank()!r}.')
    if info.handle_optim:
        if not (optim_state_dict and optim_state_dict[STATE]) and (not (info.cpu_offload and info.full_state_dict)):
            raise RuntimeError(f'The option indicates that model state_dict is required to save, or load but optim state_dict is empty. {optim_state_dict}')
    for key in model_state_dict.keys():
        if FLAT_PARAM in key:
            raise RuntimeError(f'{key} contains {FLAT_PARAM}. This can happen if the model is not the root module.')

def _state_dict_fn(obj: Union[nn.Module, torch.optim.Optimizer], api: str) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call

def _get_model_state_dict(model: nn.Module, info: _StateDictInfo) -> Dict[str, ValueType]:
    if False:
        return 10
    if not info.handle_model:
        return {}
    with info.fsdp_context():
        state_dict = _state_dict_fn(model, 'state_dict')()
    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        assert len(fqns) == 1
        fqn = next(iter(fqns))
        if fqn != key:

            def verify(key, fqn) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                if len(fqn) >= len(key):
                    return False
                fqn_split = fqn.split('.')
                key_split = key.split('.')
                fqn_idx = 0
                for (key_idx, key_name) in enumerate(key_split):
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        if fqn_idx == len(fqn_split):
                            return key_idx == len(key_split) - 1
                    elif key_name == 'module':
                        continue
                    else:
                        return False
                return True
            if not verify(key, fqn):
                raise RuntimeError(f'An unexpected key, {key}, exists. FQN is {fqn}')
            state_dict[fqn] = state_dict.pop(key)
    if info.submodule_prefixes:
        new_state_dict: Dict[str, ValueType] = {}
        for fqn in state_dict.keys():
            for prefix in info.submodule_prefixes:
                if not fqn.startswith(prefix):
                    continue
                if info.keep_submodule_prefixes:
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    new_fqn = fqn[len(prefix):]
                    new_state_dict[new_fqn] = state_dict[fqn]
        state_dict = new_state_dict
    if info.ignore_frozen_params:
        for (key, param) in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)
    for (key, p) in list(state_dict.items()):
        if p.is_meta:
            state_dict.pop(key)
    if info.full_state_dict:
        ranks_only = tuple() if not info.cpu_offload else (0,)
        return _gather_state_dict(state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict

def _load_model_state_dict(model: nn.Module, state_dict: Dict[str, ValueType], info: _StateDictInfo) -> _IncompatibleKeys:
    if False:
        for i in range(10):
            print('nop')
    if not info.handle_model or not state_dict:
        return _IncompatibleKeys({}, {})
    for (key, _) in model.named_parameters():
        fqns = _get_fqns(model, key)
        fqns_with_ddp_prefix = _get_fqns(model, key, skip_ddp_prefix=False)
        for (fqn, fqn_with_ddp_prefix) in zip(fqns, fqns_with_ddp_prefix):
            if fqn != fqn_with_ddp_prefix:
                state_dict[fqn_with_ddp_prefix] = state_dict.pop(fqn)
    with info.fsdp_context():
        return cast(_IncompatibleKeys, _state_dict_fn(model, 'load_state_dict')(state_dict, strict=info.strict))

def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    if False:
        print('Hello World!')
    '\n    Initialize optim states by calling the step() with zero grads.\n    '
    if optim.state:
        return
    for param_group in optim.param_groups:
        for param in param_group[PARAMS]:
            if param.grad is not None:
                raise RuntimeError('state_dict can only be used if the optimizer states are initialized (usually after one step() with gradients) or gradients are None. For the later case, state_dict will fake the gradients as zero to initialize the optimizer states. However, the gradients are not None.')
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
    optim.step(closure=None)
    optim.zero_grad(set_to_none=True)

def _get_optim_state_dict(model: nn.Module, optimizers: Tuple[torch.optim.Optimizer, ...], info: _StateDictInfo) -> OptimizerStateType:
    if False:
        for i in range(10):
            print('nop')
    if not info.handle_optim:
        return {}
    optim_state_dict: OptimizerStateType = {STATE: {}, PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, 'state_dict')()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)
        else:
            params = list(chain.from_iterable((g[PARAMS] for g in optim.param_groups)))
            param_pid_mapping = dict(zip(params, range(len(params))))
            fqn_pid_mapping = {}
            for (key, param) in model.named_parameters():
                fqns = _get_fqns(model, key)
                assert len(fqns) == 1
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn
            for key in list(osd[STATE].keys()):
                fqn = fqn_pid_mapping[key]
                osd[STATE][fqn] = osd[STATE].pop(key)
            for group in osd[PG]:
                group[PARAMS] = [fqn_pid_mapping[pid] for pid in group[PARAMS]]
        if not osd:
            continue
        cast(DictValueType, optim_state_dict[STATE]).update(osd[STATE])
        cast(ListDictValueType, optim_state_dict[PG]).extend(osd[PG])
    if info.full_state_dict:
        ranks_only = tuple() if not info.cpu_offload else (0,)
        return _gather_state_dict(optim_state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(optim_state_dict)
    else:
        return optim_state_dict

def _split_optim_state_dict(model: nn.Module, optim: torch.optim.Optimizer, optim_state_dict: OptimizerStateType, info: _StateDictInfo) -> OptimizerStateType:
    if False:
        i = 10
        return i + 15
    '\n    Extract the corresponding optim state_dict from ``optim_state_dict`` for\n    ``optim`` and return the result optim state_dict.\n\n    Args:\n        model (nn.Module): the root model.\n        optim (torch.optim.Optimizer): the optimizer.\n        optim_state_dict (Dict[str, ValueType]): the superset optim state_dict that\n            contains the optim state_dict of ``optim``.\n        info (_StateDictInfo): state dict information.\n\n    Returns:\n        The optim state_dict of ``optim``.\n    '
    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {STATE: state, PG: pg_state}
    pg_mapping: Dict[int, int] = {}
    for param_group in optim.param_groups:
        pg_state.append({PARAMS: []})
        for param in param_group[PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                params = pg_state[-1][PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[STATE])[fqn]
                for loaded_param_group in cast(ListDictValueType, optim_state_dict[PG]):
                    params = loaded_param_group[PARAMS]
                    assert isinstance(params, list)
                    if fqn in params:
                        pg_mapping[id(loaded_param_group)] = len(return_osd[PG]) - 1
    for param_group in cast(ListDictValueType, optim_state_dict[PG]):
        idx = pg_mapping.get(id(param_group), -1)
        if idx == -1:
            continue
        for (key, value) in param_group.items():
            if key == PARAMS:
                continue
            pg_state[idx][key] = value
    return return_osd

def _load_optim_state_dict(model: nn.Module, optimizers: Tuple[torch.optim.Optimizer, ...], state_dict: OptimizerStateType, info: _StateDictInfo) -> None:
    if False:
        return 10
    if not info.handle_optim:
        return
    for optim in optimizers:
        optim_state_dict = _split_optim_state_dict(model, optim, state_dict, info)
        if info.fsdp_modules:
            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(model, optim, optim_state_dict)
        _init_optim_state(optim)
        _state_dict_fn(optim, 'load_state_dict')(optim_state_dict)

def get_model_state_dict(model: nn.Module, *, submodules: Optional[Set[nn.Module]]=None, options: Optional[StateDictOptions]=None) -> Dict[str, ValueType]:
    if False:
        return 10
    '\n    Return the model state_dict of ``model``.\n\n    See ``get_state_dict`` for the detail usage.\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        submodules: Optional[Set[nn.Module]]: only return the model parameters\n            that belong to the submodules.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be returned. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        The state_dict for ``model``.\n    '
    with gc_context():
        info = _verify_options(model, tuple(), optim_only=False, submodules=submodules, options=options)
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict

def get_optimizer_state_dict(model: nn.Module, optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], *, submodules: Optional[Set[nn.Module]]=None, options: Optional[StateDictOptions]=None) -> OptimizerStateType:
    if False:
        print('Hello World!')
    '\n    Return the combined state_dict for optimizers.\n\n    See ``get_state_dict`` for the detail usage.\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):\n            The optimizers that are used to optimize ``model``.\n        submodules: Optional[Set[nn.Module]]: only return the model parameters\n            that belong to the submodules.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be returned. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        The state_dict for ``optimizers``.\n    '
    with gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=True, submodules=submodules, options=options)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict({}, optim_state_dict, info)
        return optim_state_dict

def get_state_dict(model: nn.Module, optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], *, submodules: Optional[Set[nn.Module]]=None, options: Optional[StateDictOptions]=None) -> Tuple[Dict[str, ValueType], OptimizerStateType]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the model state_dict and optimizers state_dict.\n\n    ``get_state_dict`` can process any module that is parallelized by PyTorch\n    FSDP/fully_shard, DDP/replicate, tensor_parallel/parallelize_module, and any\n    combination of these parallelisms. The main functions of ``get_state_dict``\n    are: 1.) returning a model and optimizer state_dict that can be resharded\n    with a different number of trainers and/or different parallelisms.\n    2.) hiding the parallelism-specific state_dict APIs. Users don't have to call\n    these APIs.\n    3.) sanity checking the result state_dict.\n\n    The keys of the result state dictionary are the canonical FQNs (Fully\n    Qualified Names).  A canonical FQN refers to the FQN based on a parameter's\n    position in an nn.Module hierarchy. More specifically, a canonical FQN to a\n    parameter is the FQN returned by ``module.named_parameters()`` or\n    ``module.named_buffers()`` when the module is not distributed by any\n    parallelisms. Since the optimizer internally uses parameter IDs to represent\n    a parameter, there will be a conversion from the parameter IDs to the\n    canonical FQNs when calling this API.\n\n    ``get_state_dict`` can also process a module that is not parallelized. In\n    such a case, ``get_state_dict`` only performs one function -- converting the\n    optimizer parameter IDs to the canonical FQNs.\n\n    Example:\n\n        import torch\n        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n        from torch.nn.parallel import DistributedDataParallel as DDP\n        from torch.distributed.checkpoint.state_dict import get_state_dict\n\n        fsdp_model = FSDP(copy.deepcopy(model))\n        fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)\n        ddp_model = DDP(copy.deepcopy(model))\n        ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)\n\n\n        ddp_state_dict, ddp_optim_state_dict = get_state_dict(ddp_model, ddp_optim)\n        fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(fsdp_model, fsdp_optim)\n\n        # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),\n        # the asserts will fail.\n        assert ddp_state_dict == fsdp_state_dict\n        assert ddp_optim_state == fsdp_optim_state_dict\n\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):\n            The optimizers that are used to optimize ``model``.\n        submodules: Optional[Set[nn.Module]]: only return the model parameters\n            that belong to the submodules.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be returned. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        ``Tuple`` that contain model state_dict and optimizer state_dict.\n    "
    with gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=False, submodules=submodules, options=options)
        model_state_dict = _get_model_state_dict(model, info)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        return (model_state_dict, optim_state_dict)

def _unflatten_model_state_dict(model: nn.Module, state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]) -> Dict[str, ValueType]:
    if False:
        for i in range(10):
            print('nop')
    if not state_dict:
        return {}
    if isinstance(next(iter(state_dict.keys())), nn.Module):
        cast_state_dict = cast(Dict[nn.Module, Dict[str, ValueType]], state_dict)
        new_state_dict: Dict[str, ValueType] = {}
        for (submodule, sub_state_dict) in cast_state_dict.items():
            for (name, m) in model.named_modules():
                if m != submodule:
                    continue
                fqns = _get_fqns(model, name)
                assert len(fqns) == 1, 'FQNs for a submodule should only have 1 element'
                prefix = f'{next(iter(fqns))}.'
                new_state_dict.update({prefix + subfqn: value for (subfqn, value) in sub_state_dict.items()})
        return new_state_dict
    else:
        return cast(Dict[str, ValueType], state_dict)

def set_model_state_dict(model: nn.Module, model_state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]], *, options: Optional[StateDictOptions]=None) -> _IncompatibleKeys:
    if False:
        while True:
            i = 10
    'Load the model state_dict.\n\n    The counterpart of ``get_model_state_dict`` to set the state_dict to the\n    model. See ``set_state_dict`` for the detail usage.\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):\n           the model state_dict to load. If the key of the ``model_state_dict``\n           is nn.Module, the key is a submodule of ``model`` and the value should\n           be the state_dict of the submodule. When loading the state_dict,\n           the prefix of the submodule will be append to the state_dict.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be loaded. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:\n            * **missing_keys** is a list of str containing the missing keys\n            * **unexpected_keys** is a list of str containing the unexpected keys\n    '
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(model, model_state_dict)
    with gc_context():
        info = _verify_options(model, tuple(), optim_only=False, options=options)
        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)

def set_optimizer_state_dict(model: nn.Module, optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], *, optim_state_dict: OptimizerStateType, options: Optional[StateDictOptions]=None) -> None:
    if False:
        print('Hello World!')
    'Load the optimizers state_dict.\n\n    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the\n    optimizers. See ``set_state_dict`` for the detail usage.\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        optimizers (Union[Optimizer, Iterable[Optimizer]]):\n            The optimizers that are used to optimize ``model``.\n        optim_state_dict: OptimizerStateType:\n            the optimizer state_dict to load.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be loaded. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        None\n    '
    with gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=True, options=options)
        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)

def set_state_dict(model: nn.Module, optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], *, model_state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]], optim_state_dict: OptimizerStateType, options: Optional[StateDictOptions]=None) -> _IncompatibleKeys:
    if False:
        for i in range(10):
            print('nop')
    'Load the model state_dict and optimizers state_dict.\n\n    The counterpart of ``get_state_dict`` to set the state_dict to the model and\n    optimizers.  The given ``model_state_dict`` and ``optim_state_dict`` do not\n    have to be returned by ``get_state_dict`` but must meet the following\n    requirements: 1) all FQNs are canonical FQNs as defined in ``get_state_dict``,\n    2) if a tensor is sharded, it must be either a ShardedTensor or DTensor,\n    3) optimizer state_dict cannot contain the parameter IDs; the keys should be\n    the canonical FQNs.\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        optimizers (Union[Optimizer, Iterable[Optimizer]]):\n            The optimizers that are used to optimize ``model``.\n        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):\n           the model state_dict to load. If the key of the ``model_state_dict``\n           is nn.Module, the key is a submodule of ``model`` and the value should\n           be the state_dict of the submodule. When loading the state_dict,\n           the prefix of the submodule will be append to the state_dict.\n        optim_state_dict: OptimizerStateType:\n            the optimizer state_dict to load.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be loaded. See\n            `StateDictOptions` for the details.\n\n    Returns:\n        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:\n            * **missing_keys** is a list of str containing the missing keys of the model state_dict.\n            * **unexpected_keys** is a list of str containing the unexpected keys of the model state_dict.\n    '
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(model, model_state_dict)
    with gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=not model_state_dict, options=options)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)
        return _load_model_state_dict(model, model_state_dict, info)

@no_type_check
def _patch_model_state_dict(model: nn.Module, *, options: Optional[StateDictOptions]=None) -> None:
    if False:
        return 10
    'Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model``.\n\n    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``model`` to\n    be a partial function to call ``get_state_dict`` and ``set_state_dict``.\n\n    Example:\n        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n        from torch.distributed.checkpoint.state_dict import patch_model_state_dict\n\n        model = fsdp(model)\n        patch_model_state_dict(model)\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be loaded. See\n            `StateDictOptions` for the details.\n    Returns:\n        None\n    '
    _state_dict_call = functools.partial(get_model_state_dict, model=model, options=options)

    def state_dict_call():
        if False:
            while True:
                i = 10
        return _state_dict_call()
    model.state_dict = state_dict_call
    _load_state_dict_call = functools.partial(set_model_state_dict, model=model, options=options)

    def load_state_dict_call(state_dict: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        _load_state_dict_call(model_state_dict=state_dict)[1]
    model.load_state_dict = load_state_dict_call
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)

@no_type_check
def _patch_optimizer_state_dict(model: nn.Module, *, optimizers: Tuple[torch.optim.Optimizer, ...], options: Optional[StateDictOptions]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers``.\n\n    Patch the ``state_dict`` and ``load_state_dict`` attributes of ``optimizers`` to\n    be a partial function to call ``get_state_dict`` and ``set_state_dict``.\n\n    Note that if there are multiple optimizers, all of the optimizers will be patched.\n    So users only need to call one of the state_dict() to get the full result.\n\n    Example:\n        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n        from torch.distributed.checkpoint.state_dict import patch_model_state_dict\n\n        model = fsdp(model)\n        patch_model_state_dict(model)\n\n    Args:\n        model (nn.Module): the nn.Module to the model.\n        options (StateDictOptions): the options to control how\n            model state_dict and optimizer state_dict should be loaded. See\n            `StateDictOptions` for the details.\n    Returns:\n        None\n    '
    _state_dict_call = functools.partial(get_optimizer_state_dict, model=model, optimizers=optimizers, options=options)

    def state_dict_call():
        if False:
            for i in range(10):
                print('nop')
        return _state_dict_call()
    _load_state_dict_call = functools.partial(set_optimizer_state_dict, model=model, optimizers=optimizers, options=options)

    def load_state_dict_call(state_dict: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        _load_state_dict_call(optim_state_dict=state_dict)
    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call