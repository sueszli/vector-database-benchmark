"""
NOTE: This file must be imported like
``import torch.distributed.fsdp._traversal_utils`` and not like
``from torch.distirbuted.fsdp._traversal_utils import ...`` to avoid circular
imports. For brevity, we may import the file as ``traversal_utils``.
"""
import collections
from typing import Deque, List, Set, Tuple
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state
'\n[Note: FSDP State Traversal]\nFor the wrapper code path, ``_FSDPState`` is the ``FullyShardedDataParallel``\nmodule wrapping a fully sharded module, and for the non-wrapper code path,\n``_FSDPState`` is an object that gets embedded on a fully sharded module.\nSee [Note: Fully Sharded Module] for the definition.\n\nThere are three common traversal idioms: Given a root module,\n- ``_get_fsdp_states()`` returns all ``_FSDPState`` s in the tree.\n- ``get_fsdp_root_states()`` returns all local root ``_FSDPState`` s in the\ntree (i.e. those with ``_is_root == True``).\n- ``_get_fsdp_handles()``returns all ``FlatParamHandle`` s in the tree.\n\nAll of these methods must take in the root module (i.e. an ``nn.Module``) and\nnot a general ``_FSDPState`` because ``_FSDPState`` does not support a graph\ntraversal, whereas ``nn.Module`` has ``nn.Module.modules()`` for traversal.\n'

def _composable(module: nn.Module) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Returns if ``module`` can compose with ``fully_shard``.\n    '
    registry = _get_registry(module)
    if registry is None:
        return True
    return 'replicate' not in registry

def _get_fsdp_states_with_modules(module: nn.Module) -> Tuple[List[_FSDPState], List[nn.Module]]:
    if False:
        while True:
            i = 10
    '\n    Returns a tuple containing:\n    1. A list of the ``_FSDPState`` instances in the module tree rooted at\n    ``module`` without any duplicates and following the ``module.modules()``\n    traversal order (which is assumed to be depth-first).\n    2. A corresponding list of the modules owning the states in the first list.\n\n    For the wrapper code path, both returned lists are the same, each\n    containing all ``FullyShardedDataParallel`` instances. For the composable\n    code path, this returns a list of all composable state instances and a list\n    of the corresponding fully sharded modules. See [Note: Fully Sharded\n    Module].\n\n    NOTE: The traversal does not proceed into any module annotated by an\n    incompatible API (e.g. ``replicate``).\n    '
    fsdp_states: List[_FSDPState] = []
    fsdp_modules: List[nn.Module] = []
    visited_fsdp_states: Set[_FSDPState] = set()
    visited_modules: Set[nn.Module] = set()
    deque: Deque[nn.Module] = collections.deque([module])
    while deque:
        submodule = deque.popleft()
        visited_modules.add(submodule)
        if not _composable(submodule):
            continue
        for child_module in reversed(list(submodule.children())):
            if child_module not in visited_modules:
                deque.appendleft(child_module)
        optional_state = _get_module_fsdp_state(submodule)
        if optional_state is not None and optional_state not in visited_fsdp_states:
            visited_fsdp_states.add(optional_state)
            fsdp_states.append(optional_state)
            fsdp_modules.append(submodule)
    return (fsdp_states, fsdp_modules)

def _get_fsdp_states(module: nn.Module) -> List[_FSDPState]:
    if False:
        print('Hello World!')
    'See :func:`_get_fsdp_states_with_modules`.'
    (fsdp_states, _) = _get_fsdp_states_with_modules(module)
    return fsdp_states

def _get_fsdp_handles(module: nn.Module) -> List:
    if False:
        i = 10
        return i + 15
    '\n    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``\n    following the rules in :func:`_get_fsdp_state`.\n    '
    handles = [fsdp_state._handle for fsdp_state in _get_fsdp_states(module) if fsdp_state._handle is not None]
    return handles