from typing import cast, Dict, Optional
import torch.nn as nn

class _State:
    pass
_module_state_mapping: Dict[nn.Module, _State] = {}

def _insert_module_state(module: nn.Module, state: _State) -> None:
    if False:
        i = 10
        return i + 15
    global _module_state_mapping
    assert module not in _module_state_mapping, f'Inserting {module} more than once.'
    _module_state_mapping[module] = state

def _get_module_state(module: nn.Module) -> Optional[_State]:
    if False:
        i = 10
        return i + 15
    '\n    Return the ``_State`` in ``model``.\n\n    Given a ``module``, this API finds out if the module is also a ``_State``\n    instance or if the module is managed by a composable API. If the module\n    is also a ``_State``, ``module`` will be casted to ``_State` and returned.\n    If it is managed by a composable API, the corresponding ``_State`` will\n    be returned.\n    '
    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    elif module in _module_state_mapping:
        return _module_state_mapping[module]
    else:
        return None