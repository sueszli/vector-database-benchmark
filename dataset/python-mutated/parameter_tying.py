"""Utilities for automatic parameters tying.

Reference:
    https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/trainer.py#L110-L118

"""
from typing import Dict, List, Optional
from torch import nn

def find_shared_parameters(module: nn.Module) -> List[str]:
    if False:
        print('Hello World!')
    'Returns a list of names of shared parameters set in the module.'
    return _find_shared_parameters(module)

def _find_shared_parameters(module: nn.Module, tied_parameters: Optional[Dict]=None, prefix: str='') -> List[str]:
    if False:
        return 10
    if tied_parameters is None:
        tied_parameters = {}
    for (name, param) in module._parameters.items():
        param_prefix = prefix + ('.' if prefix else '') + name
        if param is None:
            continue
        if param not in tied_parameters:
            tied_parameters[param] = []
        tied_parameters[param].append(param_prefix)
    for (name, m) in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        _find_shared_parameters(m, tied_parameters, submodule_prefix)
    return [x for x in tied_parameters.values() if len(x) > 1]

def set_shared_parameters(module: nn.Module, shared_params: list) -> nn.Module:
    if False:
        while True:
            i = 10
    for shared_param in shared_params:
        ref = _get_module_by_path(module, shared_param[0])
        for path in shared_param[1:]:
            _set_module_by_path(module, path, ref)
    return module

def _get_module_by_path(module: nn.Module, path: str) -> nn.Module:
    if False:
        for i in range(10):
            print('nop')
    path = path.split('.')
    for name in path:
        module = getattr(module, name)
    return module

def _set_module_by_path(module: nn.Module, path: str, value: nn.Module) -> None:
    if False:
        for i in range(10):
            print('nop')
    path = path.split('.')
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)