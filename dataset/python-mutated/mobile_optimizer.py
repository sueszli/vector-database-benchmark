"""This module contains utility method for mobile model optimization and lint."""
import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import Optional, Set, List, AnyStr

class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4

def optimize_for_mobile(script_module: torch.jit.ScriptModule, optimization_blocklist: Optional[Set[MobileOptimizerType]]=None, preserved_methods: Optional[List[AnyStr]]=None, backend: str='CPU') -> torch.jit.RecursiveScriptModule:
    if False:
        i = 10
        return i + 15
    "\n    Optimize a torch script module for mobile deployment.\n\n    Args:\n        script_module: An instance of torch script module with type of ScriptModule.\n        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,\n            optimization method will run all the optimizer pass; otherwise, optimizer\n            method will run the optimization pass that is not included inside optimization_blocklist.\n        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked\n        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').\n    Returns:\n        A new optimized torch script module\n    "
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(f'Got {type(script_module)}, but ScriptModule is expected.')
    if optimization_blocklist is None:
        optimization_blocklist = set()
    if preserved_methods is None:
        preserved_methods = []
    preserved_methods_str: List[str] = [str(method) for method in preserved_methods]
    bundled_inputs_attributes = _get_bundled_inputs_preserved_attributes(script_module, preserved_methods_str)
    if all((hasattr(script_module, method) for method in bundled_inputs_attributes)):
        preserved_methods_str = list(set(preserved_methods_str + bundled_inputs_attributes))
    non_exist_methods = []
    for method in preserved_methods_str:
        if not hasattr(script_module, method):
            non_exist_methods.append(method)
    if non_exist_methods:
        raise AttributeError(f"The following methods to preserve do not exist in script_module: {', '.join(non_exist_methods)}")
    backend = backend.lower()
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(script_module._c, optimization_blocklist, preserved_methods_str)
    elif backend == 'vulkan':
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(script_module._c, optimization_blocklist, preserved_methods_str)
    elif backend == 'metal':
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods_str)
    else:
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")
    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)

def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a list of lints for a given torch script module.\n\n    Args:\n        script_module: An instance of torch script module with type of ScriptModule.\n\n    Returns:\n        lint_map: A list of dictionary that contains modules lints\n    '
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(f'Got {type(script_module)}, but ScriptModule is expected.')
    lint_list = []
    if not hasattr(script_module, '_generate_bundled_inputs_for_forward'):
        lint_list.append({'name': LintCode.BUNDLED_INPUT.name, 'message': 'No bundled input for forward, please add bundled inputs before saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs.'})
    for (name, param) in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({'name': LintCode.REQUIRES_GRAD.name, 'message': f'Param {name} requires grad, please set torch.no_grad() to reduce memory usage and improve computation speed during inference phase.'})
    op_names = torch.jit.export_opnames(script_module)
    for op_name in op_names:
        if 'dropout' in op_name:
            lint_list.append({'name': LintCode.DROPOUT.name, 'message': 'Operator {} exists, remember to call eval() before saving the module.and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout operator.'.format(op_name)})
        if 'batch_norm' in op_name:
            lint_list.append({'name': LintCode.BATCHNORM.name, 'message': 'Operator {} exists, remember to call eval() before saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm operator.'.format(op_name)})
    return lint_list

def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: List[str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    bundled_inputs_attributes = []
    if hasattr(script_module, 'get_all_bundled_inputs'):
        bundled_inputs_attributes.append('get_all_bundled_inputs')
        bundled_inputs_attributes.append('get_num_bundled_inputs')
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        bundled_inputs_attributes.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            if function_name not in preserved_methods:
                bundled_inputs_attributes.append(function_name)
            bundled_inputs_attributes.append('get_all_bundled_inputs_for_' + function_name)
            bundled_inputs_attributes.append('_bundled_inputs_deflated_' + function_name)
    return bundled_inputs_attributes