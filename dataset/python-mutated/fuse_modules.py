import copy
import torch.nn as nn
from torch.ao.quantization.fuser_method_mappings import get_fuser_method
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn_relu
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import List, Optional
__all__ = ['fuse_known_modules', 'fuse_modules', 'fuse_modules_qat']

def _get_module(model, submodule_key):
    if False:
        return 10
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def _set_module(model, submodule_key, module):
    if False:
        while True:
            i = 10
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    if False:
        return 10
    'Return a list of known fuse modules.\n\n    Returns a list of modules that fuses the operations specified\n     in the input module list.\n\n    Fuses only the following sequence of modules:\n    conv, bn\n    conv, bn, relu\n    conv, relu\n    linear, bn\n    linear, relu\n    For these sequences, the first element in the output module list performs\n    the fused operation. The rest of the elements are set to nn.Identity()\n    '
    types = tuple((type_before_parametrizations(m) for m in mod_list))
    fuser_method = get_fuser_method(types, additional_fuser_method_mapping)
    if fuser_method is None:
        raise NotImplementedError(f'Cannot fuse modules: {types}')
    new_mod: List[Optional[nn.Module]] = [None] * len(mod_list)
    fused = fuser_method(is_qat, *mod_list)
    for pre_hook_fn in mod_list[0]._forward_pre_hooks.values():
        fused.register_forward_pre_hook(pre_hook_fn)
    mod_list[0]._forward_pre_hooks.clear()
    for hook_fn in mod_list[-1]._forward_hooks.values():
        fused.register_forward_hook(hook_fn)
    mod_list[-1]._forward_hooks.clear()
    new_mod[0] = fused
    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = mod_list[0].training
        new_mod[i] = identity
    return new_mod

def _fuse_modules_helper(model, modules_to_fuse, is_qat, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if False:
        i = 10
        return i + 15
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get('additional_fuser_method_mapping', {})
    mod_list = []
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)
    for (i, item) in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])

def _fuse_modules(model, modules_to_fuse, is_qat, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if False:
        print('Hello World!')
    if not inplace:
        model = copy.deepcopy(model)
    if all((isinstance(module_element, str) for module_element in modules_to_fuse)):
        _fuse_modules_helper(model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict)
    else:
        for module_list in modules_to_fuse:
            _fuse_modules_helper(model, module_list, is_qat, fuser_func, fuse_custom_config_dict)
    return model

def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if False:
        while True:
            i = 10
    'Fuse a list of modules into a single module.\n\n    Fuses only the following sequence of modules:\n    conv, bn\n    conv, bn, relu\n    conv, relu\n    linear, relu\n    bn, relu\n    All other sequences are left unchanged.\n    For these sequences, replaces the first item in the list\n    with the fused module, replacing the rest of the modules\n    with identity.\n\n    Args:\n        model: Model containing the modules to be fused\n        modules_to_fuse: list of list of module names to fuse. Can also be a list\n                         of strings if there is only a single list of modules to fuse.\n        inplace: bool specifying if fusion happens in place on the model, by default\n                 a new model is returned\n        fuser_func: Function that takes in a list of modules and outputs a list of fused modules\n                    of the same length. For example,\n                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]\n                    Defaults to torch.ao.quantization.fuse_known_modules\n        `fuse_custom_config_dict`: custom configuration for fusion\n\n    .. code-block:: python\n\n       # Example of fuse_custom_config_dict\n       fuse_custom_config_dict = {\n           # Additional fuser_method mapping\n           "additional_fuser_method_mapping": {\n               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn\n           },\n       }\n\n    Returns:\n        model with fused modules. A new copy is created if inplace=True.\n\n    Examples::\n\n            >>> # xdoctest: +SKIP\n            >>> m = M().eval()\n            >>> # m is a module containing the sub-modules below\n            >>> modules_to_fuse = [ [\'conv1\', \'bn1\', \'relu1\'], [\'submodule.conv\', \'submodule.relu\']]\n            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)\n            >>> output = fused_m(input)\n\n            >>> m = M().eval()\n            >>> # Alternately provide a single list of modules to fuse\n            >>> modules_to_fuse = [\'conv1\', \'bn1\', \'relu1\']\n            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)\n            >>> output = fused_m(input)\n\n    '
    return _fuse_modules(model, modules_to_fuse, is_qat=False, inplace=inplace, fuser_func=fuser_func, fuse_custom_config_dict=fuse_custom_config_dict)

def fuse_modules_qat(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if False:
        for i in range(10):
            print('nop')
    'QAT version for `fuse_modules`.'
    return _fuse_modules(model, modules_to_fuse, is_qat=True, inplace=inplace, fuser_func=fuser_func, fuse_custom_config_dict=fuse_custom_config_dict)