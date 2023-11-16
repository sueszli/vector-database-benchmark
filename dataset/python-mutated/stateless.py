import contextlib
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
__all__ = ['functional_call']

def _untie_named_tensors_map(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if False:
        return 10
    "\n    Unties all tied tensors in the module to parameters_and_buffers.\n\n    This function returns a new untied_parameters_and_buffers dictionary and leave the original\n    untied_parameters_and_buffers dictionary unchanged. It adds new (missing) keys for tied tensors\n    in the module to untied_parameters_and_buffers. The value of the new key is the user-given value\n    in the original parameters_and_buffers dictionary.\n\n    If there are more than one user-given values for the same tied tensor, it will raise an error.\n\n    For example, if the module has two tied weights self.foo and self.tied_foo and the user passes\n    {'foo': foo_value, ...}, this will return {'foo': foo_value, 'tied_foo': foo_value, ...}. If the\n    user passes {'foo': foo_value, 'tied_foo': tied_foo_value, ...}, it will raise an error. If the\n    user passes {'foo': foo_value, 'tied_foo': foo_value, ...}, it will not raise an error.\n\n    Args:\n        module (torch.nn.Module): the module to determine which tensors are tied.\n        parameters_and_buffers (Dict[str, Tensor]): a map of {name: tensor} for reparamaterizing the module.\n\n    Returns:\n        A new untied version of the parameters_and_buffers dictionary.\n\n    Raises:\n        ValueError: if there are more than one user-given values for the same tied tensor.\n    "
    all_named_tensors: Dict[str, Tensor] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
    all_named_tensors.update(module.named_buffers(remove_duplicate=False))
    tensor_to_tied_names_map: Dict[Tensor, Set[str]] = defaultdict(set)
    for (name, tensor) in all_named_tensors.items():
        tensor_to_tied_names_map[tensor].add(name)
    tied_names_map: Dict[str, Set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names
    given_names = set(parameters_and_buffers.keys())
    given_names_for_tied_tensors = given_names.intersection(tied_names_map.keys())
    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        if len(tied_names.intersection(given_names_for_tied_tensors)) > 1 and len({parameters_and_buffers[tied_name] for tied_name in tied_names}) != 1:
            raise ValueError(f'functional_call got multiple values for keys {sorted(tied_names)}, which are tied. Consider using tie_weights=False')
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[given_name]
    return untied_parameters_and_buffers

@contextlib.contextmanager
def _reparametrize_module(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor], *, tie_weights: bool=False, strict: bool=False) -> Iterator[None]:
    if False:
        i = 10
        return i + 15
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(module, parameters_and_buffers)
    else:
        untied_parameters_and_buffers = parameters_and_buffers
    accessor = NamedMemberAccessor(module)
    if strict:
        (missing_keys, unexpected_keys) = accessor.check_keys(untied_parameters_and_buffers)
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}.")
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in reparametrizing for {}:\n\t{}'.format(module._get_name(), '\n\t'.join(error_msgs)))
    orig_parameters_and_buffers: Dict[str, Tensor] = {}
    try:
        (orig_parameters_and_buffers, _) = accessor.swap_tensors_dict(untied_parameters_and_buffers, allow_missing=True)
        yield
    finally:
        (new_parameters_and_buffers, _) = accessor.swap_tensors_dict(orig_parameters_and_buffers, allow_missing=True)
        parameters_and_buffers.update({k: new_parameters_and_buffers[k] for k in parameters_and_buffers if k in new_parameters_and_buffers})

def functional_call(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor], args: Union[Any, Tuple], kwargs: Optional[Dict[str, Any]]=None, *, tie_weights: bool=True, strict: bool=False):
    if False:
        print('Hello World!')
    "Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.\n\n    .. warning::\n\n        This API is deprecated as of PyTorch 2.0 and will be removed in a future\n        version of PyTorch. Please use :func:`torch.func.functional_call` instead,\n        which is a drop-in replacement for this API.\n\n    .. note:: If the module has active parametrizations, passing a value in the\n        :attr:`parameters_and_buffers` argument with the name set to the regular parameter\n        name will completely disable the parametrization.\n        If you want to apply the parametrization function to the value passed\n        please set the key as ``{submodule_name}.parametrizations.{parameter_name}.original``.\n\n    .. note:: If the module performs in-place operations on parameters/buffers, these will be reflected\n        in the `parameters_and_buffers` input.\n\n        Example::\n\n            >>> a = {'foo': torch.zeros(())}\n            >>> # xdoctest: +SKIP\n            >>> mod = Foo()  # does self.foo = self.foo + 1\n            >>> print(mod.foo)  # tensor(0.)\n            >>> functional_call(mod, a, torch.ones(()))\n            >>> print(mod.foo)  # tensor(0.)\n            >>> print(a['foo'])  # tensor(1.)\n\n    .. note:: If the module has tied weights, whether or not functional_call respects the tying is determined by the\n        tie_weights flag.\n\n        Example::\n\n            >>> a = {'foo': torch.zeros(())}\n            >>> # xdoctest: +SKIP\n            >>> mod = Foo()  # has both self.foo and self.foo_tied which are tied. Returns x + self.foo + self.foo_tied\n            >>> print(mod.foo)  # tensor(1.)\n            >>> mod(torch.zeros(()))  # tensor(2.)\n            >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) since it will change self.foo_tied too\n            >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied is not updated\n            >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}\n            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)\n\n    Args:\n        module (torch.nn.Module): the module to call\n        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in\n            the module call.\n        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.\n        kwargs (dict): keyword arguments to be passed to the module call\n        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as\n            tied in the reparamaterized version. Therefore, if True and different values are passed for the tied\n            parameters and buffers, it will error. If False, it will not respect the originally tied parameters and\n            buffers unless the values passed for both weights are the same. Default: True.\n        strict (bool, optional): If True, then the parameters and buffers passed in must match the parameters and\n            buffers in the original module. Therefore, if True and there are any missing or unexpected keys, it will\n            error. Default: False.\n\n    Returns:\n        Any: the result of calling ``module``.\n    "
    warnings.warn('This API is deprecated as of PyTorch 2.0 and will be removed in a future version of PyTorch. Please use torch.func.functional_call instead which is a drop-in replacement for this API.')
    return _functional_call(module, parameters_and_buffers, args, kwargs, tie_weights=tie_weights, strict=strict)

def _functional_call(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor], args: Union[Any, Tuple], kwargs: Optional[Dict[str, Any]]=None, *, tie_weights: bool=True, strict: bool=False):
    if False:
        for i in range(10):
            print('nop')
    if torch.jit.is_tracing() or torch.jit.is_scripting() or isinstance(module, (torch.jit.RecursiveScriptModule, torch.jit.ScriptModule, torch.jit.ScriptFunction)):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError("The stateless API can't be used with nn.DataParallel module")
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(module, parameters_and_buffers, tie_weights=tie_weights, strict=strict):
        return module(*args, **kwargs)