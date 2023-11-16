from typing import Dict, Iterable, List, Tuple
import torch
_MISSING: torch.Tensor = object()

def set_tensor(module: 'torch.nn.Module', name: str, tensor: torch.Tensor) -> None:
    if False:
        i = 10
        return i + 15
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f'{module} is not an instance of torch.nn.Module')
    if not isinstance(tensor, torch.Tensor) and tensor is not None:
        raise TypeError(f'{tensor} is not an instance of torch.Tensor')
    if '.' in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == '':
        raise KeyError('tensor name can\'t be empty string ""')
    if name in module._parameters:
        module._parameters[name] = tensor
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)

def swap_tensor(module: 'torch.nn.Module', name: str, tensor: torch.Tensor, allow_missing: bool=False) -> torch.Tensor:
    if False:
        print('Hello World!')
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f'{module} is not an instance of torch.nn.Module')
    if tensor is not _MISSING and (not isinstance(tensor, torch.Tensor)) and (tensor is not None):
        raise TypeError(f'{tensor} is not an instance of torch.Tensor')
    if '.' in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == '':
        raise KeyError('tensor name can\'t be empty string ""')
    orig_tensor: torch.Tensor
    if name in module._parameters:
        orig_tensor = module._parameters[name]
        if tensor is not _MISSING:
            module._parameters[name] = tensor
        else:
            del module._parameters[name]
    elif name in module._buffers:
        orig_tensor = module._buffers[name]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]
    else:
        try:
            orig_tensor = getattr(module, name)
        except AttributeError as ex:
            if not allow_missing:
                raise AttributeError(f'{module._get_name()} has no attribute `{name}`') from ex
            orig_tensor = _MISSING
        if orig_tensor is not _MISSING and (not isinstance(orig_tensor, torch.Tensor)) and (orig_tensor is not None):
            raise TypeError(f'attribute `{name}`: {orig_tensor} is not an instance of torch.Tensor')
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)
    return orig_tensor

def swap_submodule(module: 'torch.nn.Module', name: str, submodule: 'torch.nn.Module') -> 'torch.nn.Module':
    if False:
        return 10
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f'{module} is not an instance of torch.nn.Module')
    if not isinstance(submodule, torch.nn.Module):
        raise TypeError(f'{submodule} is not an instance of torch.nn.Module')
    if '.' in name:
        raise KeyError('submodule name can\'t contain "."')
    if name == '':
        raise KeyError('submodule name can\'t be empty string ""')
    if name not in module._modules:
        raise KeyError(f'submodule {name} does not exist')
    orig_submodule = module._modules[name]
    if not isinstance(orig_submodule, torch.nn.Module):
        raise TypeError(f'{name} attribute is not an instance of torch.nn.Module')
    module._modules[name] = submodule
    return orig_submodule

class NamedMemberAccessor:
    """
    A class that provides a way to access the submodules and parameters/buffers of a module.

    It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """

    def __init__(self, module: 'torch.nn.Module') -> None:
        if False:
            while True:
                i = 10
        self.module = module
        self.memo: Dict[str, torch.nn.Module] = {}

    def get_submodule(self, name: str) -> 'torch.nn.Module':
        if False:
            i = 10
            return i + 15
        '\n        Return the submodule specified by the given path.\n\n        For example, to get the submodule mod.layer1.conv1,\n        use accessor.get_submodule("layer1.conv1")\n\n        Compare to mod.get_submodule("layer1.conv1"), this method will cache the\n        intermediate submodule access to speed up future lookups.\n        '
        if not name:
            return self.module
        try:
            return self.memo[name]
        except KeyError:
            (prefix, dot, attr) = name.rpartition('.')
            if dot:
                module = self.get_submodule(prefix)
            else:
                module = self.module
            try:
                submodule = getattr(module, attr)
            except AttributeError as ex:
                raise AttributeError(f'{module._get_name()} has no attribute `{attr}`') from ex
            if not isinstance(submodule, torch.nn.Module):
                raise TypeError(f'submodule `{name}`: {submodule} is not an instance of torch.nn.Module')
            self.memo[name] = submodule
            return submodule

    def swap_submodule(self, path: str, value: 'torch.nn.Module') -> 'torch.nn.Module':
        if False:
            i = 10
            return i + 15
        '\n        Swap the submodule specified by the given ``path`` to ``value``.\n\n        For example, to swap the attribute mod.layer1.conv1 use\n        ``accessor.swap_submodule("layer1.conv1", conv2)``.\n        '
        (prefix, _, attr) = path.rpartition('.')
        return swap_submodule(self.get_submodule(prefix), attr, value)

    def get_tensor(self, name: str) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Get the tensor specified by the given path to value.\n\n        For example, to get the attribute mod.layer1.conv1.weight,\n        use accessor.get_tensor(\'layer1.conv1.weight\')\n\n        Compare to mod.get_parameter("layer1.conv1.weight"), this method will\n        cache the intermediate submodule access to speed up future lookups.\n        '
        (prefix, _, attr) = name.rpartition('.')
        submodule = self.get_submodule(prefix)
        try:
            tensor = getattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(f'{submodule._get_name()} has no attribute `{name}`') from ex
        if not isinstance(tensor, torch.Tensor) and tensor is not None:
            raise TypeError(f'{tensor} is not an instance of torch.Tensor')
        return tensor

    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the attribute specified by the given path to value.\n\n        For example, to set the attribute mod.layer1.conv1.weight,\n        use accessor.set_tensor("layer1.conv1.weight", value)\n        '
        (prefix, _, attr) = name.rpartition('.')
        set_tensor(self.get_submodule(prefix), attr, value)

    def del_tensor(self, name: str) -> None:
        if False:
            print('Hello World!')
        '\n        Delete the attribute specified by the given path.\n\n        For example, to delete the attribute mod.layer1.conv1.weight,\n        use accessor.del_tensor("layer1.conv1.weight")\n        '
        (prefix, _, attr) = name.rpartition('.')
        submodule = self.get_submodule(prefix)
        try:
            delattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(f'{submodule._get_name()} has no attribute `{name}`') from ex

    def swap_tensor(self, name: str, value: torch.Tensor, allow_missing: bool=False) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Swap the attribute specified by the given path to value.\n\n        For example, to swap the attribute mod.layer1.conv1.weight,\n        use accessor.swap_tensor("layer1.conv1.weight", value)\n        '
        (prefix, _, attr) = name.rpartition('.')
        return swap_tensor(self.get_submodule(prefix), attr, value, allow_missing=allow_missing)

    def get_tensors(self, names: Iterable[str]) -> List[torch.Tensor]:
        if False:
            return 10
        '\n        Get the tensors specified by the given paths.\n\n        For example, to get the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.get_tensors(["layer1.conv1.weight",\n        "layer1.conv1.bias"])\n        '
        return [self.get_tensor(name) for name in names]

    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the attributes specified by the given paths to values.\n\n        For example, to set the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",\n        "layer1.conv1.bias"], [weight, bias])\n        '
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        assert len(names) == len(values), 'names and values must have the same length'
        for (name, value) in zip(names, values):
            self.set_tensor(name, value)

    def set_tensors_dict(self, named_tensors: Dict[str, torch.Tensor]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the attributes specified by the given paths to values.\n\n        For example, to set the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.set_tensors_dict({\n            "layer1.conv1.weight": weight,\n            "layer1.conv1.bias": bias,\n        })\n        '
        for (name, value) in named_tensors.items():
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        if False:
            print('Hello World!')
        '\n        Delete the attributes specified by the given paths.\n\n        For example, to delete the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",\n        "layer1.conv1.bias"])\n        '
        for name in names:
            self.del_tensor(name)

    def swap_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor], allow_missing: bool=False) -> List[torch.Tensor]:
        if False:
            return 10
        '\n        Swap the attributes specified by the given paths to values.\n\n        For example, to swap the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",\n        "layer1.conv1.bias"], [weight, bias])\n        '
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        assert len(names) == len(values), 'names and values must have the same length'
        return [self.swap_tensor(name, value, allow_missing=allow_missing) for (name, value) in zip(names, values)]

    def swap_tensors_dict(self, named_tensors: Dict[str, torch.Tensor], allow_missing: bool=False) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Swap the attributes specified by the given paths to values.\n\n        For example, to swap the attributes mod.layer1.conv1.weight and\n        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({\n            "layer1.conv1.weight": weight,\n            "layer1.conv1.bias": bias,\n        })\n        '
        orig_named_tensors = {}
        missing_keys = []
        try:
            for (name, tensor) in named_tensors.items():
                orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
                if orig_tensor is _MISSING:
                    missing_keys.append(name)
                orig_named_tensors[name] = orig_tensor
        except Exception:
            for (name, orig_tensor) in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise
        if missing_keys and (not allow_missing):
            for (name, orig_tensor) in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise RuntimeError(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        return (orig_named_tensors, missing_keys)

    def check_keys(self, keys: Iterable[str]) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        'Check that the given keys are valid.'
        keys = set(keys)
        valid_keys = {name for (name, _) in self.named_tensors(remove_duplicate=False)}
        missing_keys = valid_keys - keys
        unexpected_keys = keys - valid_keys
        return (sorted(missing_keys), sorted(unexpected_keys))

    def named_parameters(self, remove_duplicate: bool=True) -> Iterable[Tuple[str, torch.Tensor]]:
        if False:
            return 10
        'Iterate over all the parameters in the module.'
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(self, remove_duplicate: bool=True) -> Iterable[Tuple[str, torch.Tensor]]:
        if False:
            while True:
                i = 10
        'Iterate over all the buffers in the module.'
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(self, remove_duplicate: bool=True) -> Iterable[Tuple[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        'Iterate over all the tensors in the module.'
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(self, remove_duplicate: bool=True) -> Iterable[Tuple[str, 'torch.nn.Module']]:
        if False:
            while True:
                i = 10
        'Iterate over all the modules in the module.'
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)