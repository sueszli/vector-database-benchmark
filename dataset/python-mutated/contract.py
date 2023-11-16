import uuid
from collections import OrderedDict
from functools import wraps
from typing import Callable, Dict, List, Optional, Type
import torch.nn as nn
from torch.distributed._composable_state import _State

def generate_state_key(string='__composable_api_state_key'):
    if False:
        while True:
            i = 10
    return f'{string}_{str(uuid.uuid4())}'
STATE_KEY = generate_state_key()
REGISTRY_KEY = generate_state_key()

class RegistryItem:
    pass

def contract(state_cls: Type[_State]=_State):
    if False:
        while True:
            i = 10
    '\n    Decorate a function as a composable distributed API, where the first\n    argument of the function must be an :class:`nn.Module` instance. The\n    decorator verifies that the wrapped function does not modify parameter,\n    buffer or sub-module fully-qualified names (FQN).\n\n    When a function ``func`` is decorated by ``@contract()``, a\n    ``.state(module: nn.Module)`` method will be installed to the decorated\n    function. Then you can retrieve and modify the state on a module by calling\n    ``func.state(module)``.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> import torch.nn as nn\n        >>>\n        >>> class MyModel(nn.Module):\n        >>>     def __init__(self):\n        >>>         super().__init__()\n        >>>         self.l1 = nn.Linear(10, 10)\n        >>>         self.l2 = nn.Linear(10, 10)\n        >>>\n        >>>     def forward(self, x):\n        >>>         return self.l2(self.l1(x))\n        >>>\n        >>> @contract()\n        >>> def my_feature(module: nn.Module) -> nn.Module:\n        >>>     my_feature.state(module).some_state = "any value"\n        >>>     return module\n        >>>\n        >>> model = MyModel()\n        >>> my_feature(model.l1)\n        >>> assert my_feature.state(model.l1).some_state == "any value"\n        >>> my_feature(model.l2)\n        >>> model(torch.randn(2, 10)).sum().backward()\n    '

    @wraps(state_cls)
    def inner(func):
        if False:
            return 10

        @wraps(func)
        def wrapper(module: nn.Module, *args, **kwargs) -> Optional[nn.Module]:
            if False:
                while True:
                    i = 10
            default_all_state: Dict[Callable, _State] = OrderedDict()
            all_state: Dict[Callable, _State] = module.__dict__.setdefault(STATE_KEY, default_all_state)
            assert isinstance(all_state, dict), 'Distributed composable API states corrupted'
            default_registry: Dict[str, RegistryItem] = OrderedDict()
            registry: Dict[str, RegistryItem] = module.__dict__.setdefault(REGISTRY_KEY, default_registry)
            assert isinstance(registry, dict), 'Distributed composable API registry corrupted'
            assert func not in all_state and func.__name__ not in registry, f'Each distinct composable distributed API can only be applied to a module once. {func.__name__} has already been applied to the following module.\n{module}'
            all_state.setdefault(func, state_cls())
            registry.setdefault(func.__name__, RegistryItem())
            orig_named_params = OrderedDict(module.named_parameters())
            orig_named_buffers = OrderedDict(module.named_buffers(remove_duplicate=False))
            orig_named_modules = OrderedDict(module.named_modules(remove_duplicate=False))
            updated = func(module, *args, **kwargs)
            if updated is None:
                updated = module
            new_named_params = OrderedDict(updated.named_parameters())
            new_named_buffers = OrderedDict(updated.named_buffers(remove_duplicate=False))
            new_named_modules = OrderedDict(updated.named_modules(remove_duplicate=False))
            assert isinstance(updated, nn.Module), f'Output of composable distributed APIs must be either None or nn.Module, but got {type(updated)}'

            def check_fqn(orig_fqns: List[str], new_fqns: List[str], check_key: str):
                if False:
                    return 10
                if orig_fqns == new_fqns:
                    return
                (orig_fqn_set, new_fqn_set) = (set(orig_fqns), set(new_fqns))
                orig_only = orig_fqn_set - new_fqn_set
                new_only = new_fqn_set - orig_fqn_set
                if len(orig_only) or len(new_only):
                    raise RuntimeError(f'{check_key}Composable distributed API implementations cannot modify FQNs.\nOnly in original FQNs: {orig_only},\nOnly in new FQNs: {new_only}')
                else:
                    raise RuntimeError(f'{check_key}Composable distributed API implementations cannot modify the order of FQNs.\nOriginal FQNs: {orig_only}\nNew FQNs: {new_only}')
            check_fqn(list(orig_named_params.keys()), list(new_named_params.keys()), 'Check parameters, ')
            check_fqn(list(orig_named_buffers.keys()), list(new_named_buffers.keys()), 'Check buffer, ')
            check_fqn(list(orig_named_modules.keys()), list(new_named_modules.keys()), 'Check modules, ')
            return updated

        def get_state(module: nn.Module) -> Optional[_State]:
            if False:
                while True:
                    i = 10
            return module.__dict__.setdefault(STATE_KEY, {}).get(func)
        wrapper.state = get_state
        return wrapper
    return inner

def _get_registry(module: nn.Module) -> Optional[Dict[str, RegistryItem]]:
    if False:
        while True:
            i = 10
    '\n    Get an ``OrderedDict`` of composable APIs that have been applied to the\n    ``module``, indexed by the API name. If no API has been applied, then this\n    returns ``None``.\n    '
    return getattr(module, REGISTRY_KEY, None)