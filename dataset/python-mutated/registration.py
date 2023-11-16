"""Module for handling symbolic function registration."""
import warnings
from typing import Callable, Collection, Dict, Generic, Optional, Sequence, Set, TypeVar, Union
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
OpsetVersion = int

def _dispatch_opset_version(target: OpsetVersion, registered_opsets: Collection[OpsetVersion]) -> Optional[OpsetVersion]:
    if False:
        for i in range(10):
            print('nop')
    'Finds the registered opset given a target opset version and the available opsets.\n\n    Args:\n        target: The target opset version.\n        registered_opsets: The available opsets.\n\n    Returns:\n        The registered opset version.\n    '
    if not registered_opsets:
        return None
    descending_registered_versions = sorted(registered_opsets, reverse=True)
    if target >= _constants.ONNX_BASE_OPSET:
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None
    for version in reversed(descending_registered_versions):
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version
    return None
_K = TypeVar('_K')
_V = TypeVar('_V')

class OverrideDict(Generic[_K, _V], Collection[_K]):
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self):
        if False:
            return 10
        self._base: Dict[_K, _V] = {}
        self._overrides: Dict[_K, _V] = {}
        self._merged: Dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        if False:
            return 10
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if a key is in the base dictionary.'
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        if False:
            while True:
                i = 10
        'Overrides a base key-value with a new pair.'
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        if False:
            i = 10
            return i + 15
        'Un-overrides a key-value pair.'
        self._overrides.pop(key, None)
        self._merged.pop(key, None)
        if key in self._base:
            self._merged[key] = self._base[key]

    def overridden(self, key: _K) -> bool:
        if False:
            return 10
        'Checks if a key-value pair is overridden.'
        return key in self._overrides

    def __getitem__(self, key: _K) -> _V:
        if False:
            return 10
        return self._merged[key]

    def get(self, key: _K, default: Optional[_V]=None):
        if False:
            while True:
                i = 10
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        if False:
            i = 10
            return i + 15
        return key in self._merged

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._merged)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self._merged)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'OverrideDict(base={self._base}, overrides={self._overrides})'

    def __bool__(self) -> bool:
        if False:
            print('Hello World!')
        return bool(self._merged)

class _SymbolicFunctionGroup:
    """Different versions of symbolic functions registered to the same name.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._name = name
        self._functions: OverrideDict[OpsetVersion, Callable] = OverrideDict()

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'_SymbolicFunctionGroup({self._name}, registered={self._functions})'

    def __getitem__(self, key: OpsetVersion) -> Callable:
        if False:
            return 10
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, opset: OpsetVersion) -> Optional[Callable]:
        if False:
            return 10
        'Find the most recent version of the function.'
        version = _dispatch_opset_version(opset, self._functions)
        if version is None:
            return None
        return self._functions[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a symbolic function.\n\n        Args:\n            func: The function to add.\n            opset: The opset version of the function to add.\n        '
        if self._functions.in_base(opset):
            warnings.warn(f"Symbolic function '{self._name}' already registered for opset {opset}. Replacing the existing function with new function. This is unexpected. Please report it on {_constants.PYTORCH_GITHUB_ISSUES_URL}.", errors.OnnxExporterWarning)
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds a custom symbolic function.\n\n        Args:\n            func: The symbolic function to register.\n            opset: The corresponding opset version.\n        '
        self._functions.override(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        if False:
            i = 10
            return i + 15
        'Removes a custom symbolic function.\n\n        Args:\n            opset: The opset version of the custom function to remove.\n        '
        if not self._functions.overridden(opset):
            warnings.warn(f"No custom function registered for '{self._name}' opset {opset}")
            return
        self._functions.remove_override(opset)

    def get_min_supported(self) -> OpsetVersion:
        if False:
            for i in range(10):
                print('nop')
        'Returns the lowest built-in opset version supported by the function.'
        return min(self._functions)

class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}

    def register(self, name: str, opset: OpsetVersion, func: Callable, custom: bool=False) -> None:
        if False:
            while True:
                i = 10
        "Registers a symbolic function.\n\n        Args:\n            name: The qualified name of the function to register. In the form of 'domain::op'.\n                E.g. 'aten::add'.\n            opset: The opset version of the function to register.\n            func: The symbolic function to register.\n            custom: Whether the function is a custom function that overrides existing ones.\n\n        Raises:\n            ValueError: If the separator '::' is not in the name.\n        "
        if '::' not in name:
            raise ValueError(f"The name must be in the form of 'domain::op', not '{name}'")
        symbolic_functions = self._registry.setdefault(name, _SymbolicFunctionGroup(name))
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        if False:
            return 10
        'Unregisters a symbolic function.\n\n        Args:\n            name: The qualified name of the function to unregister.\n            opset: The opset version of the function to unregister.\n        '
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        if False:
            i = 10
            return i + 15
        'Returns the function group for the given name.'
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        if False:
            print('Hello World!')
        'Returns whether the given op is registered for the given opset version.'
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> Set[str]:
        if False:
            return 10
        'Returns the set of all registered function names.'
        return set(self._registry)

@_beartype.beartype
def onnx_symbolic(name: str, opset: Union[OpsetVersion, Sequence[OpsetVersion]], decorate: Optional[Sequence[Callable]]=None, custom: bool=False) -> Callable:
    if False:
        return 10
    'Registers a symbolic function.\n\n    Usage::\n\n    ```\n    @onnx_symbolic("aten::symbolic_b", opset=10, decorate=[quantized_aten_handler(scale=1/128, zero_point=0)])\n    @symbolic_helper.parse_args("v", "v", "b")\n    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value:\n        ...\n    ```\n\n    Args:\n        name: The qualified name of the function in the form of \'domain::op\'.\n            E.g. \'aten::add\'.\n        opset: The opset versions of the function to register at.\n        decorate: A sequence of decorators to apply to the function.\n        custom: Whether the function is a custom symbolic function.\n\n    Raises:\n        ValueError: If the separator \'::\' is not in the name.\n    '

    def wrapper(func: Callable) -> Callable:
        if False:
            while True:
                i = 10
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)
        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)
        return func
    return wrapper

@_beartype.beartype
def custom_onnx_symbolic(name: str, opset: Union[OpsetVersion, Sequence[OpsetVersion]], decorate: Optional[Sequence[Callable]]=None) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    "Registers a custom symbolic function.\n\n    Args:\n        name: the qualified name of the function.\n        opset: the opset version of the function.\n        decorate: a sequence of decorators to apply to the function.\n\n    Returns:\n        The decorator.\n\n    Raises:\n        ValueError: If the separator '::' is not in the name.\n    "
    return onnx_symbolic(name, opset, decorate, custom=True)
registry = SymbolicRegistry()