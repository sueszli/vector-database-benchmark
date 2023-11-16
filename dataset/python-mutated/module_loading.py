from __future__ import annotations
import pkgutil
from importlib import import_module
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from types import ModuleType

def import_string(dotted_path: str):
    if False:
        return 10
    '\n    Import a dotted module path and return the attribute/class designated by the last name in the path.\n\n    Raise ImportError if the import failed.\n    '
    try:
        (module_path, class_name) = dotted_path.rsplit('.', 1)
    except ValueError:
        raise ImportError(f"{dotted_path} doesn't look like a module path")
    module = import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f'Module "{module_path}" does not define a "{class_name}" attribute/class')

def qualname(o: object | Callable) -> str:
    if False:
        i = 10
        return i + 15
    'Convert an attribute/class/function to a string importable by ``import_string``.'
    if callable(o) and hasattr(o, '__module__') and hasattr(o, '__name__'):
        return f'{o.__module__}.{o.__name__}'
    cls = o
    if not isinstance(cls, type):
        cls = type(cls)
    name = cls.__qualname__
    module = cls.__module__
    if module and module != '__builtin__':
        return f'{module}.{name}'
    return name

def iter_namespace(ns: ModuleType):
    if False:
        while True:
            i = 10
    return pkgutil.iter_modules(ns.__path__, ns.__name__ + '.')