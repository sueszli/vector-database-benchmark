from __future__ import annotations
import functools
import importlib
import sys
import warnings
from types import ModuleType

def getattr_with_deprecation(imports: dict[str, str], module: str, override_deprecated_classes: dict[str, str], extra_message: str, name: str):
    if False:
        while True:
            i = 10
    '\n    Retrieve the imported attribute from the redirected module and raises a deprecation warning.\n\n    :param imports: dict of imports and their redirection for the module\n    :param module: name of the module in the package to get the attribute from\n    :param override_deprecated_classes: override target classes with deprecated ones. If target class is\n       found in the dictionary, it will be displayed in the warning message.\n    :param extra_message: extra message to display in the warning or import error message\n    :param name: attribute name\n    :return:\n    '
    target_class_full_name = imports.get(name)
    if not target_class_full_name:
        raise AttributeError(f'The module `{module!r}` has no attribute `{name!r}`')
    warning_class_name = target_class_full_name
    if override_deprecated_classes and name in override_deprecated_classes:
        warning_class_name = override_deprecated_classes[name]
    message = f'The `{module}.{name}` class is deprecated. Please use `{warning_class_name!r}`.'
    if extra_message:
        message += f' {extra_message}.'
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    (new_module, new_class_name) = target_class_full_name.rsplit('.', 1)
    try:
        return getattr(importlib.import_module(new_module), new_class_name)
    except ImportError as e:
        error_message = f'Could not import `{new_module}.{new_class_name}` while trying to import `{module}.{name}`.'
        if extra_message:
            error_message += f' {extra_message}.'
        raise ImportError(error_message) from e

def add_deprecated_classes(module_imports: dict[str, dict[str, str]], package: str, override_deprecated_classes: dict[str, dict[str, str]] | None=None, extra_message: str | None=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add deprecated class PEP-563 imports and warnings modules to the package.\n\n    :param module_imports: imports to use\n    :param package: package name\n    :param override_deprecated_classes: override target classes with deprecated ones. If module +\n       target class is found in the dictionary, it will be displayed in the warning message.\n    :param extra_message: extra message to display in the warning or import error message\n    '
    for (module_name, imports) in module_imports.items():
        full_module_name = f'{package}.{module_name}'
        module_type = ModuleType(full_module_name)
        if override_deprecated_classes and module_name in override_deprecated_classes:
            override_deprecated_classes_for_module = override_deprecated_classes[module_name]
        else:
            override_deprecated_classes_for_module = {}
        module_type.__getattr__ = functools.partial(getattr_with_deprecation, imports, full_module_name, override_deprecated_classes_for_module, extra_message or '')
        sys.modules.setdefault(full_module_name, module_type)