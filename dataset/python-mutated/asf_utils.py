import importlib
import importlib.util
import inspect
import pkgutil
import re
from types import FunctionType, ModuleType
from typing import Optional, Pattern

def _import_submodules(package_name: str, module_regex: Optional[Pattern]=None, recursive: bool=True) -> dict[str, ModuleType]:
    if False:
        i = 10
        return i + 15
    '\n    Imports all submodules of the given package with the defined (optional) module_suffix.\n\n    :param package_name: To start the loading / importing at\n    :param module_regex: Optional regex to filter the module names for\n    :param recursive: True if the package should be loaded recursively\n    :return:\n    '
    package = importlib.import_module(package_name)
    results = {}
    for (loader, name, is_pkg) in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if not module_regex or module_regex.match(name):
            results[name] = importlib.import_module(name)
        if recursive and is_pkg:
            results.update(_import_submodules(name, module_regex, recursive))
    return results

def _collect_provider_classes(provider_module: str, provider_module_regex: Pattern, provider_class_regex: Pattern) -> list[type]:
    if False:
        return 10
    '\n    Collects all provider implementation classes which should be tested.\n    :param provider_module: module to start collecting in\n    :param provider_module_regex: Regex to filter the module names for\n    :param provider_class_regex: Regex to filter the provider class names for\n    :return: list of classes to check the operation signatures of\n    '
    provider_classes = []
    provider_modules = _import_submodules(provider_module, provider_module_regex)
    for (_, mod) in provider_modules.items():
        classes = [cls_obj for (cls_name, cls_obj) in inspect.getmembers(mod) if inspect.isclass(cls_obj) and provider_class_regex.match(cls_name)]
        provider_classes.extend(classes)
    return provider_classes

def collect_implemented_provider_operations(provider_module: str='localstack.services', provider_module_regex: Pattern=re.compile('.*\\.provider[A-Za-z_0-9]*$'), provider_class_regex: Pattern=re.compile('.*Provider$'), asf_api_module: str='localstack.aws.api') -> list[tuple[type, type, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Collects all implemented operations on all provider classes together with their base classes (generated API classes).\n    :param provider_module: module to start collecting in\n    :param provider_module_regex: Regex to filter the module names for\n    :param provider_class_regex: Regex to filter the provider class names for\n    :param asf_api_module: module which contains the generated ASF APIs\n    :return: list of tuple, where each tuple is (provider_class: type, base_class: type, provider_function: str)\n    '
    results = []
    provider_classes = _collect_provider_classes(provider_module, provider_module_regex, provider_class_regex)
    for provider_class in provider_classes:
        for base_class in provider_class.__bases__:
            base_parent_module = '.'.join(base_class.__module__.split('.')[:-1])
            if base_parent_module == asf_api_module:
                provider_functions = [method for method in dir(provider_class) if hasattr(base_class, method) and isinstance(getattr(base_class, method), FunctionType) and (method.startswith('__') is False)]
                for provider_function in provider_functions:
                    results.append((provider_class, base_class, provider_function))
    return results

def check_provider_signature(sub_class: type, base_class: type, method_name: str) -> None:
    if False:
        while True:
            i = 10
    "\n    Checks if the signature of a given provider method is equal to the signature of the function with the same name on the base class.\n\n    :param sub_class: provider class to check the given method's signature of\n    :param base_class: API class to check the given method's signature against\n    :param method_name: name of the method on the sub_class and base_class to compare\n    :raise: AssertionError if the two signatures are not equal\n    "
    try:
        sub_function = getattr(sub_class, method_name)
    except AttributeError:
        raise AttributeError(f"Given method name ('{method_name}') is not a method of the sub class ('{sub_class.__name__}').")
    if not isinstance(sub_function, FunctionType):
        raise AttributeError(f"Given method name ('{method_name}') is not a method of the sub class ('{sub_class.__name__}').")
    if not getattr(sub_function, 'expand_parameters', True):
        return
    if (wrapped := getattr(sub_function, '__wrapped__', False)):
        sub_function = wrapped
    try:
        base_function = getattr(base_class, method_name)
        base_function = base_function.__wrapped__
        sub_spec = inspect.getfullargspec(sub_function)
        base_spec = inspect.getfullargspec(base_function)
        assert sub_spec == base_spec, f'{sub_class.__name__}#{method_name} breaks with {base_class.__name__}#{method_name}'
    except AttributeError:
        pass