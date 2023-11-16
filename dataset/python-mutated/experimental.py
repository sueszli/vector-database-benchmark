"""
Utilities for experimental features.

Experimental features have a group, feature name, and optional help string.

When an experimental feature is used, a warning will be displayed. Warnings may be
disabled per feature group with the setting `PREFECT_EXPERIMENTAL_WARN_<GROUP>`.
Warnings may also be disabled globally with the setting `PREFECT_EXPERIMENTAL_WARN`.

Some experimental features require opt-in to enable any usage. These require the setting
`PREFECT_EXPERIMENTAL_ENABLE_<GROUP>` to be set or an error will be thrown on use.
"""
import functools
import warnings
from typing import Any, Callable, Optional, Set, Type, TypeVar
from prefect._internal.pydantic import HAS_PYDANTIC_V2
if HAS_PYDANTIC_V2:
    import pydantic.v1 as pydantic
else:
    import pydantic
from prefect.settings import PREFECT_EXPERIMENTAL_WARN, SETTING_VARIABLES, Setting
from prefect.utilities.callables import get_call_parameters
T = TypeVar('T', bound=Callable)
M = TypeVar('M', bound=pydantic.BaseModel)
EXPERIMENTAL_WARNING = '{feature} is experimental. {help}The interface or behavior may change without warning, we recommend pinning versions to prevent unexpected changes. To disable warnings for this group of experiments, disable PREFECT_EXPERIMENTAL_WARN_{group}.'
EXPERIMENTAL_ERROR = '{feature} is experimental and requires opt-in for usage. {help}To use this feature, enable PREFECT_EXPERIMENTAL_ENABLE_{group}.'

class ExperimentalWarning(Warning):
    """
    A warning related to experimental code.
    """

class ExperimentalError(Exception):
    """
    An exception related to experimental code.
    """

class ExperimentalFeature(ExperimentalWarning):
    """
    A warning displayed on use of an experimental feature.

    These can be globally disabled by the PREFECT_EXPIRIMENTAL_WARN setting.
    """

class ExperimentalFeatureDisabled(ExperimentalError):
    """
    An error displayed on use of a disabled experimental feature that requires opt-in.
    """

def _opt_in_setting_for_group(group: str) -> Setting[bool]:
    if False:
        i = 10
        return i + 15
    group_opt_in_setting_name = f'PREFECT_EXPERIMENTAL_ENABLE_{group.upper()}'
    group_opt_in = SETTING_VARIABLES.get(group_opt_in_setting_name)
    if group_opt_in is None:
        raise ValueError(f'A opt-in setting for experimental feature {group!r} does not exist yet. {group_opt_in_setting_name!r} must be created before the group can be used.')
    return group_opt_in

def _warn_setting_for_group(group: str) -> Setting[bool]:
    if False:
        for i in range(10):
            print('nop')
    group_warn_setting_name = f'PREFECT_EXPERIMENTAL_WARN_{group.upper()}'
    group_warn = SETTING_VARIABLES.get(group_warn_setting_name)
    if group_warn is None:
        raise ValueError(f'A warn setting for experimental feature {group!r} does not exist yet. {group_warn_setting_name!r} must be created before the group can be used.')
    return group_warn

def experimental(feature: str, *, group: str, help: str='', stacklevel: int=2, opt_in: bool=False) -> Callable[[T], T]:
    if False:
        i = 10
        return i + 15
    group = group.upper()
    if help:
        help = help.rstrip() + ' '
    warn_message = EXPERIMENTAL_WARNING.format(feature=feature, group=group, help=help)
    error_message = EXPERIMENTAL_ERROR.format(feature=feature, group=group, help=help)
    if opt_in:
        group_opt_in = _opt_in_setting_for_group(group)
    group_warn = _warn_setting_for_group(group)

    def decorator(fn: T):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if opt_in and (not group_opt_in):
                raise ExperimentalFeatureDisabled(error_message)
            if PREFECT_EXPERIMENTAL_WARN and group_warn:
                warnings.warn(warn_message, ExperimentalFeature, stacklevel=stacklevel)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def experiment_enabled(group: str) -> bool:
    if False:
        print('Hello World!')
    group_opt_in = _opt_in_setting_for_group(group)
    return group_opt_in.value()

def experimental_parameter(name: str, *, group: str, help: str='', stacklevel: int=2, opt_in: bool=False, when: Optional[Callable[[Any], bool]]=None) -> Callable[[T], T]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Mark a parameter in a callable as experimental.\n\n    Example:\n\n        ```python\n\n        @experimental_parameter("y", group="example", when=lambda y: y is not None)\n        def foo(x, y = None):\n            return x + 1 + (y or 0)\n        ```\n    '
    when = when or (lambda _: True)

    @experimental(group=group, feature=f'The parameter {name!r}', help=help, opt_in=opt_in, stacklevel=stacklevel + 2)
    def experimental_check():
        if False:
            i = 10
            return i + 15
        pass

    def decorator(fn: T):
        if False:
            return 10

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                parameters = get_call_parameters(fn, args, kwargs, apply_defaults=False)
            except Exception:
                parameters = kwargs
            if name in parameters and when(parameters[name]):
                experimental_check()
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def experimental_field(name: str, *, group: str, help: str='', stacklevel: int=2, opt_in: bool=False, when: Optional[Callable[[Any], bool]]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Mark a field in a Pydantic model as experimental.\n\n    Raises warning only if the field is specified during init.\n\n    Example:\n\n        ```python\n\n        @experimental_parameter("y", group="example", when=lambda y: y is not None)\n        def foo(x, y = None):\n            return x + 1 + (y or 0)\n        ```\n    '
    when = when or (lambda _: True)

    @experimental(group=group, feature=f'The field {name!r}', help=help, opt_in=opt_in, stacklevel=stacklevel + 2)
    def experimental_check():
        if False:
            i = 10
            return i + 15
        'Utility function for performing a warning check for the specified group'

    def decorator(model_cls: Type[M]) -> Type[M]:
        if False:
            while True:
                i = 10
        cls_init = model_cls.__init__

        @functools.wraps(model_cls.__init__)
        def __init__(__pydantic_self__, **data: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            cls_init(__pydantic_self__, **data)
            if name in data.keys() and when(data[name]):
                experimental_check()
            field = __pydantic_self__.__fields__.get(name)
            if field is not None:
                field.field_info.extra['experimental'] = True
                field.field_info.extra['experimental-group'] = group
        model_cls.__init__ = __init__
        return model_cls
    return decorator

def enabled_experiments() -> Set[str]:
    if False:
        print('Hello World!')
    '\n    Return the set of all enabled experiments.\n    '
    return {name[len('PREFECT_EXPERIMENTAL_ENABLE_'):].lower() for (name, setting) in SETTING_VARIABLES.items() if name.startswith('PREFECT_EXPERIMENTAL_ENABLE_') and setting.value()}