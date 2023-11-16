import inspect
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, TypeVar, Union, overload
from typing_extensions import Annotated, Final, TypeAlias
from dagster import _check as check
from dagster._core.decorator_utils import Decoratable, apply_pre_call_decorator, get_decorator_target, is_resource_def
from dagster._utils.warnings import deprecation_warning, experimental_warning
Annotatable: TypeAlias = Decoratable
T_Annotatable = TypeVar('T_Annotatable', bound=Annotatable)
_PUBLIC_ATTR_NAME: Final[str] = '_is_public'

def public(obj: T_Annotatable) -> T_Annotatable:
    if False:
        i = 10
        return i + 15
    'Mark a method on a public class as public. This distinguishes the method from "internal"\n    methods, which are methods that are public in the Python sense of being non-underscored, but\n    not intended for user access. Only `public` methods of a class are rendered in the docs.\n    '
    target = _get_annotation_target(obj)
    setattr(target, _PUBLIC_ATTR_NAME, True)
    return obj

def is_public(obj: Annotatable) -> bool:
    if False:
        while True:
            i = 10
    target = _get_annotation_target(obj)
    return hasattr(target, _PUBLIC_ATTR_NAME) and getattr(target, _PUBLIC_ATTR_NAME)
T = TypeVar('T')
PUBLIC: Final[str] = 'public'
PublicAttr: TypeAlias = Annotated[T, PUBLIC]
_DEPRECATED_ATTR_NAME: Final[str] = '_deprecated'

@dataclass
class DeprecatedInfo:
    breaking_version: str
    additional_warn_text: Optional[str] = None
    subject: Optional[str] = None

@overload
def deprecated(__obj: T_Annotatable, *, breaking_version: str, additional_warn_text: Optional[str]=..., subject: Optional[str]=..., emit_runtime_warning: bool=...) -> T_Annotatable:
    if False:
        i = 10
        return i + 15
    ...

@overload
def deprecated(__obj: None=..., *, breaking_version: str, additional_warn_text: Optional[str]=..., subject: Optional[str]=..., emit_runtime_warning: bool=...) -> Callable[[T_Annotatable], T_Annotatable]:
    if False:
        print('Hello World!')
    ...

def deprecated(__obj: Optional[T_Annotatable]=None, *, breaking_version: str, additional_warn_text: Optional[str]=None, subject: Optional[str]=None, emit_runtime_warning: bool=True) -> Union[T_Annotatable, Callable[[T_Annotatable], T_Annotatable]]:
    if False:
        print('Hello World!')
    'Mark an object as deprecated. This appends some metadata to the object that causes it to be\n    rendered with a "deprecated" tag and associated warning in the docs.\n\n    If `emit_runtime_warning` is True, a warning will also be emitted when the function is called,\n    having the same text as is displayed in the docs. For consistency between docs and runtime\n    warnings, this decorator is preferred to manual calls to `deprecation_warning`.\n\n    Args:\n        breaking_version (str): The version at which the deprecated function will be removed.\n        additional_warn_text (Optional[str]): Additional text to display after the deprecation warning.\n            Typically this should suggest a newer API.\n        subject (Optional[str]): The subject of the deprecation warning. Defaults to a string\n            representation of the decorated object. This is useful when marking usage of\n            a deprecated API inside an otherwise non-deprecated function, so\n            that it can be easily cleaned up later. It should only be used with\n            `emit_runtime_warning=False`, as we don\'t want to warn users when a\n            deprecated API is used internally.\n        emit_runtime_warning (bool): Whether to emit a warning when the function is called.\n\n    Usage:\n\n        .. code-block:: python\n\n            @deprecated(breaking_version="2.0", additional_warn_text="Use my_new_function instead")\n            def my_deprecated_function(my_arg):\n                ...\n\n            @deprecated(breaking_version="2.0", additional_warn_text="Use MyNewClass instead")\n            class MyDeprecatedClass:\n                ...\n\n            @deprecated(breaking_version="2.0", subject="some_deprecated_function", emit_runtime_warning=False)\n            def not_deprecated_function():\n                ...\n                some_deprecated_function()\n                ...\n    '
    if __obj is None:
        return lambda obj: deprecated(obj, subject=subject, emit_runtime_warning=emit_runtime_warning, breaking_version=breaking_version, additional_warn_text=additional_warn_text)
    else:
        target = _get_annotation_target(__obj)
        setattr(target, _DEPRECATED_ATTR_NAME, DeprecatedInfo(breaking_version, additional_warn_text, subject))
        if emit_runtime_warning:
            warning_fn = lambda : deprecation_warning(subject or _get_subject(__obj), breaking_version=breaking_version, additional_warn_text=additional_warn_text, stacklevel=_get_warning_stacklevel(__obj))
            return apply_pre_call_decorator(__obj, warning_fn)
        else:
            return __obj

def is_deprecated(obj: Annotatable) -> bool:
    if False:
        print('Hello World!')
    target = _get_annotation_target(obj)
    return hasattr(target, _DEPRECATED_ATTR_NAME)

def get_deprecated_info(obj: Annotatable) -> DeprecatedInfo:
    if False:
        i = 10
        return i + 15
    target = _get_annotation_target(obj)
    return getattr(target, _DEPRECATED_ATTR_NAME)
_DEPRECATED_PARAM_ATTR_NAME: Final[str] = '_deprecated_params'

@overload
def deprecated_param(__obj: T_Annotatable, *, param: str, breaking_version: str, additional_warn_text: Optional[str]=..., emit_runtime_warning: bool=...) -> T_Annotatable:
    if False:
        return 10
    ...

@overload
def deprecated_param(__obj: None=..., *, param: str, breaking_version: str, additional_warn_text: Optional[str]=..., emit_runtime_warning: bool=...) -> Callable[[T_Annotatable], T_Annotatable]:
    if False:
        print('Hello World!')
    ...

def deprecated_param(__obj: Optional[T_Annotatable]=None, *, param: str, breaking_version: str, additional_warn_text: Optional[str]=None, emit_runtime_warning: bool=True) -> T_Annotatable:
    if False:
        print('Hello World!')
    'Mark a parameter of a class initializer or function/method as deprecated. This appends some\n    metadata to the decorated object that causes the specified argument to be rendered with a\n    "deprecated" tag and associated warning in the docs.\n\n    If `emit_runtime_warning` is True, a warning will also be emitted when the function is called\n    and a non-None value is passed for the parameter. For consistency between docs and runtime\n    warnings, this decorator is preferred to manual calls to `deprecation_warning`. Note that the\n    warning will only be emitted if the value is passed as a keyword argument.\n\n    Args:\n        param (str): The name of the parameter to deprecate.\n        breaking_version (str): The version at which the deprecated function will be removed.\n        additional_warn_text (str): Additional text to display after the deprecation warning.\n            Typically this should suggest a newer API.\n        emit_runtime_warning (bool): Whether to emit a warning when the function is called.\n    '
    if __obj is None:
        return lambda obj: deprecated_param(obj, param=param, breaking_version=breaking_version, additional_warn_text=additional_warn_text, emit_runtime_warning=emit_runtime_warning)
    else:
        check.invariant(_annotatable_has_param(__obj, param), f'Attempted to mark undefined parameter `{param}` deprecated.')
        target = _get_annotation_target(__obj)
        if not hasattr(target, _DEPRECATED_PARAM_ATTR_NAME):
            setattr(target, _DEPRECATED_PARAM_ATTR_NAME, {})
        getattr(target, _DEPRECATED_PARAM_ATTR_NAME)[param] = DeprecatedInfo(breaking_version=breaking_version, additional_warn_text=additional_warn_text)
        if emit_runtime_warning:
            condition = lambda *_, **kwargs: kwargs.get(param) is not None
            warning_fn = lambda : deprecation_warning(_get_subject(__obj, param=param), breaking_version=breaking_version, additional_warn_text=additional_warn_text, stacklevel=4)
            return apply_pre_call_decorator(__obj, warning_fn, condition=condition)
        else:
            return __obj

def has_deprecated_params(obj: Annotatable) -> bool:
    if False:
        while True:
            i = 10
    return hasattr(_get_annotation_target(obj), _DEPRECATED_PARAM_ATTR_NAME)

def get_deprecated_params(obj: Annotatable) -> Mapping[str, DeprecatedInfo]:
    if False:
        return 10
    return getattr(_get_annotation_target(obj), _DEPRECATED_PARAM_ATTR_NAME)

def is_deprecated_param(obj: Annotatable, param_name: str) -> bool:
    if False:
        i = 10
        return i + 15
    target = _get_annotation_target(obj)
    return param_name in getattr(target, _DEPRECATED_PARAM_ATTR_NAME, {})

def get_deprecated_param_info(obj: Annotatable, param_name: str) -> DeprecatedInfo:
    if False:
        while True:
            i = 10
    target = _get_annotation_target(obj)
    return getattr(target, _DEPRECATED_PARAM_ATTR_NAME)[param_name]
_EXPERIMENTAL_ATTR_NAME: Final[str] = '_experimental'

@dataclass
class ExperimentalInfo:
    additional_warn_text: Optional[str] = None
    subject: Optional[str] = None

@overload
def experimental(__obj: T_Annotatable, *, additional_warn_text: Optional[str]=..., subject: Optional[str]=..., emit_runtime_warning: bool=...) -> T_Annotatable:
    if False:
        return 10
    ...

@overload
def experimental(__obj: None=..., *, additional_warn_text: Optional[str]=..., subject: Optional[str]=..., emit_runtime_warning: bool=...) -> Callable[[T_Annotatable], T_Annotatable]:
    if False:
        return 10
    ...

def experimental(__obj: Optional[T_Annotatable]=None, *, additional_warn_text: Optional[str]=None, subject: Optional[str]=None, emit_runtime_warning: bool=True) -> Union[T_Annotatable, Callable[[T_Annotatable], T_Annotatable]]:
    if False:
        i = 10
        return i + 15
    'Mark an object as experimental. This appends some metadata to the object that causes it\n    to be rendered with an "experimental" tag and associated warning in the docs.\n\n    If `emit_runtime_warning` is True, a warning will also be emitted when the function is called,\n    having the same text as is displayed in the docs. For consistency between docs and runtime\n    warnings, this decorator is preferred to manual calls to `experimental_warning`.\n\n    Args:\n        additional_warn_text (str): Additional text to display after the experimental warning.\n        emit_runtime_warning (bool): Whether to emit a warning when the function is called.\n        subject (Optional[str]): The subject of the experimental warning. Defaults to a string\n            representation of the decorated object. This is useful when marking usage of\n            an experimental API inside an otherwise non-deprecated function, so\n            that it can be easily cleaned up later. It should only be used with\n            `emit_runtime_warning=False`, as we don\'t want to warn users when an\n            experimental API is used internally.\n\n    Usage:\n\n        .. code-block:: python\n\n            @experimental\n            def my_experimental_function(my_arg):\n                do_stuff()\n\n            @experimental\n            class MyExperimentalClass:\n                pass\n    '
    if __obj is None:
        return lambda obj: experimental(obj, additional_warn_text=additional_warn_text, subject=subject, emit_runtime_warning=emit_runtime_warning)
    else:
        target = _get_annotation_target(__obj)
        setattr(target, _EXPERIMENTAL_ATTR_NAME, ExperimentalInfo(additional_warn_text, subject))
        if emit_runtime_warning:
            warning_fn = lambda : experimental_warning(subject or _get_subject(__obj), additional_warn_text=additional_warn_text, stacklevel=_get_warning_stacklevel(__obj))
            return apply_pre_call_decorator(__obj, warning_fn)
        else:
            return __obj

def is_experimental(obj: Annotatable) -> bool:
    if False:
        print('Hello World!')
    target = _get_annotation_target(obj)
    return hasattr(target, _EXPERIMENTAL_ATTR_NAME) and getattr(target, _EXPERIMENTAL_ATTR_NAME)

def get_experimental_info(obj: Annotatable) -> ExperimentalInfo:
    if False:
        i = 10
        return i + 15
    target = _get_annotation_target(obj)
    return getattr(target, _EXPERIMENTAL_ATTR_NAME)
_EXPERIMENTAL_PARAM_ATTR_NAME: Final[str] = '_experimental_params'

@overload
def experimental_param(__obj: T_Annotatable, *, param: str, additional_warn_text: Optional[str]=..., emit_runtime_warning: bool=...) -> T_Annotatable:
    if False:
        i = 10
        return i + 15
    ...

@overload
def experimental_param(__obj: None=..., *, param: str, additional_warn_text: Optional[str]=..., emit_runtime_warning: bool=...) -> Callable[[T_Annotatable], T_Annotatable]:
    if False:
        return 10
    ...

def experimental_param(__obj: Optional[T_Annotatable]=None, *, param: str, additional_warn_text: Optional[str]=None, emit_runtime_warning: bool=True) -> Union[T_Annotatable, Callable[[T_Annotatable], T_Annotatable]]:
    if False:
        for i in range(10):
            print('nop')
    'Mark a parameter of a class initializer or function/method as experimental. This appends some\n    metadata to the decorated object that causes the specified argument to be rendered with an\n    "experimental" tag and associated warning in the docs.\n\n    If `emit_runtime_warning` is True, a warning will also be emitted when the function is called\n    and a non-None value is passed for the parameter. For consistency between docs and runtime\n    warnings, this decorator is preferred to manual calls to `experimental_warning`. Note that the\n    warning will only be emitted if the value is passed as a keyword argument.\n\n    Args:\n        param (str): The name of the parameter to mark experimental.\n        additional_warn_text (str): Additional text to display after the deprecation warning.\n            Typically this should suggest a newer API.\n        emit_runtime_warning (bool): Whether to emit a warning when the function is called.\n    '
    if __obj is None:
        return lambda obj: experimental_param(obj, param=param, additional_warn_text=additional_warn_text, emit_runtime_warning=emit_runtime_warning)
    else:
        check.invariant(_annotatable_has_param(__obj, param), f'Attempted to mark undefined parameter `{param}` experimental.')
        target = _get_annotation_target(__obj)
        if not hasattr(target, _EXPERIMENTAL_PARAM_ATTR_NAME):
            setattr(target, _EXPERIMENTAL_PARAM_ATTR_NAME, {})
        getattr(target, _EXPERIMENTAL_PARAM_ATTR_NAME)[param] = ExperimentalInfo(additional_warn_text=additional_warn_text)
        if emit_runtime_warning:
            condition = lambda *_, **kwargs: kwargs.get(param) is not None
            warning_fn = lambda : experimental_warning(_get_subject(__obj, param=param), additional_warn_text=additional_warn_text, stacklevel=4)
            return apply_pre_call_decorator(__obj, warning_fn, condition=condition)
        else:
            return __obj

def has_experimental_params(obj: Annotatable) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return hasattr(_get_annotation_target(obj), _EXPERIMENTAL_PARAM_ATTR_NAME)

def get_experimental_params(obj: Annotatable) -> Mapping[str, ExperimentalInfo]:
    if False:
        i = 10
        return i + 15
    return getattr(_get_annotation_target(obj), _EXPERIMENTAL_PARAM_ATTR_NAME)

def is_experimental_param(obj: Annotatable, param_name: str) -> bool:
    if False:
        while True:
            i = 10
    target = _get_annotation_target(obj)
    return param_name in getattr(target, _EXPERIMENTAL_PARAM_ATTR_NAME, {})

def get_experimental_param_info(obj: Annotatable, param_name: str) -> ExperimentalInfo:
    if False:
        return 10
    target = _get_annotation_target(obj)
    return getattr(target, _EXPERIMENTAL_PARAM_ATTR_NAME)[param_name]

def copy_annotations(dest: Annotatable, src: Annotatable) -> None:
    if False:
        print('Hello World!')
    'Copy all Dagster annotations from one object to another object.'
    dest_target = _get_annotation_target(dest)
    src_target = _get_annotation_target(src)
    if hasattr(src_target, _PUBLIC_ATTR_NAME):
        setattr(dest_target, _PUBLIC_ATTR_NAME, getattr(src_target, _PUBLIC_ATTR_NAME))
    if hasattr(src_target, _DEPRECATED_ATTR_NAME):
        setattr(dest_target, _DEPRECATED_ATTR_NAME, getattr(src_target, _DEPRECATED_ATTR_NAME))
    if hasattr(src_target, _DEPRECATED_PARAM_ATTR_NAME):
        setattr(dest_target, _DEPRECATED_PARAM_ATTR_NAME, getattr(src_target, _DEPRECATED_PARAM_ATTR_NAME))
    if hasattr(src_target, _EXPERIMENTAL_ATTR_NAME):
        setattr(dest_target, _EXPERIMENTAL_ATTR_NAME, getattr(src_target, _EXPERIMENTAL_ATTR_NAME))
    if hasattr(src_target, _EXPERIMENTAL_PARAM_ATTR_NAME):
        setattr(dest_target, _EXPERIMENTAL_PARAM_ATTR_NAME, getattr(src_target, _EXPERIMENTAL_PARAM_ATTR_NAME))

def _get_annotation_target(obj: Annotatable) -> object:
    if False:
        i = 10
        return i + 15
    "Given an object to be annotated, return the underlying object that will actually store the annotations.\n    This is necessary because not all objects are mutable, and so can't be annotated directly.\n    "
    if isinstance(obj, property):
        return obj.fget
    elif isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    else:
        return obj

def _get_subject(obj: Annotatable, param: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    'Get the string representation of an annotated object that will appear in\n    annotation-generated warnings about the object.\n    '
    if param:
        if isinstance(obj, type):
            return f'Parameter `{param}` of initializer `{obj.__qualname__}.__init__`'
        else:
            fn_subject = _get_subject(obj)
            return f'Parameter `{param}` of {fn_subject[:1].lower() + fn_subject[1:]}'
    elif isinstance(obj, type):
        return f'Class `{obj.__qualname__}`'
    elif isinstance(obj, property):
        return f'Property `{(obj.fget.__qualname__ if obj.fget else obj)}`'
    elif isinstance(obj, classmethod):
        return f'Class method `{_get_annotation_target(obj).__qualname__}`'
    elif isinstance(obj, staticmethod):
        return f'Static method `{_get_annotation_target(obj).__qualname__}`'
    elif inspect.isfunction(obj):
        return f'Function `{obj.__qualname__}`'
    elif is_resource_def(obj):
        return f'Dagster resource `{obj.__qualname__}`'
    else:
        check.failed(f'Unexpected object type: {type(obj)}')

def _get_warning_stacklevel(obj: Annotatable):
    if False:
        while True:
            i = 10
    "Get the stacklevel to use for warnings that are attached to a target via decorator.\n\n    The goal is to have the warning point to the line where the function in the\n    underlying object is actually invoked. This isn't straightforward\n    because some objects have complicated logic in between `__call__` and\n    the site at which a wrapped function containing the warning is actually\n    called. Can be determined through trial and error.\n    "
    if is_resource_def(obj):
        return 6
    else:
        return 4

def _annotatable_has_param(obj: Annotatable, param: str) -> bool:
    if False:
        while True:
            i = 10
    target_fn = get_decorator_target(obj)
    return param in inspect.signature(target_fn).parameters