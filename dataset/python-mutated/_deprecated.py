import functools
import textwrap
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings
from packaging import version
from optuna._experimental import _get_docstring_indent
from optuna._experimental import _validate_version
if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    FT = TypeVar('FT')
    FP = ParamSpec('FP')
    CT = TypeVar('CT')
_DEPRECATION_NOTE_TEMPLATE = '\n\n.. warning::\n    Deprecated in v{d_ver}. This feature will be removed in the future. The removal of this\n    feature is currently scheduled for v{r_ver}, but this schedule is subject to change.\n    See https://github.com/optuna/optuna/releases/tag/v{d_ver}.\n'
_DEPRECATION_WARNING_TEMPLATE = '{name} has been deprecated in v{d_ver}. This feature will be removed in v{r_ver}. See https://github.com/optuna/optuna/releases/tag/v{d_ver}.'

def _validate_two_version(old_version: str, new_version: str) -> None:
    if False:
        return 10
    if version.parse(old_version) > version.parse(new_version):
        raise ValueError('Invalid version relationship. The deprecated version must be smaller than the removed version, but (deprecated version, removed version) = ({}, {}) are specified.'.format(old_version, new_version))

def _format_text(text: str) -> str:
    if False:
        print('Hello World!')
    return '\n\n' + textwrap.indent(text.strip(), '    ') + '\n'

def deprecated_func(deprecated_version: str, removed_version: str, name: Optional[str]=None, text: Optional[str]=None) -> 'Callable[[Callable[FP, FT]], Callable[FP, FT]]':
    if False:
        i = 10
        return i + 15
    'Decorate function as deprecated.\n\n    Args:\n        deprecated_version:\n            The version in which the target feature is deprecated.\n        removed_version:\n            The version in which the target feature will be removed.\n        name:\n            The name of the feature. Defaults to the function name. Optional.\n        text:\n            The additional text for the deprecation note. The default note is build using specified\n            ``deprecated_version`` and ``removed_version``. If you want to provide additional\n            information, please specify this argument yourself.\n\n            .. note::\n                The default deprecation note is as follows: "Deprecated in v{d_ver}. This feature\n                will be removed in the future. The removal of this feature is currently scheduled\n                for v{r_ver}, but this schedule is subject to change. See\n                https://github.com/optuna/optuna/releases/tag/v{d_ver}."\n\n            .. note::\n                The specified text is concatenated after the default deprecation note.\n    '
    _validate_version(deprecated_version)
    _validate_version(removed_version)
    _validate_two_version(deprecated_version, removed_version)

    def decorator(func: 'Callable[FP, FT]') -> 'Callable[FP, FT]':
        if False:
            while True:
                i = 10
        if func.__doc__ is None:
            func.__doc__ = ''
        note = _DEPRECATION_NOTE_TEMPLATE.format(d_ver=deprecated_version, r_ver=removed_version)
        if text is not None:
            note += _format_text(text)
        indent = _get_docstring_indent(func.__doc__)
        func.__doc__ = func.__doc__.strip() + textwrap.indent(note, indent) + indent

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> 'FT':
            if False:
                return 10
            'Decorates a function as deprecated.\n\n            This decorator is supposed to be applied to the deprecated function.\n            '
            message = _DEPRECATION_WARNING_TEMPLATE.format(name=name if name is not None else func.__name__, d_ver=deprecated_version, r_ver=removed_version)
            if text is not None:
                message += ' ' + text
            warnings.warn(message, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def deprecated_class(deprecated_version: str, removed_version: str, name: Optional[str]=None, text: Optional[str]=None) -> 'Callable[[CT], CT]':
    if False:
        i = 10
        return i + 15
    'Decorate class as deprecated.\n\n    Args:\n        deprecated_version:\n            The version in which the target feature is deprecated.\n        removed_version:\n            The version in which the target feature will be removed.\n        name:\n            The name of the feature. Defaults to the class name. Optional.\n        text:\n            The additional text for the deprecation note. The default note is build using specified\n            ``deprecated_version`` and ``removed_version``. If you want to provide additional\n            information, please specify this argument yourself.\n\n            .. note::\n                The default deprecation note is as follows: "Deprecated in v{d_ver}. This feature\n                will be removed in the future. The removal of this feature is currently scheduled\n                for v{r_ver}, but this schedule is subject to change. See\n                https://github.com/optuna/optuna/releases/tag/v{d_ver}."\n\n            .. note::\n                The specified text is concatenated after the default deprecation note.\n    '
    _validate_version(deprecated_version)
    _validate_version(removed_version)
    _validate_two_version(deprecated_version, removed_version)

    def decorator(cls: 'CT') -> 'CT':
        if False:
            print('Hello World!')

        def wrapper(cls: 'CT') -> 'CT':
            if False:
                i = 10
                return i + 15
            'Decorates a class as deprecated.\n\n            This decorator is supposed to be applied to the deprecated class.\n            '
            _original_init = getattr(cls, '__init__')
            _original_name = getattr(cls, '__name__')

            @functools.wraps(_original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                if False:
                    while True:
                        i = 10
                message = _DEPRECATION_WARNING_TEMPLATE.format(name=name if name is not None else _original_name, d_ver=deprecated_version, r_ver=removed_version)
                if text is not None:
                    message += ' ' + text
                warnings.warn(message, FutureWarning, stacklevel=2)
                _original_init(self, *args, **kwargs)
            setattr(cls, '__init__', wrapped_init)
            if cls.__doc__ is None:
                cls.__doc__ = ''
            note = _DEPRECATION_NOTE_TEMPLATE.format(d_ver=deprecated_version, r_ver=removed_version)
            if text is not None:
                note += _format_text(text)
            indent = _get_docstring_indent(cls.__doc__)
            cls.__doc__ = cls.__doc__.strip() + textwrap.indent(note, indent) + indent
            return cls
        return wrapper(cls)
    return decorator