import inspect
import warnings
from functools import wraps
from typing import Callable, TypeVar
T = TypeVar('T', bound=Callable)
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
EMPTY = inspect.Parameter.empty

def _deprecate_positional_args(version) -> Callable[[T], T]:
    if False:
        print('Hello World!')
    'Decorator for methods that issues warnings for positional arguments\n\n    Using the keyword-only argument syntax in pep 3102, arguments after the\n    ``*`` will issue a warning when passed as a positional argument.\n\n    Parameters\n    ----------\n    version : str\n        version of the library when the positional arguments were deprecated\n\n    Examples\n    --------\n    Deprecate passing `b` as positional argument:\n\n    def func(a, b=1):\n        pass\n\n    @_deprecate_positional_args("v0.1.0")\n    def func(a, *, b=2):\n        pass\n\n    func(1, 2)\n\n    Notes\n    -----\n    This function is adapted from scikit-learn under the terms of its license. See\n    licences/SCIKIT_LEARN_LICENSE\n    '

    def _decorator(func):
        if False:
            print('Hello World!')
        signature = inspect.signature(func)
        pos_or_kw_args = []
        kwonly_args = []
        for (name, param) in signature.parameters.items():
            if param.kind in (POSITIONAL_OR_KEYWORD, POSITIONAL_ONLY):
                pos_or_kw_args.append(name)
            elif param.kind == KEYWORD_ONLY:
                kwonly_args.append(name)
                if param.default is EMPTY:
                    raise TypeError('Keyword-only param without default disallowed.')

        @wraps(func)
        def inner(*args, **kwargs):
            if False:
                print('Hello World!')
            name = func.__name__
            n_extra_args = len(args) - len(pos_or_kw_args)
            if n_extra_args > 0:
                extra_args = ', '.join(kwonly_args[:n_extra_args])
                warnings.warn(f"Passing '{extra_args}' as positional argument(s) to {name} was deprecated in version {version} and will raise an error two releases later. Please pass them as keyword arguments.", FutureWarning, stacklevel=2)
                zip_args = zip(kwonly_args[:n_extra_args], args[-n_extra_args:])
                kwargs.update({name: arg for (name, arg) in zip_args})
                return func(*args[:-n_extra_args], **kwargs)
            return func(*args, **kwargs)
        return inner
    return _decorator