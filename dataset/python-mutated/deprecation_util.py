from __future__ import annotations
import functools
from typing import Any, Callable, List, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
_LOGGER = get_logger(__name__)
TFunc = TypeVar('TFunc', bound=Callable[..., Any])
TObj = TypeVar('TObj', bound=object)

def _should_show_deprecation_warning_in_browser() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True if we should print deprecation warnings to the browser.'
    return bool(config.get_option('client.showErrorDetails'))

def show_deprecation_warning(message: str) -> None:
    if False:
        i = 10
        return i + 15
    'Show a deprecation warning message.'
    if _should_show_deprecation_warning_in_browser():
        streamlit.warning(message)
    _LOGGER.warning(message)

def make_deprecated_name_warning(old_name: str, new_name: str, removal_date: str, extra_message: str | None=None, include_st_prefix: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    if include_st_prefix:
        old_name = f'st.{old_name}'
        new_name = f'st.{new_name}'
    return f'Please replace `{old_name}` with `{new_name}`.\n\n`{old_name}` will be removed after {removal_date}.' + (f'\n\n{extra_message}' if extra_message else '')

def deprecate_func_name(func: TFunc, old_name: str, removal_date: str, extra_message: str | None=None, name_override: str | None=None) -> TFunc:
    if False:
        while True:
            i = 10
    'Wrap an `st` function whose name has changed.\n\n    Wrapped functions will run as normal, but will also show an st.warning\n    saying that the old name will be removed after removal_date.\n\n    (We generally set `removal_date` to 3 months from the deprecation date.)\n\n    Parameters\n    ----------\n    func\n        The `st.` function whose name has changed.\n\n    old_name\n        The function\'s deprecated name within __init__.py.\n\n    removal_date\n        A date like "2020-01-01", indicating the last day we\'ll guarantee\n        support for the deprecated name.\n\n    extra_message\n        An optional extra message to show in the deprecation warning.\n\n    name_override\n        An optional name to use in place of func.__name__.\n    '

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        if False:
            print('Hello World!')
        result = func(*args, **kwargs)
        show_deprecation_warning(make_deprecated_name_warning(old_name, name_override or func.__name__, removal_date, extra_message))
        return result
    wrapped_func.__name__ = old_name
    wrapped_func.__doc__ = func.__doc__
    return cast(TFunc, wrapped_func)

def deprecate_obj_name(obj: TObj, old_name: str, new_name: str, removal_date: str, include_st_prefix: bool=True) -> TObj:
    if False:
        print('Hello World!')
    'Wrap an `st` object whose name has changed.\n\n    Wrapped objects will behave as normal, but will also show an st.warning\n    saying that the old name will be removed after `removal_date`.\n\n    (We generally set `removal_date` to 3 months from the deprecation date.)\n\n    Parameters\n    ----------\n    obj\n        The `st.` object whose name has changed.\n\n    old_name\n        The object\'s deprecated name within __init__.py.\n\n    new_name\n        The object\'s new name within __init__.py.\n\n    removal_date\n        A date like "2020-01-01", indicating the last day we\'ll guarantee\n        support for the deprecated name.\n\n    include_st_prefix\n        If False, does not prefix each of the object names in the deprecation\n        essage with `st.*`. Defaults to True.\n    '
    return _create_deprecated_obj_wrapper(obj, lambda : show_deprecation_warning(make_deprecated_name_warning(old_name, new_name, removal_date, include_st_prefix=include_st_prefix)))

def _create_deprecated_obj_wrapper(obj: TObj, show_warning: Callable[[], Any]) -> TObj:
    if False:
        return 10
    "Create a wrapper for an object that has been deprecated. The first\n    time one of the object's properties or functions is accessed, the\n    given `show_warning` callback will be called.\n    "
    has_shown_warning = False

    def maybe_show_warning() -> None:
        if False:
            i = 10
            return i + 15
        nonlocal has_shown_warning
        if not has_shown_warning:
            has_shown_warning = True
            show_warning()

    class Wrapper:

        def __init__(self):
            if False:
                while True:
                    i = 10
            for name in Wrapper._get_magic_functions(obj.__class__):
                setattr(self.__class__, name, property(self._make_magic_function_proxy(name)))

        def __getattr__(self, attr):
            if False:
                for i in range(10):
                    print('nop')
            if attr in self.__dict__:
                return getattr(self, attr)
            maybe_show_warning()
            return getattr(obj, attr)

        @staticmethod
        def _get_magic_functions(cls) -> List[str]:
            if False:
                i = 10
                return i + 15
            ignore = ('__class__', '__dict__', '__getattribute__', '__getattr__')
            return [name for name in dir(cls) if name not in ignore and name.startswith('__')]

        @staticmethod
        def _make_magic_function_proxy(name):
            if False:
                return 10

            def proxy(self, *args):
                if False:
                    for i in range(10):
                        print('nop')
                maybe_show_warning()
                return getattr(obj, name)
            return proxy
    return cast(TObj, Wrapper())