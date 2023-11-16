import types
from typing import Any, Optional
from streamlit import type_util
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException, StreamlitAPIWarning
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
CACHE_DOCS_URL = 'https://docs.streamlit.io/library/advanced-features/caching'

def get_cached_func_name_md(func: Any) -> str:
    if False:
        return 10
    'Get markdown representation of the function name.'
    if hasattr(func, '__name__'):
        return '`%s()`' % func.__name__
    elif hasattr(type(func), '__name__'):
        return f'`{type(func).__name__}`'
    return f'`{type(func)}`'

def get_return_value_type(return_value: Any) -> str:
    if False:
        print('Hello World!')
    if hasattr(return_value, '__module__') and hasattr(type(return_value), '__name__'):
        return f'`{return_value.__module__}.{type(return_value).__name__}`'
    return get_cached_func_name_md(return_value)

class UnhashableTypeError(Exception):
    pass

class UnhashableParamError(StreamlitAPIException):

    def __init__(self, cache_type: CacheType, func: types.FunctionType, arg_name: Optional[str], arg_value: Any, orig_exc: BaseException):
        if False:
            for i in range(10):
                print('nop')
        msg = self._create_message(cache_type, func, arg_name, arg_value)
        super().__init__(msg)
        self.with_traceback(orig_exc.__traceback__)

    @staticmethod
    def _create_message(cache_type: CacheType, func: types.FunctionType, arg_name: Optional[str], arg_value: Any) -> str:
        if False:
            print('Hello World!')
        arg_name_str = arg_name if arg_name is not None else '(unnamed)'
        arg_type = type_util.get_fqn_type(arg_value)
        func_name = func.__name__
        arg_replacement_name = f'_{arg_name}' if arg_name is not None else '_arg'
        return f"\nCannot hash argument '{arg_name_str}' (of type `{arg_type}`) in '{func_name}'.\n\nTo address this, you can tell Streamlit not to hash this argument by adding a\nleading underscore to the argument's name in the function signature:\n\n```\n@st.{get_decorator_api_name(cache_type)}\ndef {func_name}({arg_replacement_name}, ...):\n    ...\n```\n            ".strip('\n')

class CacheKeyNotFoundError(Exception):
    pass

class CacheError(Exception):
    pass

class CachedStFunctionWarning(StreamlitAPIWarning):

    def __init__(self, cache_type: CacheType, st_func_name: str, cached_func: types.FunctionType):
        if False:
            print('Hello World!')
        args = {'st_func_name': f'`st.{st_func_name}()`', 'func_name': self._get_cached_func_name_md(cached_func), 'decorator_name': get_decorator_api_name(cache_type)}
        msg = ('\nYour script uses %(st_func_name)s to write to your Streamlit app from within\nsome cached code at %(func_name)s. This code will only be called when we detect\na cache "miss", which can lead to unexpected results.\n\nHow to fix this:\n* Move the %(st_func_name)s call outside %(func_name)s.\n* Or, if you know what you\'re doing, use `@st.%(decorator_name)s(experimental_allow_widgets=True)`\nto enable widget replay and suppress this warning.\n            ' % args).strip('\n')
        super().__init__(msg)

    @staticmethod
    def _get_cached_func_name_md(func: types.FunctionType) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get markdown representation of the function name.'
        if hasattr(func, '__name__'):
            return '`%s()`' % func.__name__
        else:
            return 'a cached function'

class CacheReplayClosureError(StreamlitAPIException):

    def __init__(self, cache_type: CacheType, cached_func: types.FunctionType):
        if False:
            while True:
                i = 10
        func_name = get_cached_func_name_md(cached_func)
        decorator_name = get_decorator_api_name(cache_type)
        msg = f'\nWhile running {func_name}, a streamlit element is called on some layout block created outside the function.\nThis is incompatible with replaying the cached effect of that element, because the\nthe referenced block might not exist when the replay happens.\n\nHow to fix this:\n* Move the creation of $THING inside {func_name}.\n* Move the call to the streamlit element outside of {func_name}.\n* Remove the `@st.{decorator_name}` decorator from {func_name}.\n            '.strip('\n')
        super().__init__(msg)

class UnserializableReturnValueError(MarkdownFormattedException):

    def __init__(self, func: types.FunctionType, return_value: types.FunctionType):
        if False:
            return 10
        MarkdownFormattedException.__init__(self, f'\n            Cannot serialize the return value (of type {get_return_value_type(return_value)}) in {get_cached_func_name_md(func)}.\n            `st.cache_data` uses [pickle](https://docs.python.org/3/library/pickle.html) to\n            serialize the function’s return value and safely store it in the cache without mutating the original object. Please convert the return value to a pickle-serializable type.\n            If you want to cache unserializable objects such as database connections or Tensorflow\n            sessions, use `st.cache_resource` instead (see [our docs]({CACHE_DOCS_URL}) for differences).')

class UnevaluatedDataFrameError(StreamlitAPIException):
    """Used to display a message about uncollected dataframe being used"""
    pass

class BadTTLStringError(StreamlitAPIException):
    """Raised when a bad ttl= argument string is passed."""

    def __init__(self, ttl: str):
        if False:
            return 10
        MarkdownFormattedException.__init__(self, f"TTL string doesn't look right. It should be formatted as`'1d2h34m'` or `2 days`, for example. Got: {ttl}")