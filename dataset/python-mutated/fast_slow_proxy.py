from __future__ import annotations
import functools
import inspect
import operator
import pickle
import types
from collections.abc import Iterator
from enum import IntEnum
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Set, Tuple, Type
from .annotation import nvtx
_CUDF_PANDAS_NVTX_COLORS = {'COPY_SLOW_TO_FAST': 13238304, 'COPY_FAST_TO_SLOW': 16033154, 'EXECUTE_FAST': 9618910, 'EXECUTE_SLOW': 356784}
_WRAPPER_ASSIGNMENTS = tuple((attr for attr in functools.WRAPPER_ASSIGNMENTS if attr not in ('__annotations__', '__doc__')))

def callers_module_name():
    if False:
        i = 10
        return i + 15
    return inspect.currentframe().f_back.f_back.f_globals['__name__']

class _State(IntEnum):
    """Simple enum to track the type of wrapped object of a final proxy"""
    SLOW = 0
    FAST = 1

class _Unusable:
    """
    A totally unusable type. When a "fast" object is not available,
    it's useful to set it to _Unusable() so that any operations
    on it fail, and ensure fallback to the corresponding
    "slow" object.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if False:
            return 10
        raise NotImplementedError('Fast implementation not available. Falling back to the slow implementation')

    def __getattribute__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        if name in {'__class__'}:
            return super().__getattribute__(name)
        raise TypeError('Unusable type. Falling back to the slow object')

class _PickleConstructor:
    """A pickleable object to support construction in __reduce__.

    This object is used to avoid having unpickling call __init__ on the
    objects, instead only invoking __new__. __init__ may have required
    arguments or otherwise perform invalid initialization that we could skip
    altogether since we're going to overwrite the wrapped object.
    """

    def __init__(self, type_):
        if False:
            i = 10
            return i + 15
        self._type = type_

    def __call__(self):
        if False:
            return 10
        return object.__new__(self._type)
_DELETE = object()

def make_final_proxy_type(name: str, fast_type: type, slow_type: type, *, fast_to_slow: Callable, slow_to_fast: Callable, module: Optional[str]=None, additional_attributes: Mapping[str, Any] | None=None, postprocess: Callable[[_FinalProxy, Any, Any], Any] | None=None, bases: Tuple=()) -> Type[_FinalProxy]:
    if False:
        i = 10
        return i + 15
    '\n    Defines a fast-slow proxy type for a pair of "final" fast and slow\n    types. Final types are types for which known operations exist for\n    converting an object of "fast" type to "slow" and vice-versa.\n\n    Parameters\n    ----------\n    name: str\n        The name of the class returned\n    fast_type: type\n    slow_type: type\n    fast_to_slow: callable\n        Function that accepts a single argument of type `fast_type`\n        and returns an object of type `slow_type`\n    slow_to_fast: callable\n        Function that accepts a single argument of type `slow_type`\n        and returns an object of type `fast_type`\n    additional_attributes\n        Mapping of additional attributes to add to the class\n       (optional), these will override any defaulted attributes (e.g.\n       ``__init__`). If you want to remove a defaulted attribute\n       completely, pass the special sentinel ``_DELETE`` as a value.\n    postprocess\n        Optional function called to allow the proxy to postprocess\n        itself when being wrapped up, called with the proxy object,\n        the unwrapped result object, and the function that was used to\n        construct said unwrapped object. See also `_maybe_wrap_result`.\n    bases\n        Optional tuple of base classes to insert into the mro.\n\n    Notes\n    -----\n    As a side-effect, this function adds `fast_type` and `slow_type`\n    to a global mapping of final types to their corresponding proxy\n    types, accessible via `get_final_type_map()`.\n    '

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        _fast_slow_function_call(lambda cls, args, kwargs: setattr(self, '_fsproxy_wrapped', cls(*args, **kwargs)), type(self), args, kwargs)

    @nvtx.annotate('COPY_SLOW_TO_FAST', color=_CUDF_PANDAS_NVTX_COLORS['COPY_SLOW_TO_FAST'], domain='cudf_pandas')
    def _fsproxy_slow_to_fast(self):
        if False:
            while True:
                i = 10
        if self._fsproxy_state is _State.SLOW:
            return slow_to_fast(self._fsproxy_wrapped)
        return self._fsproxy_wrapped

    @nvtx.annotate('COPY_FAST_TO_SLOW', color=_CUDF_PANDAS_NVTX_COLORS['COPY_FAST_TO_SLOW'], domain='cudf_pandas')
    def _fsproxy_fast_to_slow(self):
        if False:
            for i in range(10):
                print('nop')
        if self._fsproxy_state is _State.FAST:
            return fast_to_slow(self._fsproxy_wrapped)
        return self._fsproxy_wrapped

    @property
    def _fsproxy_state(self) -> _State:
        if False:
            for i in range(10):
                print('nop')
        return _State.FAST if isinstance(self._fsproxy_wrapped, self._fsproxy_fast_type) else _State.SLOW

    def __reduce__(self):
        if False:
            print('Hello World!')
        from .module_accelerator import disable_module_accelerator
        with disable_module_accelerator():
            pickled_wrapped_obj = pickle.dumps(self._fsproxy_wrapped)
        return (_PickleConstructor(type(self)), (), pickled_wrapped_obj)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        from .module_accelerator import disable_module_accelerator
        with disable_module_accelerator():
            unpickled_wrapped_obj = pickle.loads(state)
        self._fsproxy_wrapped = unpickled_wrapped_obj
    slow_dir = dir(slow_type)
    cls_dict = {'__init__': __init__, '__doc__': inspect.getdoc(slow_type), '_fsproxy_slow_dir': slow_dir, '_fsproxy_fast_type': fast_type, '_fsproxy_slow_type': slow_type, '_fsproxy_slow_to_fast': _fsproxy_slow_to_fast, '_fsproxy_fast_to_slow': _fsproxy_fast_to_slow, '_fsproxy_state': _fsproxy_state, '__reduce__': __reduce__, '__setstate__': __setstate__}
    if additional_attributes is None:
        additional_attributes = {}
    for method in _SPECIAL_METHODS:
        if getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)
    for (k, v) in additional_attributes.items():
        if v is _DELETE and k in cls_dict:
            del cls_dict[k]
        elif v is not _DELETE:
            cls_dict[k] = v
    cls = types.new_class(name, (*bases, _FinalProxy), {'metaclass': _FastSlowProxyMeta}, lambda ns: ns.update(cls_dict))
    functools.update_wrapper(cls, slow_type, assigned=_WRAPPER_ASSIGNMENTS, updated=())
    cls.__module__ = module if module is not None else callers_module_name()
    final_type_map = get_final_type_map()
    if fast_type is not _Unusable:
        final_type_map[fast_type] = cls
    final_type_map[slow_type] = cls
    return cls

def make_intermediate_proxy_type(name: str, fast_type: type, slow_type: type, *, module: Optional[str]=None) -> Type[_IntermediateProxy]:
    if False:
        while True:
            i = 10
    '\n    Defines a proxy type for a pair of "intermediate" fast and slow\n    types. Intermediate types are the types of the results of\n    operations invoked on final types.\n\n    As a side-effect, this function adds `fast_type` and `slow_type`\n    to a global mapping of intermediate types to their corresponding\n    proxy types, accessible via `get_intermediate_type_map()`.\n\n    Parameters\n    ----------\n    name: str\n        The name of the class returned\n    fast_type: type\n    slow_type: type\n    '

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise TypeError(f'Cannot directly instantiate object of type {type(self)}')

    @property
    def _fsproxy_state(self):
        if False:
            for i in range(10):
                print('nop')
        return _State.FAST if isinstance(self._fsproxy_wrapped, self._fsproxy_fast_type) else _State.SLOW

    @nvtx.annotate('COPY_SLOW_TO_FAST', color=_CUDF_PANDAS_NVTX_COLORS['COPY_SLOW_TO_FAST'], domain='cudf_pandas')
    def _fsproxy_slow_to_fast(self):
        if False:
            while True:
                i = 10
        if self._fsproxy_state is _State.SLOW:
            return super(type(self), self)._fsproxy_slow_to_fast()
        return self._fsproxy_wrapped

    @nvtx.annotate('COPY_FAST_TO_SLOW', color=_CUDF_PANDAS_NVTX_COLORS['COPY_FAST_TO_SLOW'], domain='cudf_pandas')
    def _fsproxy_fast_to_slow(self):
        if False:
            return 10
        if self._fsproxy_state is _State.FAST:
            return super(type(self), self)._fsproxy_fast_to_slow()
        return self._fsproxy_wrapped
    slow_dir = dir(slow_type)
    cls_dict = {'__init__': __init__, '__doc__': inspect.getdoc(slow_type), '_fsproxy_slow_dir': slow_dir, '_fsproxy_fast_type': fast_type, '_fsproxy_slow_type': slow_type, '_fsproxy_slow_to_fast': _fsproxy_slow_to_fast, '_fsproxy_fast_to_slow': _fsproxy_fast_to_slow, '_fsproxy_state': _fsproxy_state}
    for method in _SPECIAL_METHODS:
        if getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)
    cls = types.new_class(name, (_IntermediateProxy,), {'metaclass': _FastSlowProxyMeta}, lambda ns: ns.update(cls_dict))
    functools.update_wrapper(cls, slow_type, assigned=_WRAPPER_ASSIGNMENTS, updated=())
    cls.__module__ = module if module is not None else callers_module_name()
    intermediate_type_map = get_intermediate_type_map()
    if fast_type is not _Unusable:
        intermediate_type_map[fast_type] = cls
    intermediate_type_map[slow_type] = cls
    return cls

def register_proxy_func(slow_func: Callable):
    if False:
        return 10
    '\n    Decorator to register custom function as a proxy for slow_func.\n\n    Parameters\n    ----------\n    slow_func: Callable\n        The function to register a wrapper for.\n\n    Returns\n    -------\n    Callable\n    '

    def wrapper(func):
        if False:
            while True:
                i = 10
        registered_functions = get_registered_functions()
        registered_functions[slow_func] = func
        functools.update_wrapper(func, slow_func)
        return func
    return wrapper

@functools.lru_cache(maxsize=None)
def get_final_type_map():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the mapping of all known fast and slow final types to their\n    corresponding proxy types.\n    '
    return dict()

@functools.lru_cache(maxsize=None)
def get_intermediate_type_map():
    if False:
        while True:
            i = 10
    '\n    Return a mapping of all known fast and slow intermediate types to their\n    corresponding proxy types.\n    '
    return dict()

@functools.lru_cache(maxsize=None)
def get_registered_functions():
    if False:
        return 10
    return dict()

def _raise_attribute_error(obj, name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Raise an AttributeError with a message that is consistent with\n    the error raised by Python for a non-existent attribute on a\n    proxy object.\n    '
    raise AttributeError(f"'{obj}' object has no attribute '{name}'")

class _FastSlowAttribute:
    """
    A descriptor type used to define attributes of fast-slow proxies.
    """

    def __init__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self._name = name

    def __get__(self, obj, owner=None) -> Any:
        if False:
            i = 10
            return i + 15
        if obj is None:
            obj = owner
        if not (isinstance(obj, _FastSlowProxy) or issubclass(type(obj), _FastSlowProxyMeta)):
            _raise_attribute_error(owner if owner else obj, self._name)
        (result, _) = _fast_slow_function_call(getattr, obj, self._name)
        if isinstance(result, functools.cached_property):
            result = property(result.func)
        if isinstance(result, (_MethodProxy, property)):
            from .module_accelerator import disable_module_accelerator
            type_ = owner if owner else type(obj)
            slow_result_type = getattr(type_._fsproxy_slow, self._name)
            with disable_module_accelerator():
                result.__doc__ = inspect.getdoc(slow_result_type)
            if isinstance(result, _MethodProxy):
                result._fsproxy_slow_dir = dir(slow_result_type)
        return result

class _FastSlowProxyMeta(type):
    """
    Metaclass used to dynamically find class attributes and
    classmethods of fast-slow proxy types.
    """

    @property
    def _fsproxy_slow(self) -> type:
        if False:
            while True:
                i = 10
        return self._fsproxy_slow_type

    @property
    def _fsproxy_fast(self) -> type:
        if False:
            print('Hello World!')
        return self._fsproxy_fast_type

    def __dir__(self):
        if False:
            while True:
                i = 10
        try:
            return self._fsproxy_slow_dir
        except AttributeError:
            return type.__dir__(self)

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        if name.startswith('_fsproxy') or name.startswith('__'):
            _raise_attribute_error(self.__class__.__name__, name)
        attr = _FastSlowAttribute(name)
        return attr.__get__(None, owner=self)

    def __subclasscheck__(self, __subclass: type) -> bool:
        if False:
            while True:
                i = 10
        if super().__subclasscheck__(__subclass):
            return True
        if hasattr(__subclass, '_fsproxy_slow'):
            return issubclass(__subclass._fsproxy_slow, self._fsproxy_slow)
        return False

    def __instancecheck__(self, __instance: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if super().__instancecheck__(__instance):
            return True
        elif hasattr(type(__instance), '_fsproxy_slow'):
            return issubclass(type(__instance), self)
        return False

class _FastSlowProxy:
    """
    Base class for all fast=slow proxy types.

    A fast-slow proxy is proxy for a pair of types that provide "fast"
    and "slow" implementations of the same API.  At any time, a
    fast-slow proxy wraps an object of either "fast" type, or "slow"
    type. Operations invoked on the fast-slow proxy are first
    delegated to the "fast" type, and if that fails, to the "slow"
    type.
    """
    _fsproxy_wrapped: Any

    def _fsproxy_fast_to_slow(self) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        If the wrapped object is of "fast" type, returns the\n        corresponding "slow" object. Otherwise, returns the wrapped\n        object as-is.\n        '
        raise NotImplementedError('Abstract base class')

    def _fsproxy_slow_to_fast(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        If the wrapped object is of "slow" type, returns the\n        corresponding "fast" object. Otherwise, returns the wrapped\n        object as-is.\n        '
        raise NotImplementedError('Abstract base class')

    @property
    def _fsproxy_fast(self) -> Any:
        if False:
            return 10
        '\n        Returns the wrapped object. If the wrapped object is of "slow"\n        type, replaces it with the corresponding "fast" object before\n        returning it.\n        '
        self._fsproxy_wrapped = self._fsproxy_slow_to_fast()
        return self._fsproxy_wrapped

    @property
    def _fsproxy_slow(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the wrapped object. If the wrapped object is of "fast"\n        type, replaces it with the corresponding "slow" object before\n        returning it.\n        '
        self._fsproxy_wrapped = self._fsproxy_fast_to_slow()
        return self._fsproxy_wrapped

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._fsproxy_slow_dir
        except AttributeError:
            return object.__dir__(self)

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        if name.startswith('_fsproxy'):
            _raise_attribute_error(self.__class__.__name__, name)
        if name in {'_ipython_canary_method_should_not_exist_', '_ipython_display_', '_repr_mimebundle_', '__array_struct__'}:
            _raise_attribute_error(self.__class__.__name__, name)
        if name.startswith('_'):
            return getattr(self._fsproxy_slow, name)
        attr = _FastSlowAttribute(name)
        return attr.__get__(self)

    def __setattr__(self, name, value):
        if False:
            i = 10
            return i + 15
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        return _FastSlowAttribute('__setattr__').__get__(self)(name, value)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.add, self, other)[0]

    def __radd__(self, other):
        if False:
            return 10
        return _fast_slow_function_call(operator.add, other, self)[0]

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.sub, self, other)[0]

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return _fast_slow_function_call(operator.sub, other, self)[0]

    def __mul__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.mul, self, other)[0]

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.mul, other, self)[0]

    def __truediv__(self, other):
        if False:
            while True:
                i = 10
        return _fast_slow_function_call(operator.truediv, self, other)[0]

    def __rtruediv__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.truediv, other, self)[0]

    def __floordiv__(self, other):
        if False:
            while True:
                i = 10
        return _fast_slow_function_call(operator.floordiv, self, other)[0]

    def __rfloordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.floordiv, other, self)[0]

    def __mod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.mod, self, other)[0]

    def __rmod__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.mod, other, self)[0]

    def __divmod__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(divmod, self, other)[0]

    def __rdivmod__(self, other):
        if False:
            while True:
                i = 10
        return _fast_slow_function_call(divmod, other, self)[0]

    def __pow__(self, other):
        if False:
            while True:
                i = 10
        return _fast_slow_function_call(operator.pow, self, other)[0]

    def __rpow__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.pow, other, self)[0]

    def __lshift__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.lshift, self, other)[0]

    def __rlshift__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.lshift, other, self)[0]

    def __rshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.rshift, self, other)[0]

    def __rrshift__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.rshift, other, self)[0]

    def __and__(self, other):
        if False:
            i = 10
            return i + 15
        return _fast_slow_function_call(operator.and_, self, other)[0]

    def __rand__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.and_, other, self)[0]

    def __xor__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.xor, self, other)[0]

    def __rxor__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.xor, other, self)[0]

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return _fast_slow_function_call(operator.or_, self, other)[0]

    def __ror__(self, other):
        if False:
            return 10
        return _fast_slow_function_call(operator.or_, other, self)[0]

    def __matmul__(self, other):
        if False:
            return 10
        return _fast_slow_function_call(operator.matmul, self, other)[0]

    def __rmatmul__(self, other):
        if False:
            print('Hello World!')
        return _fast_slow_function_call(operator.matmul, other, self)[0]

class _FinalProxy(_FastSlowProxy):
    """
    Proxy type for a pair of fast and slow "final" types for which
    there is a known conversion from fast to slow, and vice-versa.
    The conversion between fast and slow types is done using
    user-provided conversion functions.

    Do not attempt to use this class directly. Instead, use
    `make_final_proxy_type` to create subtypes.
    """

    @classmethod
    def _fsproxy_wrap(cls, value, func):
        if False:
            return 10
        'Default mechanism to wrap a value in a proxy type\n\n        Parameters\n        ----------\n        cls\n            The proxy type\n        value\n            The value to wrap up\n        func\n            The function called that constructed value\n\n        Returns\n        -------\n        A new proxied object\n\n        Notes\n        -----\n        _FinalProxy subclasses can override this classmethod if they\n        need particular behaviour when wrapped up.\n        '
        proxy = object.__new__(cls)
        proxy._fsproxy_wrapped = value
        return proxy

class _IntermediateProxy(_FastSlowProxy):
    """
    Proxy type for a pair of "intermediate" types that appear as
    intermediate values when invoking operations on "final" types.
    The conversion between fast and slow types is done by keeping
    track of the sequence of operations that created the wrapped
    object, and "playing back" that sequence starting from the "slow"
    version of the originating _FinalProxy.

    Do not attempt to use this class directly. Instead, use
    `make_intermediate_proxy_type` to create subtypes.
    """
    _method_chain: Tuple[Callable, Tuple, Dict]

    @classmethod
    def _fsproxy_wrap(cls, obj: Any, method_chain: Tuple[Callable, Tuple, Dict]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        obj: The object to wrap\n        method_chain: A tuple of the form (func, args, kwargs) where\n            `func` is the function that was called to create `obj`,\n            and `args` and `kwargs` are the arguments that were passed\n            to `func`.\n        '
        proxy = object.__new__(cls)
        proxy._fsproxy_wrapped = obj
        proxy._method_chain = method_chain
        return proxy

    @nvtx.annotate('COPY_SLOW_TO_FAST', color=_CUDF_PANDAS_NVTX_COLORS['COPY_SLOW_TO_FAST'], domain='cudf_pandas')
    def _fsproxy_slow_to_fast(self) -> Any:
        if False:
            i = 10
            return i + 15
        (func, args, kwargs) = self._method_chain
        (args, kwargs) = (_fast_arg(args), _fast_arg(kwargs))
        return func(*args, **kwargs)

    @nvtx.annotate('COPY_FAST_TO_SLOW', color=_CUDF_PANDAS_NVTX_COLORS['COPY_FAST_TO_SLOW'], domain='cudf_pandas')
    def _fsproxy_fast_to_slow(self) -> Any:
        if False:
            print('Hello World!')
        (func, args, kwargs) = self._method_chain
        (args, kwargs) = (_slow_arg(args), _slow_arg(kwargs))
        return func(*args, **kwargs)

class _CallableProxyMixin:
    """
    Mixin class that implements __call__ for fast-slow proxies.
    """
    __class__ = types.FunctionType

    def __call__(self, *args, **kwargs) -> Any:
        if False:
            return 10
        (result, _) = _fast_slow_function_call(lambda fn, args, kwargs: fn(*args, **kwargs), self, args, kwargs)
        return result

class _FunctionProxy(_CallableProxyMixin):
    """
    Proxy for a pair of fast and slow functions.
    """
    __name__: str

    def __init__(self, fast: Callable | _Unusable, slow: Callable):
        if False:
            while True:
                i = 10
        self._fsproxy_fast = fast
        self._fsproxy_slow = slow
        functools.update_wrapper(self, slow)

class _MethodProxy(_CallableProxyMixin, _IntermediateProxy):
    """
    Methods of fast-slow proxies are of type _MethodProxy.
    """

def _fast_slow_function_call(func: Callable, /, *args, **kwargs) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Call `func` with all `args` and `kwargs` converted to their\n    respective fast type. If that fails, call `func` with all\n    `args` and `kwargs` converted to their slow type.\n\n    Wrap the result in a fast-slow proxy if it is a type we know how\n    to wrap.\n    '
    from .module_accelerator import disable_module_accelerator
    fast = False
    try:
        with nvtx.annotate('EXECUTE_FAST', color=_CUDF_PANDAS_NVTX_COLORS['EXECUTE_FAST'], domain='cudf_pandas'):
            (fast_args, fast_kwargs) = (_fast_arg(args), _fast_arg(kwargs))
            result = func(*fast_args, **fast_kwargs)
            if result is NotImplemented:
                raise Exception()
            fast = True
    except Exception:
        with nvtx.annotate('EXECUTE_SLOW', color=_CUDF_PANDAS_NVTX_COLORS['EXECUTE_SLOW'], domain='cudf_pandas'):
            (slow_args, slow_kwargs) = (_slow_arg(args), _slow_arg(kwargs))
            with disable_module_accelerator():
                result = func(*slow_args, **slow_kwargs)
    return (_maybe_wrap_result(result, func, *args, **kwargs), fast)

def _transform_arg(arg: Any, attribute_name: Literal['_fsproxy_slow', '_fsproxy_fast'], seen: Set[int]) -> Any:
    if False:
        print('Hello World!')
    '\n    Transform "arg" into its corresponding slow (or fast) type.\n    '
    import numpy as np
    if isinstance(arg, (_FastSlowProxy, _FastSlowProxyMeta, _FunctionProxy)):
        typ = getattr(arg, attribute_name)
        if typ is _Unusable:
            raise Exception('Cannot transform _Unusable')
        return typ
    elif isinstance(arg, types.ModuleType) and attribute_name in arg.__dict__:
        return arg.__dict__[attribute_name]
    elif isinstance(arg, list):
        return type(arg)((_transform_arg(a, attribute_name, seen) for a in arg))
    elif isinstance(arg, tuple):
        if type(arg) is tuple:
            return tuple((_transform_arg(a, attribute_name, seen) for a in arg))
        elif hasattr(arg, '__getnewargs_ex__'):
            (args, kwargs) = (_transform_arg(a, attribute_name, seen) for a in arg.__getnewargs_ex__())
            obj = type(arg).__new__(type(arg), *args, **kwargs)
            if hasattr(obj, '__setstate__'):
                raise NotImplementedError('Transforming tuple-like with __getnewargs_ex__ and __setstate__ not implemented')
            if not hasattr(obj, '__dict__') and kwargs:
                raise NotImplementedError('Transforming tuple-like with kwargs from __getnewargs_ex__ and no __dict__ not implemented')
            obj.__dict__.update(kwargs)
            return obj
        elif hasattr(arg, '__getnewargs__'):
            args = _transform_arg(arg.__getnewargs__(), attribute_name, seen)
            return type(arg).__new__(type(arg), *args)
        else:
            return type(arg)((_transform_arg(a, attribute_name, seen) for a in args))
    elif isinstance(arg, dict):
        return {_transform_arg(k, attribute_name, seen): _transform_arg(a, attribute_name, seen) for (k, a) in arg.items()}
    elif isinstance(arg, np.ndarray) and arg.dtype == 'O':
        transformed = [_transform_arg(a, attribute_name, seen) for a in arg.flat]
        if arg.flags['F_CONTIGUOUS'] and (not arg.flags['C_CONTIGUOUS']):
            order = 'F'
        else:
            order = 'C'
        result = np.empty(int(np.prod(arg.shape)), dtype=object, order=order)
        result[...] = transformed
        return result.reshape(arg.shape)
    elif isinstance(arg, Iterator) and attribute_name == '_fsproxy_fast':
        raise Exception()
    elif isinstance(arg, types.FunctionType):
        if id(arg) in seen:
            return arg
        seen.add(id(arg))
        return _replace_closurevars(arg, attribute_name, seen)
    else:
        return arg

def _fast_arg(arg: Any) -> Any:
    if False:
        i = 10
        return i + 15
    '\n    Transform "arg" into its corresponding fast type.\n    '
    seen: Set[int] = set()
    return _transform_arg(arg, '_fsproxy_fast', seen)

def _slow_arg(arg: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform "arg" into its corresponding slow type.\n    '
    seen: Set[int] = set()
    return _transform_arg(arg, '_fsproxy_slow', seen)

def _maybe_wrap_result(result: Any, func: Callable, /, *args, **kwargs) -> Any:
    if False:
        print('Hello World!')
    '\n    Wraps "result" in a fast-slow proxy if is a "proxiable" object.\n    '
    if _is_final_type(result):
        typ = get_final_type_map()[type(result)]
        return typ._fsproxy_wrap(result, func)
    elif _is_intermediate_type(result):
        typ = get_intermediate_type_map()[type(result)]
        return typ._fsproxy_wrap(result, method_chain=(func, args, kwargs))
    elif _is_final_class(result):
        return get_final_type_map()[result]
    elif isinstance(result, list):
        return type(result)([_maybe_wrap_result(r, operator.getitem, result, i) for (i, r) in enumerate(result)])
    elif isinstance(result, tuple):
        wrapped = (_maybe_wrap_result(r, operator.getitem, result, i) for (i, r) in enumerate(result))
        if hasattr(result, '_make'):
            return type(result)._make(wrapped)
        else:
            return type(result)(wrapped)
    elif isinstance(result, Iterator):
        return (_maybe_wrap_result(r, lambda x: x, r) for r in result)
    elif _is_function_or_method(result):
        return _MethodProxy._fsproxy_wrap(result, method_chain=(func, args, kwargs))
    else:
        return result

def _is_final_type(result: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return type(result) in get_final_type_map()

def _is_final_class(result: Any) -> bool:
    if False:
        while True:
            i = 10
    if not isinstance(result, type):
        return False
    return result in get_final_type_map()

def _is_intermediate_type(result: Any) -> bool:
    if False:
        print('Hello World!')
    return type(result) in get_intermediate_type_map()

def _is_function_or_method(obj: Any) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.WrapperDescriptorType, types.MethodWrapperType, types.MethodDescriptorType, types.BuiltinMethodType))

def _replace_closurevars(f: types.FunctionType, attribute_name: Literal['_fsproxy_slow', '_fsproxy_fast'], seen: Set[int]) -> types.FunctionType:
    if False:
        return 10
    '\n    Return a copy of `f` with its closure variables replaced with\n    their corresponding slow (or fast) types.\n    '
    if f.__closure__:
        if any((c == types.CellType() for c in f.__closure__)):
            return f
    (f_nonlocals, f_globals, f_builtins, _) = inspect.getclosurevars(f)
    g_globals = _transform_arg(f_globals, attribute_name, seen)
    g_nonlocals = _transform_arg(f_nonlocals, attribute_name, seen)
    if all((f_globals[k] is g_globals[k] for k in f_globals)) and all((g_nonlocals[k] is f_nonlocals[k] for k in f_nonlocals)):
        return f
    g_closure = tuple((types.CellType(val) for val in g_nonlocals.values()))
    g_globals['__builtins__'] = f_builtins
    g = types.FunctionType(f.__code__, g_globals, name=f.__name__, argdefs=f.__defaults__, closure=g_closure)
    g = functools.update_wrapper(g, f, assigned=functools.WRAPPER_ASSIGNMENTS + ('__kwdefaults__',))
    return g
_SPECIAL_METHODS: Set[str] = {'__repr__', '__str__', '__len__', '__contains__', '__getitem__', '__setitem__', '__delitem__', '__getslice__', '__setslice__', '__delslice__', '__iter__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__pos__', '__neg__', '__invert__', '__abs__', '__round__', '__format__', '__bool__', '__float__', '__int__', '__complex__', '__enter__', '__exit__', '__next__', '__copy__', '__deepcopy__', '__dataframe__', '__call__'}