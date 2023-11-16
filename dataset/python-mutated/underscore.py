import copy
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
if TYPE_CHECKING:
    from .doc import Doc
    from .span import Span
    from .token import Token

class Underscore:
    mutable_types = (dict, list, set)
    doc_extensions: Dict[Any, Any] = {}
    span_extensions: Dict[Any, Any] = {}
    token_extensions: Dict[Any, Any] = {}
    _extensions: Dict[str, Any]
    _obj: Union['Doc', 'Span', 'Token']
    _start: Optional[int]
    _end: Optional[int]

    def __init__(self, extensions: Dict[str, Any], obj: Union['Doc', 'Span', 'Token'], start: Optional[int]=None, end: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        object.__setattr__(self, '_extensions', extensions)
        object.__setattr__(self, '_obj', obj)
        object.__setattr__(self, '_doc', obj.doc)
        object.__setattr__(self, '_start', start)
        object.__setattr__(self, '_end', end)

    def __dir__(self) -> List[str]:
        if False:
            return 10
        extensions = list(self._extensions.keys())
        return ['set', 'get', 'has'] + extensions

    def __getattr__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        if name not in self._extensions:
            raise AttributeError(Errors.E046.format(name=name))
        (default, method, getter, setter) = self._extensions[name]
        if getter is not None:
            return getter(self._obj)
        elif method is not None:
            method_partial = functools.partial(method, self._obj)
            method_docstring = method.__doc__ or ''
            method_docstring_prefix = "This method is a partial function and its first argument (the object it's called on) will be filled automatically. "
            method_partial.__doc__ = method_docstring_prefix + method_docstring
            return method_partial
        else:
            key = self._get_key(name)
            if key in self._doc.user_data:
                return self._doc.user_data[key]
            elif isinstance(default, self.mutable_types):
                new_default = copy.copy(default)
                self.__setattr__(name, new_default)
                return new_default
            return default

    def __setattr__(self, name: str, value: Any):
        if False:
            for i in range(10):
                print('nop')
        if name not in self._extensions:
            raise AttributeError(Errors.E047.format(name=name))
        (default, method, getter, setter) = self._extensions[name]
        if setter is not None:
            return setter(self._obj, value)
        else:
            self._doc.user_data[self._get_key(name)] = value

    def set(self, name: str, value: Any):
        if False:
            while True:
                i = 10
        return self.__setattr__(name, value)

    def get(self, name: str) -> Any:
        if False:
            return 10
        return self.__getattr__(name)

    def has(self, name: str) -> bool:
        if False:
            i = 10
            return i + 15
        return name in self._extensions

    def _get_key(self, name: str) -> Tuple[str, str, Optional[int], Optional[int]]:
        if False:
            print('Hello World!')
        return ('._.', name, self._start, self._end)

    @classmethod
    def get_state(cls) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
        if False:
            print('Hello World!')
        return (cls.token_extensions, cls.span_extensions, cls.doc_extensions)

    @classmethod
    def load_state(cls, state: Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]) -> None:
        if False:
            while True:
                i = 10
        (cls.token_extensions, cls.span_extensions, cls.doc_extensions) = state

def get_ext_args(**kwargs: Any):
    if False:
        print('Hello World!')
    'Validate and convert arguments. Reused in Doc, Token and Span.'
    default = kwargs.get('default')
    getter = kwargs.get('getter')
    setter = kwargs.get('setter')
    method = kwargs.get('method')
    if getter is None and setter is not None:
        raise ValueError(Errors.E089)
    valid_opts = ('default' in kwargs, method is not None, getter is not None)
    nr_defined = sum((t is True for t in valid_opts))
    if nr_defined != 1:
        raise ValueError(Errors.E083.format(nr_defined=nr_defined))
    if setter is not None and (not hasattr(setter, '__call__')):
        raise ValueError(Errors.E091.format(name='setter', value=repr(setter)))
    if getter is not None and (not hasattr(getter, '__call__')):
        raise ValueError(Errors.E091.format(name='getter', value=repr(getter)))
    if method is not None and (not hasattr(method, '__call__')):
        raise ValueError(Errors.E091.format(name='method', value=repr(method)))
    return (default, method, getter, setter)

def is_writable_attr(ext):
    if False:
        print('Hello World!')
    'Check if an extension attribute is writable.\n    ext (tuple): The (default, getter, setter, method) tuple available  via\n        {Doc,Span,Token}.get_extension.\n    RETURNS (bool): Whether the attribute is writable.\n    '
    (default, method, getter, setter) = ext
    if setter is not None or default is not None or all((e is None for e in ext)):
        return True
    return False