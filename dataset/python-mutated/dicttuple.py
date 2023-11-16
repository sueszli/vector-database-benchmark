from collections import OrderedDict
from collections import deque
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable
from typing import Generic
from typing import Optional
from typing import SupportsIndex
from typing import TypeVar
from typing import Union
from typing import overload
from typing_extensions import Self
_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class _Meta(type):

    @overload
    def __call__(cls: type[_T]) -> _T:
        if False:
            print('Hello World!')
        ...

    @overload
    def __call__(cls: type[_T], __value: Iterable[tuple[_KT, _VT]]) -> _T:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __call__(cls: type[_T], __value: Mapping[_KT, _VT]) -> _T:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __call__(cls: type[_T], __value: Iterable[_VT], __key: Collection[_KT]) -> _T:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __call__(cls: type[_T], __value: Iterable[_VT], __key: Callable[[_VT], _KT]) -> _T:
        if False:
            while True:
                i = 10
        ...

    def __call__(cls: type[_T], __value: Optional[Iterable]=None, __key: Optional[Union[Callable, Collection]]=None, /) -> _T:
        if False:
            for i in range(10):
                print('nop')
        if __value is None and __key is None:
            obj = cls.__new__(cls)
            obj.__init__()
            return obj
        elif type(__value) is cls:
            return __value
        elif isinstance(__value, Mapping) and __key is None:
            obj = cls.__new__(cls, __value.values())
            obj.__init__(__value.keys())
            return obj
        elif hasattr(__value, 'items') and callable(__value.items):
            return cls.__call__(__value.items())
        elif isinstance(__value, Iterable) and __key is None:
            keys = OrderedDict()
            values = deque()
            for (i, (k, v)) in enumerate(__value):
                keys[k] = i
                values.append(v)
            obj = cls.__new__(cls, values)
            obj.__init__(keys)
            return obj
        elif isinstance(__value, Iterable) and isinstance(__key, Iterable):
            keys = OrderedDict(((k, i) for (i, k) in enumerate(__key)))
            obj = cls.__new__(cls, __value)
            obj.__init__(keys)
            return obj
        elif isinstance(__value, Iterable) and isinstance(__key, Callable):
            obj = cls.__new__(cls, __value)
            obj.__init__(__key)
            return obj
        raise NotImplementedError

class DictTuple(tuple[_VT, ...], Generic[_KT, _VT], metaclass=_Meta):
    """
    OVERVIEW

        tuple with support for dict-like __getitem__(key)

            dict_tuple = DictTuple({"x": 1, "y": 2})

            dict_tuple["x"] == 1

            dict_tuple["y"] == 2

            dict_tuple[0] == 1

            dict_tuple[1] == 2

        everything else, e.g. __contains__, __iter__, behaves similarly to a tuple


    CREATION

        DictTuple(iterable) -> DictTuple([("x", 1), ("y", 2)])

        DictTuple(mapping) -> DictTuple({"x": 1, "y": 2})

        DictTuple(values, keys) -> DictTuple([1, 2], ["x", "y"])


    IMPLEMENTATION DETAILS

        DictTuple[_KT, _VT] is essentially a tuple[_VT, ...] that maintains an immutable Mapping[_KT, int]
        from the key to the tuple index internally.

        For example DictTuple({"x": 12, "y": 34}) is just a tuple (12, 34) with a {"x": 0, "y": 1} mapping.

        types.MappingProxyType is used for the mapping for immutability.
    """
    __mapping: MappingProxyType[_KT, int]

    @overload
    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self, __value: Iterable[tuple[_KT, _VT]]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __init__(self, __value: Mapping[_KT, _VT]) -> None:
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self, __value: Iterable[_VT], __key: Collection[_KT]) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self, __value: Iterable[_VT], __key: Callable[[_VT], _KT]) -> None:
        if False:
            while True:
                i = 10
        ...

    def __init__(self, __value=None, /):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(__value, MappingProxyType):
            self.__mapping = __value
        elif isinstance(__value, Mapping):
            self.__mapping = MappingProxyType(__value)
        elif isinstance(__value, Iterable):
            self.__mapping = MappingProxyType(OrderedDict(((k, i) for (i, k) in enumerate(__value))))
        elif isinstance(__value, Callable):
            self.__mapping = MappingProxyType(OrderedDict(((__value(v), i) for (i, v) in enumerate(self))))
        super().__init__()
        if len(self.__mapping) != len(self):
            raise ValueError('`__keys` and `__values` do not have the same length')
        if any((isinstance(k, SupportsIndex) for k in self.__mapping.keys())):
            raise ValueError('values of `__keys` should not have type `int`, or implement `__index__()`')

    @overload
    def __getitem__(self, __key: _KT) -> _VT:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __getitem__(self, __key: slice) -> Self:
        if False:
            return 10
        ...

    @overload
    def __getitem__(self, __key: SupportsIndex) -> _VT:
        if False:
            i = 10
            return i + 15
        ...

    def __getitem__(self, __key, /):
        if False:
            while True:
                i = 10
        if isinstance(__key, slice):
            return self.__class__(super().__getitem__(__key), list(self.__mapping.keys()).__getitem__(__key))
        if isinstance(__key, SupportsIndex):
            return super().__getitem__(__key)
        return super().__getitem__(self.__mapping[__key])

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__qualname__}{super().__repr__()}'

    def keys(self) -> KeysView[_KT]:
        if False:
            i = 10
            return i + 15
        return self.__mapping.keys()

    def items(self) -> Iterable[tuple[_KT, _VT]]:
        if False:
            while True:
                i = 10
        return zip(self.__mapping.keys(), self)