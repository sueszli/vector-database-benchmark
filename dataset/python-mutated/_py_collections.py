from __future__ import annotations
from itertools import filterfalse
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..util.typing import Self
_T = TypeVar('_T', bound=Any)
_S = TypeVar('_S', bound=Any)
_KT = TypeVar('_KT', bound=Any)
_VT = TypeVar('_VT', bound=Any)

class ReadOnlyContainer:
    __slots__ = ()

    def _readonly(self, *arg: Any, **kw: Any) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('%s object is immutable and/or readonly' % self.__class__.__name__)

    def _immutable(self, *arg: Any, **kw: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise TypeError('%s object is immutable' % self.__class__.__name__)

    def __delitem__(self, key: Any) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        self._readonly()

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        self._readonly()

    def __setattr__(self, key: str, value: Any) -> NoReturn:
        if False:
            return 10
        self._readonly()

class ImmutableDictBase(ReadOnlyContainer, Dict[_KT, _VT]):
    if TYPE_CHECKING:

        def __new__(cls, *args: Any) -> Self:
            if False:
                while True:
                    i = 10
            ...

        def __init__(cls, *args: Any):
            if False:
                return 10
            ...

    def _readonly(self, *arg: Any, **kw: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        self._immutable()

    def clear(self) -> NoReturn:
        if False:
            return 10
        self._readonly()

    def pop(self, key: Any, default: Optional[Any]=None) -> NoReturn:
        if False:
            while True:
                i = 10
        self._readonly()

    def popitem(self) -> NoReturn:
        if False:
            while True:
                i = 10
        self._readonly()

    def setdefault(self, key: Any, default: Optional[Any]=None) -> NoReturn:
        if False:
            i = 10
            return i + 15
        self._readonly()

    def update(self, *arg: Any, **kw: Any) -> NoReturn:
        if False:
            print('Hello World!')
        self._readonly()

class immutabledict(ImmutableDictBase[_KT, _VT]):

    def __new__(cls, *args):
        if False:
            return 10
        new = ImmutableDictBase.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args: Union[Mapping[_KT, _VT], Iterable[Tuple[_KT, _VT]]]):
        if False:
            return 10
        pass

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (immutabledict, (dict(self),))

    def union(self, __d: Optional[Mapping[_KT, _VT]]=None) -> immutabledict[_KT, _VT]:
        if False:
            while True:
                i = 10
        if not __d:
            return self
        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        dict.update(new, __d)
        return new

    def _union_w_kw(self, __d: Optional[Mapping[_KT, _VT]]=None, **kw: _VT) -> immutabledict[_KT, _VT]:
        if False:
            i = 10
            return i + 15
        if not __d and (not kw):
            return self
        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        if __d:
            dict.update(new, __d)
        dict.update(new, kw)
        return new

    def merge_with(self, *dicts: Optional[Mapping[_KT, _VT]]) -> immutabledict[_KT, _VT]:
        if False:
            return 10
        new = None
        for d in dicts:
            if d:
                if new is None:
                    new = ImmutableDictBase.__new__(self.__class__)
                    dict.__init__(new, self)
                dict.update(new, d)
        if new is None:
            return self
        return new

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'immutabledict(%s)' % dict.__repr__(self)

    def __ior__(self, __value: Any, /) -> NoReturn:
        if False:
            print('Hello World!')
        self._readonly()

    def __or__(self, __value: Mapping[_KT, _VT], /) -> immutabledict[_KT, _VT]:
        if False:
            i = 10
            return i + 15
        return immutabledict(super().__or__(__value))

    def __ror__(self, __value: Mapping[_KT, _VT], /) -> immutabledict[_KT, _VT]:
        if False:
            while True:
                i = 10
        return immutabledict(super().__ror__(__value))

class OrderedSet(Set[_T]):
    __slots__ = ('_list',)
    _list: List[_T]

    def __init__(self, d: Optional[Iterable[_T]]=None) -> None:
        if False:
            i = 10
            return i + 15
        if d is not None:
            self._list = unique_list(d)
            super().update(self._list)
        else:
            self._list = []

    def copy(self) -> OrderedSet[_T]:
        if False:
            while True:
                i = 10
        cp = self.__class__()
        cp._list = self._list.copy()
        set.update(cp, cp._list)
        return cp

    def add(self, element: _T) -> None:
        if False:
            return 10
        if element not in self:
            self._list.append(element)
        super().add(element)

    def remove(self, element: _T) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().remove(element)
        self._list.remove(element)

    def pop(self) -> _T:
        if False:
            while True:
                i = 10
        try:
            value = self._list.pop()
        except IndexError:
            raise KeyError('pop from an empty set') from None
        super().remove(value)
        return value

    def insert(self, pos: int, element: _T) -> None:
        if False:
            while True:
                i = 10
        if element not in self:
            self._list.insert(pos, element)
        super().add(element)

    def discard(self, element: _T) -> None:
        if False:
            while True:
                i = 10
        if element in self:
            self._list.remove(element)
            super().remove(element)

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        super().clear()
        self._list = []

    def __getitem__(self, key: int) -> _T:
        if False:
            for i in range(10):
                print('nop')
        return self._list[key]

    def __iter__(self) -> Iterator[_T]:
        if False:
            while True:
                i = 10
        return iter(self._list)

    def __add__(self, other: Iterator[_T]) -> OrderedSet[_T]:
        if False:
            i = 10
            return i + 15
        return self.union(other)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '%s(%r)' % (self.__class__.__name__, self._list)
    __str__ = __repr__

    def update(self, *iterables: Iterable[_T]) -> None:
        if False:
            print('Hello World!')
        for iterable in iterables:
            for e in iterable:
                if e not in self:
                    self._list.append(e)
                    super().add(e)

    def __ior__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        if False:
            while True:
                i = 10
        self.update(other)
        return self

    def union(self, *other: Iterable[_S]) -> OrderedSet[Union[_T, _S]]:
        if False:
            return 10
        result: OrderedSet[Union[_T, _S]] = self.copy()
        result.update(*other)
        return result

    def __or__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        if False:
            while True:
                i = 10
        return self.union(other)

    def intersection(self, *other: Iterable[Any]) -> OrderedSet[_T]:
        if False:
            i = 10
            return i + 15
        other_set: Set[Any] = set()
        other_set.update(*other)
        return self.__class__((a for a in self if a in other_set))

    def __and__(self, other: AbstractSet[object]) -> OrderedSet[_T]:
        if False:
            for i in range(10):
                print('nop')
        return self.intersection(other)

    def symmetric_difference(self, other: Iterable[_T]) -> OrderedSet[_T]:
        if False:
            return 10
        collection: Collection[_T]
        if isinstance(other, set):
            collection = other_set = other
        elif isinstance(other, Collection):
            collection = other
            other_set = set(other)
        else:
            collection = list(other)
            other_set = set(collection)
        result = self.__class__((a for a in self if a not in other_set))
        result.update((a for a in collection if a not in self))
        return result

    def __xor__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        if False:
            for i in range(10):
                print('nop')
        return cast(OrderedSet[Union[_T, _S]], self).symmetric_difference(other)

    def difference(self, *other: Iterable[Any]) -> OrderedSet[_T]:
        if False:
            for i in range(10):
                print('nop')
        other_set = super().difference(*other)
        return self.__class__((a for a in self._list if a in other_set))

    def __sub__(self, other: AbstractSet[Optional[_T]]) -> OrderedSet[_T]:
        if False:
            while True:
                i = 10
        return self.difference(other)

    def intersection_update(self, *other: Iterable[Any]) -> None:
        if False:
            i = 10
            return i + 15
        super().intersection_update(*other)
        self._list = [a for a in self._list if a in self]

    def __iand__(self, other: AbstractSet[object]) -> OrderedSet[_T]:
        if False:
            for i in range(10):
                print('nop')
        self.intersection_update(other)
        return self

    def symmetric_difference_update(self, other: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        collection = other if isinstance(other, Collection) else list(other)
        super().symmetric_difference_update(collection)
        self._list = [a for a in self._list if a in self]
        self._list += [a for a in collection if a in self]

    def __ixor__(self, other: AbstractSet[_S]) -> OrderedSet[Union[_T, _S]]:
        if False:
            for i in range(10):
                print('nop')
        self.symmetric_difference_update(other)
        return cast(OrderedSet[Union[_T, _S]], self)

    def difference_update(self, *other: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().difference_update(*other)
        self._list = [a for a in self._list if a in self]

    def __isub__(self, other: AbstractSet[Optional[_T]]) -> OrderedSet[_T]:
        if False:
            for i in range(10):
                print('nop')
        self.difference_update(other)
        return self

class IdentitySet:
    """A set that considers only object id() for uniqueness.

    This strategy has edge cases for builtin types- it's possible to have
    two 'foo' strings in one of these sets, for example.  Use sparingly.

    """
    _members: Dict[int, Any]

    def __init__(self, iterable: Optional[Iterable[Any]]=None):
        if False:
            return 10
        self._members = dict()
        if iterable:
            self.update(iterable)

    def add(self, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._members[id(value)] = value

    def __contains__(self, value: Any) -> bool:
        if False:
            return 10
        return id(value) in self._members

    def remove(self, value: Any) -> None:
        if False:
            return 10
        del self._members[id(value)]

    def discard(self, value: Any) -> None:
        if False:
            return 10
        try:
            self.remove(value)
        except KeyError:
            pass

    def pop(self) -> Any:
        if False:
            print('Hello World!')
        try:
            pair = self._members.popitem()
            return pair[1]
        except KeyError:
            raise KeyError('pop from an empty set')

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        self._members.clear()

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, IdentitySet):
            return self._members == other._members
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        if False:
            return 10
        if isinstance(other, IdentitySet):
            return self._members != other._members
        else:
            return True

    def issubset(self, iterable: Iterable[Any]) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)
        if len(self) > len(other):
            return False
        for m in filterfalse(other._members.__contains__, iter(self._members.keys())):
            return False
        return True

    def __le__(self, other: Any) -> bool:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issubset(other)

    def __lt__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) < len(other) and self.issubset(other)

    def issuperset(self, iterable: Iterable[Any]) -> bool:
        if False:
            return 10
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)
        if len(self) < len(other):
            return False
        for m in filterfalse(self._members.__contains__, iter(other._members.keys())):
            return False
        return True

    def __ge__(self, other: Any) -> bool:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issuperset(other)

    def __gt__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) > len(other) and self.issuperset(other)

    def union(self, iterable: Iterable[Any]) -> IdentitySet:
        if False:
            for i in range(10):
                print('nop')
        result = self.__class__()
        members = self._members
        result._members.update(members)
        result._members.update(((id(obj), obj) for obj in iterable))
        return result

    def __or__(self, other: Any) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.union(other)

    def update(self, iterable: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        self._members.update(((id(obj), obj) for obj in iterable))

    def __ior__(self, other: Any) -> IdentitySet:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.update(other)
        return self

    def difference(self, iterable: Iterable[Any]) -> IdentitySet:
        if False:
            i = 10
            return i + 15
        result = self.__new__(self.__class__)
        other: Collection[Any]
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {k: v for (k, v) in self._members.items() if k not in other}
        return result

    def __sub__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.difference(other)

    def difference_update(self, iterable: Iterable[Any]) -> None:
        if False:
            i = 10
            return i + 15
        self._members = self.difference(iterable)._members

    def __isub__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.difference_update(other)
        return self

    def intersection(self, iterable: Iterable[Any]) -> IdentitySet:
        if False:
            print('Hello World!')
        result = self.__new__(self.__class__)
        other: Collection[Any]
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {k: v for (k, v) in self._members.items() if k in other}
        return result

    def __and__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.intersection(other)

    def intersection_update(self, iterable: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._members = self.intersection(iterable)._members

    def __iand__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.intersection_update(other)
        return self

    def symmetric_difference(self, iterable: Iterable[Any]) -> IdentitySet:
        if False:
            i = 10
            return i + 15
        result = self.__new__(self.__class__)
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj): obj for obj in iterable}
        result._members = {k: v for (k, v) in self._members.items() if k not in other}
        result._members.update(((k, v) for (k, v) in other.items() if k not in self._members))
        return result

    def __xor__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, iterable: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        self._members = self.symmetric_difference(iterable)._members

    def __ixor__(self, other: IdentitySet) -> IdentitySet:
        if False:
            return 10
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.symmetric_difference(other)
        return self

    def copy(self) -> IdentitySet:
        if False:
            while True:
                i = 10
        result = self.__new__(self.__class__)
        result._members = self._members.copy()
        return result
    __copy__ = copy

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._members)

    def __iter__(self) -> Iterator[Any]:
        if False:
            print('Hello World!')
        return iter(self._members.values())

    def __hash__(self) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('set objects are unhashable')

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '%s(%r)' % (type(self).__name__, list(self._members.values()))

def unique_list(seq: Iterable[_T], hashfunc: Optional[Callable[[_T], int]]=None) -> List[_T]:
    if False:
        i = 10
        return i + 15
    seen: Set[Any] = set()
    seen_add = seen.add
    if not hashfunc:
        return [x for x in seq if x not in seen and (not seen_add(x))]
    else:
        return [x for x in seq if hashfunc(x) not in seen and (not seen_add(hashfunc(x)))]