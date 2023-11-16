from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable, List, Any, Iterator, Dict, Tuple
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
__all__ = ['DataView', 'ListView', 'MapView']

class DataView(ABC):
    """
    A DataView is a collection type that can be used in the accumulator of an user defined
    :class:`pyflink.table.AggregateFunction`. Depending on the context in which the function
    is used, a DataView can be backed by a normal collection or a state backend.
    """

    @abstractmethod
    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Clears the DataView and removes all data.\n        '
        pass

class ListView(DataView, Generic[T]):
    """
    A :class:`DataView` that provides list-like functionality in the accumulator of an
    AggregateFunction when large amounts of data are expected.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._list = []

    def get(self) -> Iterable[T]:
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterable of this list view.\n        '
        return self._list

    def add(self, value: T) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds the given value to this list view.\n        '
        self._list.append(value)

    def add_all(self, values: List[T]) -> None:
        if False:
            return 10
        '\n        Adds all of the elements of the specified list to this list view.\n        '
        self._list.extend(values)

    def clear(self) -> None:
        if False:
            return 10
        self._list = []

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ListView):
            iter_obj = other.get()
            self_iterator = iter(self)
            for value in iter_obj:
                try:
                    self_value = next(self_iterator)
                except StopIteration:
                    return False
                if self_value != value:
                    return False
            try:
                next(self_iterator)
            except StopIteration:
                return True
            else:
                return False
        else:
            return False

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(self._list)

    def __iter__(self) -> Iterator[T]:
        if False:
            return 10
        return iter(self.get())

class MapView(Generic[K, V]):
    """
    A :class:`DataView` that provides dict-like functionality in the accumulator of an
    AggregateFunction when large amounts of data are expected.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._dict = dict()

    def get(self, key: K) -> V:
        if False:
            i = 10
            return i + 15
        '\n        Return the value for the specified key.\n        '
        return self._dict[key]

    def put(self, key: K, value: V) -> None:
        if False:
            while True:
                i = 10
        '\n        Inserts a value for the given key into the map view.\n        If the map view already contains a value for the key, the existing value is overwritten.\n        '
        self._dict[key] = value

    def put_all(self, dict_value: Dict[K, V]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Inserts all mappings from the specified map to this map view.\n        '
        self._dict.update(dict_value)

    def remove(self, key: K) -> None:
        if False:
            print('Hello World!')
        '\n        Deletes the value for the given key.\n        '
        del self._dict[key]

    def contains(self, key: K) -> bool:
        if False:
            return 10
        '\n        Checks if the map view contains a value for a given key.\n        '
        return key in self._dict

    def items(self) -> Iterable[Tuple[K, V]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns all entries of the map view.\n        '
        return self._dict.items()

    def keys(self) -> Iterable[K]:
        if False:
            print('Hello World!')
        '\n        Returns all the keys in the map view.\n        '
        return self._dict.keys()

    def values(self) -> Iterable[V]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns all the values in the map view.\n        '
        return self._dict.values()

    def is_empty(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns true if the map view contains no key-value mappings, otherwise false.\n        '
        return len(self._dict) == 0

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes all entries of this map.\n        '
        self._dict.clear()

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if other is None:
            return False
        if other.__class__ == MapView:
            return self._dict == other._dict
        else:
            return other is self

    def __getitem__(self, key: K) -> V:
        if False:
            return 10
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        if False:
            while True:
                i = 10
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        if False:
            return 10
        self.remove(key)

    def __contains__(self, key: K) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.contains(key)

    def __iter__(self) -> Iterator[K]:
        if False:
            print('Hello World!')
        return iter(self.keys())