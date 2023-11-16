from functools import total_ordering
from typing import Any, Iterable, Iterator, MutableSequence, overload, TYPE_CHECKING, Type, TypeVar
from robot.utils import copy_signature, KnownAtRuntime, type_name
from .modelobject import DataDict
if TYPE_CHECKING:
    from .visitor import SuiteVisitor
T = TypeVar('T')
Self = TypeVar('Self', bound='ItemList')

@total_ordering
class ItemList(MutableSequence[T]):
    """List of items of a certain enforced type.

    New items can be created using the :meth:`create` method and existing items
    added using the common list methods like :meth:`append` or :meth:`insert`.
    In addition to the common type, items can have certain common and
    automatically assigned attributes.

    Starting from Robot Framework 6.1, items can be added as dictionaries and
    actual items are generated based on them automatically. If the type has
    a ``from_dict`` class method, it is used, and otherwise dictionary data is
    passed to the type as keyword arguments.
    """
    __slots__ = ['_item_class', '_common_attrs', '_items']
    item_type: Type[T] = KnownAtRuntime

    def __init__(self, item_class: Type[T], common_attrs: 'dict[str, Any]|None'=None, items: 'Iterable[T|DataDict]'=()):
        if False:
            return 10
        self._item_class = item_class
        self._common_attrs = common_attrs
        self._items: 'list[T]' = []
        if items:
            self.extend(items)

    @copy_signature(item_type)
    def create(self, *args, **kwargs) -> T:
        if False:
            i = 10
            return i + 15
        'Create a new item using the provided arguments.'
        return self.append(self._item_class(*args, **kwargs))

    def append(self, item: 'T|DataDict') -> T:
        if False:
            while True:
                i = 10
        item = self._check_type_and_set_attrs(item)
        self._items.append(item)
        return item

    def _check_type_and_set_attrs(self, item: 'T|DataDict') -> T:
        if False:
            return 10
        if not isinstance(item, self._item_class):
            if isinstance(item, dict):
                item = self._item_from_dict(item)
            else:
                raise TypeError(f'Only {type_name(self._item_class)} objects accepted, got {type_name(item)}.')
        if self._common_attrs:
            for (attr, value) in self._common_attrs.items():
                setattr(item, attr, value)
        return item

    def _item_from_dict(self, data: DataDict) -> T:
        if False:
            i = 10
            return i + 15
        if hasattr(self._item_class, 'from_dict'):
            return self._item_class.from_dict(data)
        return self._item_class(**data)

    def extend(self, items: 'Iterable[T|DataDict]'):
        if False:
            print('Hello World!')
        self._items.extend((self._check_type_and_set_attrs(i) for i in items))

    def insert(self, index: int, item: 'T|DataDict'):
        if False:
            for i in range(10):
                print('nop')
        item = self._check_type_and_set_attrs(item)
        self._items.insert(index, item)

    def index(self, item: T, *start_and_end) -> int:
        if False:
            print('Hello World!')
        return self._items.index(item, *start_and_end)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._items = []

    def visit(self, visitor: 'SuiteVisitor'):
        if False:
            print('Hello World!')
        for item in self:
            item.visit(visitor)

    def __iter__(self) -> Iterator[T]:
        if False:
            return 10
        index = 0
        while index < len(self._items):
            yield self._items[index]
            index += 1

    @overload
    def __getitem__(self, index: int) -> T:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __getitem__(self: Self, index: slice) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if isinstance(index, slice):
            return self._create_new_from(self._items[index])
        return self._items[index]

    def _create_new_from(self: Self, items: Iterable[T]) -> Self:
        if False:
            i = 10
            return i + 15
        new = type(self)(self._item_class)
        new._common_attrs = self._common_attrs
        new.extend(items)
        return new

    @overload
    def __setitem__(self, index: int, item: 'T|DataDict'):
        if False:
            while True:
                i = 10
        ...

    @overload
    def __setitem__(self, index: slice, item: 'Iterable[T|DataDict]'):
        if False:
            return 10
        ...

    def __setitem__(self, index, item):
        if False:
            while True:
                i = 10
        if isinstance(index, slice):
            self._items[index] = [self._check_type_and_set_attrs(i) for i in item]
        else:
            self._items[index] = self._check_type_and_set_attrs(item)

    def __delitem__(self, index: 'int|slice'):
        if False:
            while True:
                i = 10
        del self._items[index]

    def __contains__(self, item: object) -> bool:
        if False:
            return 10
        return item in self._items

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._items)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(list(self))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        class_name = type(self).__name__
        item_name = self._item_class.__name__
        return f'{class_name}(item_class={item_name}, items={self._items})'

    def count(self, item: T) -> int:
        if False:
            while True:
                i = 10
        return self._items.count(item)

    def sort(self, **config):
        if False:
            for i in range(10):
                print('nop')
        self._items.sort(**config)

    def reverse(self):
        if False:
            while True:
                i = 10
        self._items.reverse()

    def __reversed__(self) -> Iterator[T]:
        if False:
            print('Hello World!')
        index = 0
        while index < len(self._items):
            yield self._items[len(self._items) - index - 1]
            index += 1

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, ItemList) and self._is_compatible(other) and (self._items == other._items)

    def _is_compatible(self, other) -> bool:
        if False:
            return 10
        return self._item_class is other._item_class and self._common_attrs == other._common_attrs

    def __lt__(self, other: 'ItemList[T]') -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, ItemList):
            raise TypeError(f'Cannot order ItemList and {type_name(other)}.')
        if not self._is_compatible(other):
            raise TypeError('Cannot order incompatible ItemLists.')
        return self._items < other._items

    def __add__(self: Self, other: 'ItemList[T]') -> Self:
        if False:
            while True:
                i = 10
        if not isinstance(other, ItemList):
            raise TypeError(f'Cannot add ItemList and {type_name(other)}.')
        if not self._is_compatible(other):
            raise TypeError('Cannot add incompatible ItemLists.')
        return self._create_new_from(self._items + other._items)

    def __iadd__(self: Self, other: Iterable[T]) -> Self:
        if False:
            return 10
        if isinstance(other, ItemList) and (not self._is_compatible(other)):
            raise TypeError('Cannot add incompatible ItemLists.')
        self.extend(other)
        return self

    def __mul__(self: Self, count: int) -> Self:
        if False:
            return 10
        return self._create_new_from(self._items * count)

    def __imul__(self: Self, count: int) -> Self:
        if False:
            return 10
        self._items *= count
        return self

    def __rmul__(self: Self, count: int) -> Self:
        if False:
            while True:
                i = 10
        return self * count

    def to_dicts(self) -> 'list[DataDict]':
        if False:
            while True:
                i = 10
        'Return list of items converted to dictionaries.\n\n        Items are converted to dictionaries using the ``to_dict`` method, if\n        they have it, or the built-in ``vars()``.\n\n        New in Robot Framework 6.1.\n        '
        if not hasattr(self._item_class, 'to_dict'):
            return [vars(item) for item in self]
        return [item.to_dict() for item in self]