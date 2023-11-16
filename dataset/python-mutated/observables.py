from __future__ import annotations
import abc
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, SupportsIndex, Union
from . import events

class ObservableCollection(abc.ABC):

    def __init__(self, *, factory: Callable, data: Optional[Collection], on_change: Optional[Callable], _parent: Optional[ObservableCollection]) -> None:
        if False:
            while True:
                i = 10
        super().__init__(factory() if data is None else data)
        self._parent = _parent
        self._change_handlers: List[Callable] = [on_change] if on_change else []

    @property
    def change_handlers(self) -> List[Callable]:
        if False:
            while True:
                i = 10
        'Return a list of all change handlers registered on this collection and its parents.'
        change_handlers = self._change_handlers[:]
        if self._parent is not None:
            change_handlers.extend(self._parent.change_handlers)
        return change_handlers

    def _handle_change(self) -> None:
        if False:
            print('Hello World!')
        for handler in self.change_handlers:
            events.handle_event(handler, events.ObservableChangeEventArguments(sender=self))

    def on_change(self, handler: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register a handler to be called when the collection changes.'
        self._change_handlers.append(handler)

    def _observe(self, data: Any) -> Any:
        if False:
            print('Hello World!')
        if isinstance(data, dict):
            return ObservableDict(data, _parent=self)
        if isinstance(data, list):
            return ObservableList(data, _parent=self)
        if isinstance(data, set):
            return ObservableSet(data, _parent=self)
        return data

class ObservableDict(ObservableCollection, dict):

    def __init__(self, data: Dict=None, *, on_change: Optional[Callable]=None, _parent: Optional[ObservableCollection]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(factory=dict, data=data, on_change=on_change, _parent=_parent)
        for (key, value) in self.items():
            super().__setitem__(key, self._observe(value))

    def pop(self, k: Any, d: Any=None) -> Any:
        if False:
            print('Hello World!')
        item = super().pop(k, d)
        self._handle_change()
        return item

    def popitem(self) -> Any:
        if False:
            print('Hello World!')
        item = super().popitem()
        self._handle_change()
        return item

    def update(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().update(self._observe(dict(*args, **kwargs)))
        self._handle_change()

    def clear(self) -> None:
        if False:
            return 10
        super().clear()
        self._handle_change()

    def setdefault(self, __key: Any, __default: Any=None) -> Any:
        if False:
            i = 10
            return i + 15
        item = super().setdefault(__key, self._observe(__default))
        self._handle_change()
        return item

    def __setitem__(self, __key: Any, __value: Any) -> None:
        if False:
            print('Hello World!')
        super().__setitem__(__key, self._observe(__value))
        self._handle_change()

    def __delitem__(self, __key: Any) -> None:
        if False:
            while True:
                i = 10
        super().__delitem__(__key)
        self._handle_change()

    def __or__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return super().__or__(other)

    def __ior__(self, other: Any) -> Any:
        if False:
            return 10
        super().__ior__(self._observe(dict(other)))
        self._handle_change()
        return self

class ObservableList(ObservableCollection, list):

    def __init__(self, data: List=None, *, on_change: Optional[Callable]=None, _parent: Optional[ObservableCollection]=None) -> None:
        if False:
            return 10
        super().__init__(factory=list, data=data, on_change=on_change, _parent=_parent)
        for (i, item) in enumerate(self):
            super().__setitem__(i, self._observe(item))

    def append(self, item: Any) -> None:
        if False:
            while True:
                i = 10
        super().append(self._observe(item))
        self._handle_change()

    def extend(self, iterable: Iterable) -> None:
        if False:
            while True:
                i = 10
        super().extend(self._observe(list(iterable)))
        self._handle_change()

    def insert(self, index: SupportsIndex, obj: Any) -> None:
        if False:
            return 10
        super().insert(index, self._observe(obj))
        self._handle_change()

    def remove(self, value: Any) -> None:
        if False:
            print('Hello World!')
        super().remove(value)
        self._handle_change()

    def pop(self, index: SupportsIndex=-1) -> Any:
        if False:
            for i in range(10):
                print('nop')
        item = super().pop(index)
        self._handle_change()
        return item

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().clear()
        self._handle_change()

    def sort(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().sort(**kwargs)
        self._handle_change()

    def reverse(self) -> None:
        if False:
            i = 10
            return i + 15
        super().reverse()
        self._handle_change()

    def __delitem__(self, key: Union[SupportsIndex, slice]) -> None:
        if False:
            print('Hello World!')
        super().__delitem__(key)
        self._handle_change()

    def __setitem__(self, key: Union[SupportsIndex, slice], value: Any) -> None:
        if False:
            print('Hello World!')
        super().__setitem__(key, self._observe(value))
        self._handle_change()

    def __add__(self, other: Any) -> Any:
        if False:
            return 10
        return super().__add__(other)

    def __iadd__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        super().__iadd__(self._observe(other))
        self._handle_change()
        return self

class ObservableSet(ObservableCollection, set):

    def __init__(self, data: set=None, *, on_change: Optional[Callable]=None, _parent: Optional[ObservableCollection]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(factory=set, data=data, on_change=on_change, _parent=_parent)
        for item in self:
            super().add(self._observe(item))

    def add(self, item: Any) -> None:
        if False:
            return 10
        super().add(self._observe(item))
        self._handle_change()

    def remove(self, item: Any) -> None:
        if False:
            print('Hello World!')
        super().remove(item)
        self._handle_change()

    def discard(self, item: Any) -> None:
        if False:
            return 10
        super().discard(item)
        self._handle_change()

    def pop(self) -> Any:
        if False:
            while True:
                i = 10
        item = super().pop()
        self._handle_change()
        return item

    def clear(self) -> None:
        if False:
            return 10
        super().clear()
        self._handle_change()

    def update(self, *s: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        super().update(self._observe(set(*s)))
        self._handle_change()

    def intersection_update(self, *s: Iterable[Any]) -> None:
        if False:
            i = 10
            return i + 15
        super().intersection_update(*s)
        self._handle_change()

    def difference_update(self, *s: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().difference_update(*s)
        self._handle_change()

    def symmetric_difference_update(self, *s: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        super().symmetric_difference_update(*s)
        self._handle_change()

    def __or__(self, other: Any) -> Any:
        if False:
            print('Hello World!')
        return super().__or__(other)

    def __ior__(self, other: Any) -> Any:
        if False:
            return 10
        super().__ior__(self._observe(other))
        self._handle_change()
        return self

    def __and__(self, other: Any) -> set:
        if False:
            print('Hello World!')
        return super().__and__(other)

    def __iand__(self, other: Any) -> Any:
        if False:
            i = 10
            return i + 15
        super().__iand__(self._observe(other))
        self._handle_change()
        return self

    def __sub__(self, other: Any) -> set:
        if False:
            while True:
                i = 10
        return super().__sub__(other)

    def __isub__(self, other: Any) -> Any:
        if False:
            i = 10
            return i + 15
        super().__isub__(self._observe(other))
        self._handle_change()
        return self

    def __xor__(self, other: Any) -> set:
        if False:
            for i in range(10):
                print('nop')
        return super().__xor__(other)

    def __ixor__(self, other: Any) -> Any:
        if False:
            return 10
        super().__ixor__(self._observe(other))
        self._handle_change()
        return self