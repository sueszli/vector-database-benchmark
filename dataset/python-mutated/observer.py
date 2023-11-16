"""
Implements the Observer design pattern. Observers can be
notified when an object they observe (so-called Observable)
changes.

The implementation is modelled after the Java 8 specification
of Observable and Observer.

Observer references are weakrefs to prevent objects from being
ignored the garbage collection. Weakrefs with dead references
are removed during notification of the observers.
"""
from __future__ import annotations
from typing import Any, Optional
import weakref

class Observer:
    """
    Implements a Java 8-like Observer interface.
    """

    def update(self, observable: Observable, message: Optional[Any]=None):
        if False:
            while True:
                i = 10
        '\n        Called by an Observable object that has registered this observer\n        whenever it changes.\n\n        :param observable: The obvervable object which was updated.\n        :type observable: Observable\n        :param message: An optional message of any type.\n        '
        raise NotImplementedError(f'{self} has not implemented update()')

class Observable:
    """
    Implements a Java 8-like Observable object.
    """

    def __init__(self):
        if False:
            return 10
        self.observers: weakref.WeakSet[Observer] = weakref.WeakSet()
        self.changed = False

    def add_observer(self, observer: Observer) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Adds an observer to this object's set of observers.\n\n        :param observer: An observer observing this object.\n        :type observer: Observer\n        "
        self.observers.add(observer)

    def clear_changed(self) -> None:
        if False:
            return 10
        '\n        Indicate that this object has no longer changed.\n        '
        self.changed = False

    def delete_observer(self, observer: Observer) -> None:
        if False:
            while True:
                i = 10
        '\n        Remove an observer from the set.\n\n        :param observer: An observer observing this object.\n        :type observer: Observer\n        '
        self.observers.remove(observer)

    def delete_observers(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Remove all currently registered observers.\n        '
        self.observers.clear()

    def get_observer_count(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return the number of registered observers.\n        '
        return len(self.observers)

    def has_changed(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return whether the object has changed.\n        '
        return self.changed

    def notify_observers(self, message: Optional[Any]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Notify the observers if the object has changed. Include\n        an optional message.\n\n        :param message: An optional message of any type.\n        '
        if self.changed:
            for observer in self.observers:
                observer.update(self, message=message)

    def set_changed(self) -> None:
        if False:
            return 10
        '\n        Indicate that the object has changed.\n        '
        self.changed = True