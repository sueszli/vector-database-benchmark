"""Utilities for collecting objects based on "is" comparison."""
from typing import Any, Set
import weakref
from tensorflow.python.util.compat import collections_abc

class _ObjectIdentityWrapper:
    """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """
    __slots__ = ['_wrapped', '__weakref__']

    def __init__(self, wrapped):
        if False:
            while True:
                i = 10
        self._wrapped = wrapped

    @property
    def unwrapped(self):
        if False:
            return 10
        return self._wrapped

    def _assert_type(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, _ObjectIdentityWrapper):
            raise TypeError('Cannot compare wrapped object with unwrapped object')

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        self._assert_type(other)
        return id(self._wrapped) < id(other._wrapped)

    def __gt__(self, other):
        if False:
            print('Hello World!')
        self._assert_type(other)
        return id(self._wrapped) > id(other._wrapped)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other is None:
            return False
        self._assert_type(other)
        return self._wrapped is other._wrapped

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return id(self._wrapped)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{} wrapping {!r}>'.format(type(self).__name__, self._wrapped)

class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):
    __slots__ = ()

    def __init__(self, wrapped):
        if False:
            i = 10
            return i + 15
        super(_WeakObjectIdentityWrapper, self).__init__(weakref.ref(wrapped))

    @property
    def unwrapped(self):
        if False:
            for i in range(10):
                print('nop')
        return self._wrapped()

class Reference(_ObjectIdentityWrapper):
    """Reference that refers an object.

  ```python
  x = [1]
  y = [1]

  x_ref1 = Reference(x)
  x_ref2 = Reference(x)
  y_ref2 = Reference(y)

  print(x_ref1 == x_ref2)
  ==> True

  print(x_ref1 == y)
  ==> False
  ```
  """
    __slots__ = ()
    unwrapped = property()

    def deref(self):
        if False:
            while True:
                i = 10
        'Returns the referenced object.\n\n    ```python\n    x_ref = Reference(x)\n    print(x is x_ref.deref())\n    ==> True\n    ```\n    '
        return self._wrapped

class ObjectIdentityDictionary(collections_abc.MutableMapping):
    """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """
    __slots__ = ['_storage']

    def __init__(self):
        if False:
            print('Hello World!')
        self._storage = {}

    def _wrap_key(self, key):
        if False:
            i = 10
            return i + 15
        return _ObjectIdentityWrapper(key)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._storage[self._wrap_key(key)]

    def __setitem__(self, key, value):
        if False:
            return 10
        self._storage[self._wrap_key(key)] = value

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self._storage[self._wrap_key(key)]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._storage)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for key in self._storage:
            yield key.unwrapped

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ObjectIdentityDictionary(%s)' % repr(self._storage)

class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
    """Like weakref.WeakKeyDictionary, but compares objects with "is"."""
    __slots__ = ['__weakref__']

    def _wrap_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        return _WeakObjectIdentityWrapper(key)

    def __len__(self):
        if False:
            return 10
        return len(list(self._storage))

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        keys = self._storage.keys()
        for key in keys:
            unwrapped = key.unwrapped
            if unwrapped is None:
                del self[key]
            else:
                yield unwrapped

class ObjectIdentitySet(collections_abc.MutableSet):
    """Like the built-in set, but compares objects with "is"."""
    __slots__ = ['_storage', '__weakref__']

    def __init__(self, *args):
        if False:
            print('Hello World!')
        self._storage = set((self._wrap_key(obj) for obj in list(*args)))

    def __le__(self, other: Set[Any]) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) > len(other):
            return False
        for item in self._storage:
            if item not in other:
                return False
        return True

    def __ge__(self, other: Set[Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) < len(other):
            return False
        for item in other:
            if item not in self:
                return False
        return True

    @staticmethod
    def _from_storage(storage):
        if False:
            while True:
                i = 10
        result = ObjectIdentitySet()
        result._storage = storage
        return result

    def _wrap_key(self, key):
        if False:
            return 10
        return _ObjectIdentityWrapper(key)

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return self._wrap_key(key) in self._storage

    def discard(self, key):
        if False:
            while True:
                i = 10
        self._storage.discard(self._wrap_key(key))

    def add(self, key):
        if False:
            return 10
        self._storage.add(self._wrap_key(key))

    def update(self, items):
        if False:
            i = 10
            return i + 15
        self._storage.update([self._wrap_key(item) for item in items])

    def clear(self):
        if False:
            return 10
        self._storage.clear()

    def intersection(self, items):
        if False:
            while True:
                i = 10
        return self._storage.intersection([self._wrap_key(item) for item in items])

    def difference(self, items):
        if False:
            print('Hello World!')
        return ObjectIdentitySet._from_storage(self._storage.difference([self._wrap_key(item) for item in items]))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._storage)

    def __iter__(self):
        if False:
            while True:
                i = 10
        keys = list(self._storage)
        for key in keys:
            yield key.unwrapped

class ObjectIdentityWeakSet(ObjectIdentitySet):
    """Like weakref.WeakSet, but compares objects with "is"."""
    __slots__ = ()

    def _wrap_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        return _WeakObjectIdentityWrapper(key)

    def __len__(self):
        if False:
            print('Hello World!')
        return len([_ for _ in self])

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        keys = list(self._storage)
        for key in keys:
            unwrapped = key.unwrapped
            if unwrapped is None:
                self.discard(key)
            else:
                yield unwrapped