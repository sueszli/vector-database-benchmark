"""
Custom Lazy Iterator class
"""
from __future__ import annotations
from abc import ABC, abstractmethod

class CustomIterator(ABC):
    """Lazy custom iteration and item access."""

    def __init__(self, obj):
        if False:
            print('Hello World!')
        self.obj = obj
        self._iter = 0

    @abstractmethod
    def __getitem__(self, key):
        if False:
            return 10
        'Get next item'
        pass

    def __repr__(self):
        if False:
            return 10
        return f'<{type(self.obj)}_iterator at {hex(id(self))}>'

    def __len__(self):
        if False:
            return 10
        return len(self.obj)

    def __iter__(self):
        if False:
            while True:
                i = 10
        self._iter = 0
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self._iter >= len(self):
            raise StopIteration
        self._iter += 1
        return self[self._iter - 1]