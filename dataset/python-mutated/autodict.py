from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict, defaultdict
from .py3 import values as py3lvalues

def Tree():
    if False:
        return 10
    return defaultdict(Tree)

class AutoDictList(dict):

    def __missing__(self, key):
        if False:
            return 10
        value = self[key] = list()
        return value

class DotDict(dict):

    def __getattr__(self, key):
        if False:
            while True:
                i = 10
        if key.startswith('__'):
            return super(DotDict, self).__getattr__(key)
        return self[key]

class AutoDict(dict):
    _closed = False

    def _close(self):
        if False:
            for i in range(10):
                print('nop')
        self._closed = True
        for (key, val) in self.items():
            if isinstance(val, (AutoDict, AutoOrderedDict)):
                val._close()

    def _open(self):
        if False:
            print('Hello World!')
        self._closed = False

    def __missing__(self, key):
        if False:
            return 10
        if self._closed:
            raise KeyError
        value = self[key] = AutoDict()
        return value

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15
        if False and key.startswith('_'):
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if False:
            print('Hello World!')
        if False and key.startswith('_'):
            self.__dict__[key] = value
            return
        self[key] = value

class AutoOrderedDict(OrderedDict):
    _closed = False

    def _close(self):
        if False:
            while True:
                i = 10
        self._closed = True
        for (key, val) in self.items():
            if isinstance(val, (AutoDict, AutoOrderedDict)):
                val._close()

    def _open(self):
        if False:
            return 10
        self._closed = False

    def __missing__(self, key):
        if False:
            i = 10
            return i + 15
        if self._closed:
            raise KeyError
        value = self[key] = AutoOrderedDict()
        return value

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15
        if key.startswith('_'):
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if False:
            print('Hello World!')
        if key.startswith('_'):
            self.__dict__[key] = value
            return
        self[key] = value

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        if type(self) != type(other):
            return type(other)() + other
        return self + other

    def __isub__(self, other):
        if False:
            while True:
                i = 10
        if type(self) != type(other):
            return type(other)() - other
        return self - other

    def __imul__(self, other):
        if False:
            return 10
        if type(self) != type(other):
            return type(other)() * other
        return self + other

    def __idiv__(self, other):
        if False:
            print('Hello World!')
        if type(self) != type(other):
            return type(other)() // other
        return self + other

    def __itruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if type(self) != type(other):
            return type(other)() / other
        return self + other

    def lvalues(self):
        if False:
            for i in range(10):
                print('nop')
        return py3lvalues(self)