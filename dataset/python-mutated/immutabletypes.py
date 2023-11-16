"""
    :codeauthor: Pedro Algarvio (pedro@algarvio.me)


    salt.utils.immutabletypes
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Immutable types
"""
import copy
from collections.abc import Mapping, Sequence, Set

class ImmutableDict(Mapping):
    """
    An immutable dictionary implementation
    """

    def __init__(self, obj):
        if False:
            return 10
        self.__obj = obj

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.__obj)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.__obj)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return freeze(self.__obj[key])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<{} {}>'.format(self.__class__.__name__, repr(self.__obj))

    def __deepcopy__(self, memo):
        if False:
            while True:
                i = 10
        return copy.deepcopy(self.__obj)

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        Return an un-frozen copy of self\n        '
        return copy.deepcopy(self.__obj)

class ImmutableList(Sequence):
    """
    An immutable list implementation
    """

    def __init__(self, obj):
        if False:
            print('Hello World!')
        self.__obj = obj

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.__obj)

    def __iter__(self):
        if False:
            return 10
        return iter(self.__obj)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self.__obj + other

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return other + self.__obj

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return freeze(self.__obj[key])

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{} {}>'.format(self.__class__.__name__, repr(self.__obj))

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        return copy.deepcopy(self.__obj)

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Return an un-frozen copy of self\n        '
        return copy.deepcopy(self.__obj)

class ImmutableSet(Set):
    """
    An immutable set implementation
    """

    def __init__(self, obj):
        if False:
            return 10
        self.__obj = obj

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.__obj)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.__obj)

    def __contains__(self, key):
        if False:
            return 10
        return key in self.__obj

    def __repr__(self):
        if False:
            return 10
        return '<{} {}>'.format(self.__class__.__name__, repr(self.__obj))

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        return copy.deepcopy(self.__obj)

    def copy(self):
        if False:
            return 10
        '\n        Return an un-frozen copy of self\n        '
        return copy.deepcopy(self.__obj)

def freeze(obj):
    if False:
        i = 10
        return i + 15
    '\n    Freeze python types by turning them into immutable structures.\n    '
    if isinstance(obj, dict):
        return ImmutableDict(obj)
    if isinstance(obj, list):
        return ImmutableList(obj)
    if isinstance(obj, set):
        return ImmutableSet(obj)
    return obj