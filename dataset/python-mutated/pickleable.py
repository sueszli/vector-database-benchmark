"""Classes used in pickling tests, need to be at the module level for
unpickling.
"""
from __future__ import annotations
from .entities import ComparableEntity
from ..schema import Column
from ..types import String

class User(ComparableEntity):
    pass

class Order(ComparableEntity):
    pass

class Dingaling(ComparableEntity):
    pass

class EmailUser(User):
    pass

class Address(ComparableEntity):
    pass

class Child1(ComparableEntity):
    pass

class Child2(ComparableEntity):
    pass

class Parent(ComparableEntity):
    pass

class Screen:

    def __init__(self, obj, parent=None):
        if False:
            i = 10
            return i + 15
        self.obj = obj
        self.parent = parent

class Mixin:
    email_address = Column(String)

class AddressWMixin(Mixin, ComparableEntity):
    pass

class Foo:

    def __init__(self, moredata, stuff='im stuff'):
        if False:
            for i in range(10):
                print('nop')
        self.data = 'im data'
        self.stuff = stuff
        self.moredata = moredata
    __hash__ = object.__hash__

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return other.data == self.data and other.stuff == self.stuff and (other.moredata == self.moredata)

class Bar:

    def __init__(self, x, y):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y
    __hash__ = object.__hash__

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return other.__class__ is self.__class__ and other.x == self.x and (other.y == self.y)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'Bar(%d, %d)' % (self.x, self.y)

class OldSchool:

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return other.__class__ is self.__class__ and other.x == self.x and (other.y == self.y)

class OldSchoolWithoutCompare:

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y

class BarWithoutCompare:

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Bar(%d, %d)' % (self.x, self.y)

class NotComparable:

    def __init__(self, data):
        if False:
            return 10
        self.data = data

    def __hash__(self):
        if False:
            print('Hello World!')
        return id(self)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return NotImplemented

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

class BrokenComparable:

    def __init__(self, data):
        if False:
            return 10
        self.data = data

    def __hash__(self):
        if False:
            return 10
        return id(self)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError