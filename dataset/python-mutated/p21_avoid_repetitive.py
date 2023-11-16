"""
Topic: 避免重复的属性方法
Desc : 
"""

class Person:

    def __init__(self, name, age):
        if False:
            while True:
                i = 10
        self.name = name
        self.age = age

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    @name.setter
    def name(self, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, str):
            raise TypeError('name must be a string')
        self._name = value

    @property
    def age(self):
        if False:
            while True:
                i = 10
        return self._age

    @age.setter
    def age(self, value):
        if False:
            print('Hello World!')
        if not isinstance(value, int):
            raise TypeError('age must be an int')
        self._age = value

def typed_property(name, expected_type):
    if False:
        i = 10
        return i + 15
    storage_name = '_' + name

    @property
    def prop(self):
        if False:
            while True:
                i = 10
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)
    return prop

class Person:
    name = typed_property('name', str)
    age = typed_property('age', int)

    def __init__(self, name, age):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.age = age
from functools import partial
String = partial(typed_property, expected_type=str)
Integer = partial(typed_property, expected_type=int)

class Person:
    name = String('name')
    age = Integer('age')

    def __init__(self, name, age):
        if False:
            print('Hello World!')
        self.name = name
        self.age = age