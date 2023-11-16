"""
Topic: 可管理的属性
Desc : 
"""
import math

class Person:

    def __init__(self, first_name):
        if False:
            while True:
                i = 10
        self.first_name = first_name

    @property
    def first_name(self):
        if False:
            print('Hello World!')
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    @first_name.deleter
    def first_name(self):
        if False:
            print('Hello World!')
        raise AttributeError("Can't delete attribute")
a = Person('Guido')
print(a.first_name)

class Person1:

    def __init__(self, first_name):
        if False:
            return 10
        self.set_first_name(first_name)

    def get_first_name(self):
        if False:
            while True:
                i = 10
        return self._first_name

    def set_first_name(self, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    def del_first_name(self):
        if False:
            i = 10
            return i + 15
        raise AttributeError("Can't delete attribute")
    name = property(get_first_name, set_first_name, del_first_name)
print(Person1.name.fget)
print(Person1.name.fset)
print(Person1.name.fdel)

class Circle:
    """动态计算的property"""

    def __init__(self, radius):
        if False:
            print('Hello World!')
        self.radius = radius

    @property
    def diameter(self):
        if False:
            for i in range(10):
                print('nop')
        return self.radius ** 2

    @property
    def perimeter(self):
        if False:
            while True:
                i = 10
        return 2 * math.pi * self.radius

    @property
    def area(self):
        if False:
            for i in range(10):
                print('nop')
        return math.pi * self.radius ** 2
c = Circle(4.0)
print(c.radius)
print(c.area)