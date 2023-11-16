"""
Topic: 通过描述器定义新的实例属性
Desc : 
"""

class Integer:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name

    def __get__(self, instance, cls):
        if False:
            print('Hello World!')
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if False:
            return 10
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        if False:
            while True:
                i = 10
        del instance.__dict__[self.name]

class Point:
    x = Integer('x')
    y = Integer('y')

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y
p = Point(2, 3)
print(p.x)
p.y = 5

class Typed:

    def __init__(self, name, expected_type):
        if False:
            print('Hello World!')
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if False:
            for i in range(10):
                print('nop')
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if False:
            print('Hello World!')
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        if False:
            return 10
        del instance.__dict__[self.name]

def typeassert(**kwargs):
    if False:
        return 10

    def decorate(cls):
        if False:
            while True:
                i = 10
        for (name, expected_type) in kwargs.items():
            setattr(cls, name, Typed(name, expected_type))
        return cls
    return decorate

@typeassert(name=str, shares=int, price=float)
class Stock:

    def __init__(self, name, shares, price):
        if False:
            print('Hello World!')
        self.name = name
        self.shares = shares
        self.price = price