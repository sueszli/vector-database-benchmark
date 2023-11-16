"""
Topic: 延迟属性
Desc : 
"""
import math

class lazyproperty:

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self.func = func

    def __get__(self, instance, cls):
        if False:
            i = 10
            return i + 15
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

class Circle:

    def __init__(self, radius):
        if False:
            for i in range(10):
                print('nop')
        self.radius = radius

    @lazyproperty
    def area(self):
        if False:
            for i in range(10):
                print('nop')
        print('Computing area')
        return math.pi * self.radius ** 2

    @lazyproperty
    def perimeter(self):
        if False:
            i = 10
            return i + 15
        print('Computing perimeter')
        return 2 * math.pi * self.radius
c = Circle(4.0)
print(vars(c))
print(c.area)
print(vars(c))
del c.area
print(vars(c))
print(c.area)

def lazyproperty2(func):
    if False:
        print('Hello World!')
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if False:
            while True:
                i = 10
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value
    return lazy