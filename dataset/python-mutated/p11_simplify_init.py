"""
Topic: 简化数据结构的初始化
Desc : 
"""
import math

class Structure1:
    _fields = []

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        for (name, value) in zip(self._fields, args):
            setattr(self, name, value)
if __name__ == '__main__':

    class Stock(Structure1):
        _fields = ['name', 'shares', 'price']

    class Point(Structure1):
        _fields = ['x', 'y']

    class Circle(Structure1):
        _fields = ['radius']

        def area(self):
            if False:
                return 10
            return math.pi * self.radius ** 2
    s = Stock('ACME', 50, 91.1)
    p = Point(2, 3)
    c = Circle(4.5)

class Structure2:
    _fields = []

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        for (name, value) in zip(self._fields, args):
            setattr(self, name, value)
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))
if __name__ == '__main__':

    class Stock(Structure2):
        _fields = ['name', 'shares', 'price']
    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, price=91.1)
    s3 = Stock('ACME', shares=50, price=91.1)

class Structure3:
    _fields = []

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        for (name, value) in zip(self._fields, args):
            setattr(self, name, value)
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError('Duplicate values for {}'.format(','.join(kwargs)))
if __name__ == '__main__':

    class Stock(Structure3):
        _fields = ['name', 'shares', 'price']
    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, 91.1, date='8/2/2012')

class Structure4:
    _fields = []

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        self.__dict__.update(zip(self._fields, args))