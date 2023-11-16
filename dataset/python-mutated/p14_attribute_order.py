"""
Topic: 类定义中属性的顺序
Desc : 
"""
from collections import OrderedDict

class Typed:
    _expected_type = type(None)

    def __init__(self, name=None):
        if False:
            print('Hello World!')
        self._name = name

    def __set__(self, instance, value):
        if False:
            print('Hello World!')
        if not isinstance(value, self._expected_type):
            raise TypeError('Expected ' + str(self._expected_type))
        instance.__dict__[self._name] = value

class Integer(Typed):
    _expected_type = int

class Float(Typed):
    _expected_type = float

class String(Typed):
    _expected_type = str

class OrderedMeta(type):

    def __new__(cls, clsname, bases, clsdict):
        if False:
            print('Hello World!')
        d = dict(clsdict)
        order = []
        for (name, value) in clsdict.items():
            if isinstance(value, Typed):
                value._name = name
                order.append(name)
        d['_order'] = order
        return type.__new__(cls, clsname, bases, d)

    @classmethod
    def __prepare__(cls, clsname, bases):
        if False:
            for i in range(10):
                print('nop')
        return OrderedDict()

class Structure(metaclass=OrderedMeta):

    def as_csv(self):
        if False:
            i = 10
            return i + 15
        return ','.join((str(getattr(self, name)) for name in self._order))

class Stock(Structure):
    name = String()
    shares = Integer()
    price = Float()

    def __init__(self, name, shares, price):
        if False:
            print('Hello World!')
        self.name = name
        self.shares = shares
        self.price = price