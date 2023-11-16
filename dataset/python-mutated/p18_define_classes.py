"""
Topic: 以编程方式定义类
Desc : 
"""

def __init__(self, name, shares, price):
    if False:
        return 10
    self.name = name
    self.shares = shares
    self.price = price

def cost(self):
    if False:
        i = 10
        return i + 15
    return self.shares * self.price
cls_dict = {'__init__': __init__, 'cost': cost}
import types
Stock = types.new_class('Stock', (), {}, lambda ns: ns.update(cls_dict))
Stock.__module__ = __name__
import operator
import types
import sys

def named_tuple(classname, fieldnames):
    if False:
        return 10
    cls_dict = {name: property(operator.itemgetter(n)) for (n, name) in enumerate(fieldnames)}

    def __new__(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) != len(fieldnames):
            raise TypeError('Expected {} arguments'.format(len(fieldnames)))
        return tuple.__new__(cls, args)
    cls_dict['__new__'] = __new__
    cls = types.new_class(classname, (tuple,), {}, lambda ns: ns.update(cls_dict))
    cls.__module__ = sys._getframe(1).f_globals['__name__']
    return cls