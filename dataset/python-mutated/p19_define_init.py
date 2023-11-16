"""
Topic: 在定义的时候初始化类的成员
Desc : 
"""
import operator

class StructTupleMeta(type):

    def __init__(cls, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        for (n, name) in enumerate(cls._fields):
            setattr(cls, name, property(operator.itemgetter(n)))

class StructTuple(tuple, metaclass=StructTupleMeta):
    _fields = []

    def __new__(cls, *args):
        if False:
            while True:
                i = 10
        if len(args) != len(cls._fields):
            raise ValueError('{} arguments required'.format(len(cls._fields)))
        return super().__new__(cls, args)

class Stock(StructTuple):
    _fields = ['name', 'shares', 'price']

class Point(StructTuple):
    _fields = ['x', 'y']