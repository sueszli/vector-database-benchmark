from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import class_annotate, annotate, delay_type
from .type_spec import Type

@class_annotate()
class bool:

    def __init__(self, v=False):
        if False:
            print('Hello World!')
        self.val = v

    @classmethod
    def __type_info__(cls):
        if False:
            i = 10
            return i + 15
        return Type('bool', python_class=cls)

    @annotate(delay_type.bool, other=delay_type.bool)
    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.val == other.val)

    @annotate(delay_type.bool, other=delay_type.bool)
    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return bool(self.val != other.val)

    @annotate(delay_type.bool)
    def __not__(self, other):
        if False:
            while True:
                i = 10
        return bool(not other.val)

    @annotate(delay_type.bool)
    def __bool__(self):
        if False:
            while True:
                i = 10
        return self.val

    @annotate(delay_type.int)
    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return int(self)

    @annotate(delay_type.double)
    def __double__(self):
        if False:
            while True:
                i = 10
        return float(self.val)

    @annotate(delay_type.str)
    def __str__(self):
        if False:
            return 10
        return str(self.val)

def is_bool(t):
    if False:
        return 10
    return t is bool or isinstance(t, bool)