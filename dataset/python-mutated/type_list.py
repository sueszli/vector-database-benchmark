from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import annotate
from .type_void import void
from . import type_int
from .type_spec import Type
from .get_type_info import get_type_info

def memoize(f):
    if False:
        return 10
    memo = {}

    def helper(x, init_length=None, dynamic_length=True):
        if False:
            for i in range(10):
                print('nop')
        if x not in memo:
            memo[x, init_length, dynamic_length] = f(x, init_length, dynamic_length)
        return memo[x, init_length, dynamic_length]
    return helper

class empty_list:

    @classmethod
    def __type_info__(cls):
        if False:
            for i in range(10):
                print('nop')
        return Type('empty_list', python_class=cls)

@memoize
def list(arg, init_length=None, dynamic_length=True):
    if False:
        return 10

    class list:
        T = [arg, init_length, dynamic_length]

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.val = []

        @classmethod
        def __type_info__(cls):
            if False:
                for i in range(10):
                    print('nop')
            return Type('list', [get_type_info(arg)], python_class=cls)

        @annotate(void, other=T[0])
        def append(self, other):
            if False:
                print('Hello World!')
            assert isinstance(other, self.T[0])
            self.val.append(other)

        @annotate(T[0], index=type_int.int)
        def __getitem__(self, index):
            if False:
                while True:
                    i = 10
            assert isinstance(index, type_int.int)
            return self.val[index.val]

        @annotate(void, index=type_int.int, newval=T[0])
        def __setitem__(self, index, newval):
            if False:
                i = 10
                return i + 15
            assert isinstance(index, type_int.int)
            assert isinstance(newval, self.T[0])
            self.val[index.val] = newval

        @annotate(type_int.int)
        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return type_int.int(len(self.val)) if self.T[1] is None else self.T[1]
    list.__template_name__ = 'list[' + arg.__name__ + ']'
    return list

def is_list(t):
    if False:
        i = 10
        return i + 15
    if t is None:
        return False
    return get_type_info(t).name == 'list'