from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import annotate
from .type_spec import Type
from . import type_bool
from . import type_int
from .type_void import void
from .get_type_info import get_type_info

def memoize(f):
    if False:
        return 10
    memo = {}

    def helper(x, y):
        if False:
            for i in range(10):
                print('nop')
        if (x, y) not in memo:
            memo[x, y] = f(x, y)
        return memo[x, y]
    return helper

class empty_dict:

    @classmethod
    def __type_info__(cls):
        if False:
            while True:
                i = 10
        return Type('empty_dict', python_class=cls)

@memoize
def dict(keytype, valuetype):
    if False:
        return 10

    class dict:
        T = [keytype, valuetype]

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.val = {}

        @classmethod
        def __type_info__(cls):
            if False:
                while True:
                    i = 10
            return Type('dict', [get_type_info(keytype), get_type_info(valuetype)], cls)

        @annotate(T[1], key=T[0])
        def __getitem__(self, key):
            if False:
                while True:
                    i = 10
            assert isinstance(key, self.T[0])
            return self.val[key]

        @annotate(void, key=T[0], newval=T[1])
        def __setitem__(self, key, newval):
            if False:
                while True:
                    i = 10
            assert isinstance(key, self.T[0])
            assert isinstance(newval, self.T[1])
            self.val[key] = newval

        @annotate(type_int.int)
        def __len__(self):
            if False:
                return 10
            return type_int.int(len(self.val))

        @annotate(type_bool.bool, key=T[0])
        def __contains__(self, key):
            if False:
                print('Hello World!')
            return key in self.val[key]
    dict.__template_name__ = 'dict[' + keytype.__name__ + ',' + valuetype.__name__ + ']'
    return dict