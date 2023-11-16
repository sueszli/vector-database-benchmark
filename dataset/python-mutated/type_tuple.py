from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import annotate
from . import type_int
from . import type_unknown
from .type_spec import Type
from .get_type_info import get_type_info
_global_tuple = tuple

def memoize(f):
    if False:
        for i in range(10):
            print('nop')
    memo = {}

    def helper(x):
        if False:
            return 10
        x = _global_tuple(x)
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

class empty_list:

    @classmethod
    def __type_info__(cls):
        if False:
            i = 10
            return i + 15
        return Type('empty_list', python_class=cls)

@memoize
def tuple(args):
    if False:
        print('Hello World!')
    args = _global_tuple((i if i is not None else type_unknown.unknown for i in args))

    class tuple:
        T = args

        def __init__(self):
            if False:
                print('Hello World!')
            self.val = [arg() for arg in args]

        @classmethod
        def __type_info__(cls):
            if False:
                return 10
            return Type('tuple', [get_type_info(arg) for arg in args], python_class=cls)

        @annotate(type_int.int)
        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(args)
    tuple.__template_name__ = 'tuple[' + ','.join([get_type_info(arg).name for arg in args]) + ']'
    return tuple