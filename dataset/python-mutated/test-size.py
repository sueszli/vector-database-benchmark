from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    if False:
        while True:
            i = 10
    ' Returns the approximate memory footprint an object and all of its contents.\n\n    Automatically finds the contents of the following builtin containers and\n    their subclasses:  tuple, list, deque, dict, set and frozenset.\n    To search other containers, add handlers to iterate over their contents:\n\n        handlers = {SomeContainerClass: iter,\n                    OtherContainerClass: OtherContainerClass.get_elements}\n\n    '
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter, list: iter, deque: iter, dict: dict_handler, set: iter, frozenset: iter}
    all_handlers.update(handlers)
    seen = set()
    default_size = getsizeof(0)

    def sizeof(o):
        if False:
            print('Hello World!')
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        if verbose:
            print(s, type(o), repr(o), file=stderr)
        for (typ, handler) in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
    return sizeof(o)

def doit():
    if False:
        for i in range(10):
            print('nop')
    print('HERE WE GO')
    q1 = list(range(0, 2000))
    q2 = list(range(0, 20000))
    q3 = list(range(0, 200000))
    r = range(0, 2000000)
    q4 = []
    for i in r:
        q4.append(i)
    z = 2000000 * getsizeof(1)
    print(z)
    print('q4', total_size(q4) / (1024 * 1024))
    del q4
for i in range(12):
    doit()