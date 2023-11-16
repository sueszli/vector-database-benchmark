from .funcs import compose, juxt
from .colls import some, none, one
__all__ = ['all_fn', 'any_fn', 'none_fn', 'one_fn', 'some_fn']

def all_fn(*fs):
    if False:
        i = 10
        return i + 15
    'Constructs a predicate, which holds when all fs hold.'
    return compose(all, juxt(*fs))

def any_fn(*fs):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a predicate, which holds when any fs holds.'
    return compose(any, juxt(*fs))

def none_fn(*fs):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a predicate, which holds when none of fs hold.'
    return compose(none, juxt(*fs))

def one_fn(*fs):
    if False:
        while True:
            i = 10
    'Constructs a predicate, which holds when exactly one of fs holds.'
    return compose(one, juxt(*fs))

def some_fn(*fs):
    if False:
        print('Hello World!')
    'Constructs a function, which calls fs one by one\n       and returns first truthy result.'
    return compose(some, juxt(*fs))