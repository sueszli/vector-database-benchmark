"""
The module for testing variable annotations.
Empty lines above are for good reason (testing for correct line numbers)
"""
from typing import Optional
from functools import wraps
__annotations__[1] = 2

class C:
    x = 5
    y: Optional['C'] = None
from typing import Tuple
x: int = 5
y: str = x
f: Tuple[int, int]

class M(type):
    __annotations__['123'] = 123
    o: type = object
(pars): bool = True

class D(C):
    j: str = 'hi'
    k: str = 'bye'
from types import new_class
h_class = new_class('H', (C,))
j_class = new_class('J')

class F:
    z: int = 5

    def __init__(self, x):
        if False:
            print('Hello World!')
        pass

class Y(F):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(F, self).__init__(123)

class Meta(type):

    def __new__(meta, name, bases, namespace):
        if False:
            for i in range(10):
                print('nop')
        return super().__new__(meta, name, bases, namespace)

class S(metaclass=Meta):
    x: str = 'something'
    y: str = 'something else'

def foo(x: int=10):
    if False:
        i = 10
        return i + 15

    def bar(y: List[str]):
        if False:
            for i in range(10):
                print('nop')
        x: str = 'yes'
    bar()

def dec(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return func(*args, **kwargs)
    return wrapper
u: int | float