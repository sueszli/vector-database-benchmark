""" Pep-0484 type hinted decorators """
from typing import Callable

def decorator(func):
    if False:
        print('Hello World!')

    def wrapper(*a, **k):
        if False:
            while True:
                i = 10
        return str(func(*a, **k))
    return wrapper

def typed_decorator(func: Callable[..., int]) -> Callable[..., str]:
    if False:
        i = 10
        return i + 15
    ...

@decorator
def plain_func() -> int:
    if False:
        i = 10
        return i + 15
    return 4
plain_func()

@typed_decorator
def typed_func() -> int:
    if False:
        print('Hello World!')
    return 4
typed_func()

class X:

    @decorator
    def plain_method(self) -> int:
        if False:
            return 10
        return 4

    @typed_decorator
    def typed_method(self) -> int:
        if False:
            i = 10
            return i + 15
        return 4
inst = X()
inst.plain_method()
inst.typed_method()