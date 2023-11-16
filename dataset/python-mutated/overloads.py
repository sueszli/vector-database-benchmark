from builtins import _test_sink, _test_source
from typing import overload, Union

@overload
def f(x: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

@overload
def f(x: str) -> None:
    if False:
        print('Hello World!')
    pass

def f(x: Union[int, str]) -> None:
    if False:
        return 10
    call_me(x)

def call_me(x):
    if False:
        return 10
    _test_sink(x)

class A:

    def call_me(self, x):
        if False:
            i = 10
            return i + 15
        _test_sink(x)

@overload
def g(o: A) -> None:
    if False:
        return 10
    pass

@overload
def g(o: int) -> None:
    if False:
        i = 10
        return i + 15
    pass

def g(o):
    if False:
        print('Hello World!')
    x = _test_source()
    if isinstance(o, A):
        o.call_me(x)