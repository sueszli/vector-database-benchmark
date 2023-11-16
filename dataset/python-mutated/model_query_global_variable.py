import typing
from builtins import _test_sink, _test_source
from os import environ
foo = []

def f():
    if False:
        while True:
            i = 10
    _test_sink(environ)

class Baz:
    ...
hello = 'hello'
world: str = 'world'

def g():
    if False:
        for i in range(10):
            print('nop')
    foo.append(1)

def h():
    if False:
        while True:
            i = 10
    _test_sink(Baz)
_test_sink(_test_source())

def returns_any() -> typing.Any:
    if False:
        while True:
            i = 10
    return {'hello': Baz()}
typed_global_dict: typing.Dict[str, Baz] = returns_any()
untyped_global_dict = returns_any()
typed_global_lambda: typing.Callable[[int, str], int] = lambda x, y: x + int(y)

def fun(x: int, y: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    return x + int(y)
typed_global_callable: typing.Callable[[int, str], int] = fun
untyped_global_callable = fun(1, '2')
typed_callable_assignment: int = fun(1, '2')
untyped_callable_assignment = fun