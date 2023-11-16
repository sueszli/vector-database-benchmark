import typing
from builtins import _test_sink, _test_source

def foo():
    if False:
        print('Hello World!')

    def inner():
        if False:
            return 10
        x = _test_source()
        _test_sink(x)

    def inner_with_model():
        if False:
            return 10
        return _test_source()

def outer(x: int) -> None:
    if False:
        for i in range(10):
            print('nop')

    def inner(x: int) -> None:
        if False:
            print('Hello World!')
        _test_sink(x)
    return inner(x)

def call_outer() -> None:
    if False:
        print('Hello World!')
    outer(_test_source())

def some_sink(x: int) -> None:
    if False:
        return 10
    _test_sink(x)

def outer_calling_other_function(x: int) -> None:
    if False:
        for i in range(10):
            print('nop')

    def inner_calling_other_function(x: int) -> None:
        if False:
            return 10
        some_sink(x)
    inner_calling_other_function(x)

def parameter_function(add: typing.Optional[typing.Callable[[str, str], str]], x: str) -> str:
    if False:
        return 10
    if add is None:

        def add(x: str, y: str) -> str:
            if False:
                print('Hello World!')
            return x + y
    return add('/bin/bash', x)

def duplicate_function():
    if False:
        return 10
    foo()

def duplicate_function():
    if False:
        return 10
    foo()
g = None

def nested_global_function(x: str) -> str:
    if False:
        print('Hello World!')
    global g

    def g(x: str, y: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return x + y
    return g('/bin/bash', x)

def access_variables_in_outer_scope_issue():
    if False:
        print('Hello World!')
    x = _test_source()

    def inner():
        if False:
            for i in range(10):
                print('nop')
        _test_sink(x)
    inner()

def access_variables_in_outer_scope_source():
    if False:
        return 10
    x = _test_source()

    def inner():
        if False:
            while True:
                i = 10
        return x
    return inner()

def test_access_variables_in_outer_scope_source():
    if False:
        return 10
    _test_sink(access_variables_in_outer_scope_source())

def access_parameter_in_inner_scope_sink(x):
    if False:
        return 10

    def inner():
        if False:
            print('Hello World!')
        _test_sink(x)
    inner()

def test_access_parameter_in_inner_scope_sink():
    if False:
        i = 10
        return i + 15
    access_parameter_in_inner_scope_sink(_test_source())

def access_parameter_in_inner_scope_tito(x):
    if False:
        i = 10
        return i + 15

    def inner():
        if False:
            print('Hello World!')
        return x
    return inner()

def test_access_parameter_in_inner_scope_tito():
    if False:
        return 10
    _test_sink(access_parameter_in_inner_scope_tito(_test_source()))

class A:
    a: str = ''

def test_mutation_of_class():
    if False:
        print('Hello World!')
    a = A()

    def set_a(a):
        if False:
            print('Hello World!')
        a.a = _test_source()
    set_a(a)
    _test_sink(a)