from builtins import _test_sink, _test_source
from functools import wraps
from typing import Awaitable, Callable

def with_logging(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        return 10

    def some_helper(x: int) -> None:
        if False:
            print('Hello World!')
        print(x)
        _test_sink(x)

    def inner(x: int) -> None:
        if False:
            return 10
        _test_sink(x)
        f(x)
        some_helper(x)
    return inner

def with_logging2(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        return 10

    def inner(x: int) -> None:
        if False:
            i = 10
            return i + 15
        _test_sink(x)
        f(x)
    return inner

def skip_this_decorator(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        i = 10
        return i + 15

    def inner(x: int) -> None:
        if False:
            while True:
                i = 10
        _test_sink(x)
        f(x)
    return inner

def ignore_this_decorator(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        while True:
            i = 10
    return f

def ignore_this_decorator_factory(add: int):
    if False:
        for i in range(10):
            print('nop')

    def decorator(f: Callable[[int], None]) -> Callable[[int], None]:
        if False:
            return 10

        def inner(x: int) -> None:
            if False:
                while True:
                    i = 10
            f(x + add)
        return inner
    return decorator

class ignore_this_decorator_class:

    def __call__(self, f: Callable[[int], None]) -> Callable[[int], None]:
        if False:
            i = 10
            return i + 15
        return f

@with_logging
@with_logging2
def decorated_logging_logging2(x: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x)

@skip_this_decorator
def decorated_skip_this_decorator(x: int) -> None:
    if False:
        print('Hello World!')
    _test_sink(x)

@with_logging2
@skip_this_decorator
def decorated_logging2_skip_this_decorator(x: int) -> None:
    if False:
        while True:
            i = 10
    _test_sink(x)

@ignore_this_decorator
def decorated_ignore_this_decorator(x: int) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(x)

@ignore_this_decorator_factory(1)
def decorated_ignore_this_decorator_factory(x: int) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(x)

@ignore_this_decorator_class()
def decorated_ignore_this_decorator_class(x: int) -> None:
    if False:
        print('Hello World!')
    _test_sink(x)

@ignore_this_decorator
@skip_this_decorator
def decorated_ignore_then_skip_decorator(x: int) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(x)

@with_logging
@ignore_this_decorator
def decorated_logging_ignore_this_decorator(x: int) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(x)

def pass_local_variable_to_x(f: Callable) -> Callable:
    if False:
        i = 10
        return i + 15

    @wraps(f)
    def inner(request: str, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        _test_sink(request)
        x = 42
        f(request, x, *args, **kwargs)
    return inner

@pass_local_variable_to_x
def handle_request(request: str, x: int, y: int) -> None:
    if False:
        print('Hello World!')
    _test_sink(x)

class Foo:

    def return_source(self) -> int:
        if False:
            return 10
        return _test_source()

def identity(f: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')

    @wraps(f)
    def inner(*args, **kwargs) -> Callable:
        if False:
            for i in range(10):
                print('nop')
        return f(*args, **kwargs)
    return inner

@identity
def return_foo() -> Foo:
    if False:
        for i in range(10):
            print('nop')
    return Foo()

def call_return_foo() -> None:
    if False:
        while True:
            i = 10
    foo = return_foo()
    _test_sink(foo.return_source())

def main() -> None:
    if False:
        return 10
    decorated_logging_logging2(_test_source())
    decorated_skip_this_decorator(_test_source())
    decorated_logging2_skip_this_decorator(_test_source())
    decorated_ignore_this_decorator(_test_source())
    decorated_ignore_this_decorator_factory(_test_source())
    decorated_ignore_this_decorator_class(_test_source())
    decorated_ignore_then_skip_decorator(_test_source())
    decorated_logging_ignore_this_decorator(_test_source())
    handle_request('hello', _test_source(), 42)
    handle_request(_test_source(), 42, 42)
    call_return_foo()