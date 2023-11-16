import functools
import multiprocessing
import typing
from builtins import _test_sink, _test_source

def a_flows_to_sink(a, b):
    if False:
        i = 10
        return i + 15
    _test_sink(a)

def partial_application_with_tainted():
    if False:
        print('Hello World!')
    x = _test_source()
    functools.partial(a_flows_to_sink, x)

def partial_application_with_benign():
    if False:
        i = 10
        return i + 15
    x = 1
    functools.partial(a_flows_to_sink, x)

def partial_application_with_named_a():
    if False:
        while True:
            i = 10
    x = _test_source()
    functools.partial(a_flows_to_sink, a=x)

def partial_application_with_named_b():
    if False:
        return 10
    x = _test_source()
    functools.partial(a_flows_to_sink, b=x)

def multiprocessing_tainted():
    if False:
        print('Hello World!')
    multiprocessing.Process(target=a_flows_to_sink, args=(_test_source(), 1))

def multiprocessing_not_tainted():
    if False:
        i = 10
        return i + 15
    multiprocessing.Process(target=a_flows_to_sink, args=(1, _test_source()))

class PartialDecorator:

    def __init__(self, func: typing.Callable) -> None:
        if False:
            while True:
                i = 10
        self._func = func

    def __get__(self, instance, owner) -> functools.partial[None]:
        if False:
            return 10
        return functools.partial(self.__call__, instance=instance)

    def __call__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        instance = kwargs.pop('instance')
        return self._func(instance, *args, **kwargs)

class PartialConstructor:

    @PartialDecorator
    def __init__(self, x: str, y: str) -> None:
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y

def dunder_call_partial_constructor(x: str, y: str) -> C:
    if False:
        while True:
            i = 10
    return PartialConstructor(x, y)

class NestedDefineDecorator:

    def __init__(self, func: typing.Callable) -> None:
        if False:
            while True:
                i = 10
        self._func = func

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10

        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['instance'] = instance
            return self.__call__(*args, **kwargs)
        return wrapper

    def __call__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        instance = kwargs.pop('instance')
        return self._func(instance, *args, **kwargs)

class NestedDefineConstructor:

    @NestedDefineDecorator
    def __init__(self, x: str, y: str) -> None:
        if False:
            return 10
        self.x = x
        self.y = y

def dunder_call_nested_define_constructor(x: str, y: str) -> C:
    if False:
        i = 10
        return i + 15
    return NestedDefineConstructor(x, y)