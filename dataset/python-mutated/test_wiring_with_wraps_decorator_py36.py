"""Test that wiring works properly with @functools.wraps decorator.

See issue for details: https://github.com/ets-labs/python-dependency-injector/issues/454
"""
import functools
from dependency_injector.wiring import inject, Provide
from pytest import fixture
from samples.wiring.container import Container

@fixture
def container():
    if False:
        while True:
            i = 10
    container = Container()
    yield container
    container.unwire()

def decorator1(func):
    if False:
        return 10

    @functools.wraps(func)
    @inject
    def wrapper(value1: int=Provide[Container.config.value1]):
        if False:
            print('Hello World!')
        result = func()
        return result + value1
    return wrapper

def decorator2(func):
    if False:
        return 10

    @functools.wraps(func)
    @inject
    def wrapper(value2: int=Provide[Container.config.value2]):
        if False:
            while True:
                i = 10
        result = func()
        return result + value2
    return wrapper

@decorator1
@decorator2
def sample():
    if False:
        for i in range(10):
            print('nop')
    return 2

def test_wraps(container: Container):
    if False:
        for i in range(10):
            print('nop')
    container.wire(modules=[__name__])
    container.config.from_dict({'value1': 42, 'value2': 15})
    assert sample() == 2 + 42 + 15