import pytest

@pytest.fixture()
def ok_complex_logic():
    if False:
        return 10
    if some_condition:
        resource = acquire_resource()
        yield resource
        resource.release()
        return
    yield None

@pytest.fixture()
def error():
    if False:
        for i in range(10):
            print('nop')
    resource = acquire_resource()
    yield resource
import typing
from typing import Generator

@pytest.fixture()
def ok_complex_logic() -> typing.Generator[Resource, None, None]:
    if False:
        print('Hello World!')
    if some_condition:
        resource = acquire_resource()
        yield resource
        resource.release()
        return
    yield None

@pytest.fixture()
def error() -> typing.Generator[typing.Any, None, None]:
    if False:
        print('Hello World!')
    resource = acquire_resource()
    yield resource

@pytest.fixture()
def error() -> Generator[Resource, None, None]:
    if False:
        return 10
    resource = acquire_resource()
    yield resource