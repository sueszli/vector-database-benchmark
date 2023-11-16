import abc
from abc import abstractmethod
import pytest

@pytest.fixture()
def _patch_something(mocker):
    if False:
        return 10
    mocker.patch('some.thing')

@pytest.fixture()
def _patch_something(mocker):
    if False:
        while True:
            i = 10
    if something:
        return
    mocker.patch('some.thing')

@pytest.fixture()
def _activate_context():
    if False:
        i = 10
        return i + 15
    with context:
        yield

class BaseTest:

    @pytest.fixture()
    @abc.abstractmethod
    def my_fixture():
        if False:
            return 10
        raise NotImplementedError

class BaseTest:

    @pytest.fixture()
    @abstractmethod
    def my_fixture():
        if False:
            return 10
        raise NotImplementedError

@pytest.fixture()
def my_fixture():
    if False:
        while True:
            i = 10
    yield from some_generator()

@pytest.fixture()
def my_fixture():
    if False:
        return 10
    yield 1

@pytest.fixture()
def patch_something(mocker):
    if False:
        return 10
    mocker.patch('some.thing')

@pytest.fixture()
def activate_context():
    if False:
        i = 10
        return i + 15
    with context:
        yield