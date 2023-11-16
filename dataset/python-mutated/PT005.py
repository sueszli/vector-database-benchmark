import abc
from abc import abstractmethod
import pytest

@pytest.fixture()
def my_fixture(mocker):
    if False:
        for i in range(10):
            print('nop')
    return 0

@pytest.fixture()
def activate_context():
    if False:
        for i in range(10):
            print('nop')
    with get_context() as context:
        yield context

@pytest.fixture()
def _any_fixture(mocker):
    if False:
        print('Hello World!')

    def nested_function():
        if False:
            i = 10
            return i + 15
        return 1
    mocker.patch('...', nested_function)

class BaseTest:

    @pytest.fixture()
    @abc.abstractmethod
    def _my_fixture():
        if False:
            while True:
                i = 10
        return NotImplemented

class BaseTest:

    @pytest.fixture()
    @abstractmethod
    def _my_fixture():
        if False:
            i = 10
            return i + 15
        return NotImplemented

@pytest.fixture()
def _my_fixture(mocker):
    if False:
        print('Hello World!')
    return 0

@pytest.fixture()
def _activate_context():
    if False:
        i = 10
        return i + 15
    with get_context() as context:
        yield context

@pytest.fixture()
def _activate_context():
    if False:
        while True:
            i = 10
    if some_condition:
        with get_context() as context:
            yield context
    else:
        yield from other_context()