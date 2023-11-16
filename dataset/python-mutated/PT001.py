import pytest
from pytest import fixture
from pytest import fixture as aliased

@pytest.fixture
def no_parentheses():
    if False:
        i = 10
        return i + 15
    return 42

@pytest.fixture()
def parentheses_no_params():
    if False:
        print('Hello World!')
    return 42

@pytest.fixture(scope='module')
def parentheses_with_params():
    if False:
        print('Hello World!')
    return 42

@pytest.fixture()
def parentheses_no_params_multiline():
    if False:
        i = 10
        return i + 15
    return 42

@fixture
def imported_from_no_parentheses():
    if False:
        while True:
            i = 10
    return 42

@fixture()
def imported_from_parentheses_no_params():
    if False:
        return 10
    return 42

@fixture(scope='module')
def imported_from_parentheses_with_params():
    if False:
        for i in range(10):
            print('nop')
    return 42

@fixture()
def imported_from_parentheses_no_params_multiline():
    if False:
        return 10
    return 42

@aliased
def aliased_no_parentheses():
    if False:
        return 10
    return 42

@aliased()
def aliased_parentheses_no_params():
    if False:
        i = 10
        return i + 15
    return 42

@aliased(scope='module')
def aliased_parentheses_with_params():
    if False:
        for i in range(10):
            print('nop')
    return 42

@aliased()
def aliased_parentheses_no_params_multiline():
    if False:
        for i in range(10):
            print('nop')
    return 42