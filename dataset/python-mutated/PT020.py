import pytest

@pytest.fixture()
def ok_no_parameters():
    if False:
        print('Hello World!')
    return 0

@pytest.fixture
def ok_without_parens():
    if False:
        return 10
    return 0

@pytest.yield_fixture()
def error_without_parens():
    if False:
        for i in range(10):
            print('nop')
    return 0

@pytest.yield_fixture
def error_with_parens():
    if False:
        for i in range(10):
            print('nop')
    return 0