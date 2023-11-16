import pytest

@pytest.fixture()
def my_fixture():
    if False:
        return 10
    return 0

@pytest.fixture(scope='module')
def my_fixture():
    if False:
        i = 10
        return i + 15
    return 0

@pytest.fixture('module')
def my_fixture():
    if False:
        i = 10
        return i + 15
    return 0

@pytest.fixture('module', autouse=True)
def my_fixture():
    if False:
        for i in range(10):
            print('nop')
    return 0