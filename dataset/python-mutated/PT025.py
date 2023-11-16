import pytest

@pytest.mark.usefixtures('a')
def test_something():
    if False:
        print('Hello World!')
    pass

@pytest.mark.usefixtures('a')
@pytest.fixture()
def my_fixture():
    if False:
        print('Hello World!')
    return 0

@pytest.fixture()
@pytest.mark.usefixtures('a')
def my_fixture():
    if False:
        print('Hello World!')
    return 0