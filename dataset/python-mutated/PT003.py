import pytest

@pytest.fixture()
def ok_no_scope():
    if False:
        for i in range(10):
            print('nop')
    ...

@pytest.fixture(scope='module')
def ok_other_scope():
    if False:
        i = 10
        return i + 15
    ...

@pytest.fixture(scope='function')
def error():
    if False:
        return 10
    ...

@pytest.fixture(scope='function', name='my_fixture')
def error_multiple_args():
    if False:
        i = 10
        return i + 15
    ...

@pytest.fixture(name='my_fixture', scope='function')
def error_multiple_args():
    if False:
        print('Hello World!')
    ...

@pytest.fixture(name='my_fixture', scope='function', **kwargs)
def error_second_arg():
    if False:
        while True:
            i = 10
    ...

@pytest.fixture('my_fixture', scope='function')
def error_arg():
    if False:
        while True:
            i = 10
    ...

@pytest.fixture(scope='function', name='my_fixture')
def error_multiple_args():
    if False:
        while True:
            i = 10
    ...

@pytest.fixture(name='my_fixture', scope='function')
def error_multiple_args():
    if False:
        return 10
    ...

@pytest.fixture('hello', name, *args, scope='function', name2=name, name3='my_fixture', **kwargs)
def error_multiple_args():
    if False:
        i = 10
        return i + 15
    ...