import pytest

@pytest.fixture(scope='session')
def order():
    if False:
        for i in range(10):
            print('nop')
    return []

@pytest.fixture
def func(order):
    if False:
        print('Hello World!')
    order.append('function')

@pytest.fixture(scope='class')
def cls(order):
    if False:
        print('Hello World!')
    order.append('class')

@pytest.fixture(scope='module')
def mod(order):
    if False:
        while True:
            i = 10
    order.append('module')

@pytest.fixture(scope='package')
def pack(order):
    if False:
        while True:
            i = 10
    order.append('package')

@pytest.fixture(scope='session')
def sess(order):
    if False:
        print('Hello World!')
    order.append('session')

class TestClass:

    def test_order(self, func, cls, mod, pack, sess, order):
        if False:
            i = 10
            return i + 15
        assert order == ['session', 'package', 'module', 'class', 'function']