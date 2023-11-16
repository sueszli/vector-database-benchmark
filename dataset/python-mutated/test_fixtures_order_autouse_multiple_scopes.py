import pytest

@pytest.fixture(scope='class')
def order():
    if False:
        i = 10
        return i + 15
    return []

@pytest.fixture(scope='class', autouse=True)
def c1(order):
    if False:
        for i in range(10):
            print('nop')
    order.append('c1')

@pytest.fixture(scope='class')
def c2(order):
    if False:
        while True:
            i = 10
    order.append('c2')

@pytest.fixture(scope='class')
def c3(order, c1):
    if False:
        print('Hello World!')
    order.append('c3')

class TestClassWithC1Request:

    def test_order(self, order, c1, c3):
        if False:
            print('Hello World!')
        assert order == ['c1', 'c3']

class TestClassWithoutC1Request:

    def test_order(self, order, c2):
        if False:
            return 10
        assert order == ['c1', 'c2']