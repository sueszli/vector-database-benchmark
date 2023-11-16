import pytest

@pytest.fixture
def order():
    if False:
        print('Hello World!')
    return []

@pytest.fixture
def a(order):
    if False:
        print('Hello World!')
    order.append('a')

@pytest.fixture
def b(a, order):
    if False:
        i = 10
        return i + 15
    order.append('b')

@pytest.fixture
def c(b, order):
    if False:
        for i in range(10):
            print('nop')
    order.append('c')

@pytest.fixture
def d(c, b, order):
    if False:
        print('Hello World!')
    order.append('d')

@pytest.fixture
def e(d, b, order):
    if False:
        while True:
            i = 10
    order.append('e')

@pytest.fixture
def f(e, order):
    if False:
        for i in range(10):
            print('nop')
    order.append('f')

@pytest.fixture
def g(f, c, order):
    if False:
        return 10
    order.append('g')

def test_order(g, order):
    if False:
        for i in range(10):
            print('nop')
    assert order == ['a', 'b', 'c', 'd', 'e', 'f', 'g']