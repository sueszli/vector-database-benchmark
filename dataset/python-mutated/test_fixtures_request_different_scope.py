import pytest

@pytest.fixture
def order():
    if False:
        i = 10
        return i + 15
    return []

@pytest.fixture
def outer(order, inner):
    if False:
        return 10
    order.append('outer')

class TestOne:

    @pytest.fixture
    def inner(self, order):
        if False:
            print('Hello World!')
        order.append('one')

    def test_order(self, order, outer):
        if False:
            return 10
        assert order == ['one', 'outer']

class TestTwo:

    @pytest.fixture
    def inner(self, order):
        if False:
            print('Hello World!')
        order.append('two')

    def test_order(self, order, outer):
        if False:
            i = 10
            return i + 15
        assert order == ['two', 'outer']