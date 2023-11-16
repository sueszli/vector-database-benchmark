import pytest

@pytest.fixture
def order():
    if False:
        return 10
    return []

@pytest.fixture
def c1(order):
    if False:
        i = 10
        return i + 15
    order.append('c1')

@pytest.fixture
def c2(order):
    if False:
        while True:
            i = 10
    order.append('c2')

class TestClassWithAutouse:

    @pytest.fixture(autouse=True)
    def c3(self, order, c2):
        if False:
            while True:
                i = 10
        order.append('c3')

    def test_req(self, order, c1):
        if False:
            i = 10
            return i + 15
        assert order == ['c2', 'c3', 'c1']

    def test_no_req(self, order):
        if False:
            while True:
                i = 10
        assert order == ['c2', 'c3']

class TestClassWithoutAutouse:

    def test_req(self, order, c1):
        if False:
            i = 10
            return i + 15
        assert order == ['c1']

    def test_no_req(self, order):
        if False:
            i = 10
            return i + 15
        assert order == []