import pytest

@pytest.fixture
def something(request):
    if False:
        for i in range(10):
            print('nop')
    return request.function.__name__

class TestClass:

    def test_method(self, something):
        if False:
            return 10
        assert something == 'test_method'

def test_func(something):
    if False:
        i = 10
        return i + 15
    assert something == 'test_func'