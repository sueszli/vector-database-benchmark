import pytest

@pytest.fixture
def some(request):
    if False:
        i = 10
        return i + 15
    return request.function.__name__

@pytest.fixture
def other(request):
    if False:
        while True:
            i = 10
    return 42

def test_func(some, other):
    if False:
        print('Hello World!')
    pass