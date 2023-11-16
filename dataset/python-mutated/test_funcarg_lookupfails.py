import pytest

@pytest.fixture
def xyzsomething(request):
    if False:
        print('Hello World!')
    return 42

def test_func(some):
    if False:
        return 10
    pass