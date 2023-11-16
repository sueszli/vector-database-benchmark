import pytest

@pytest.fixture(scope='module', params=range(966))
def foo(request):
    if False:
        print('Hello World!')
    return request.param

def test_it(foo):
    if False:
        for i in range(10):
            print('nop')
    pass

def test_it2(foo):
    if False:
        return 10
    pass