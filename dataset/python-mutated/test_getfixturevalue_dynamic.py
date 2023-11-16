import pytest

@pytest.fixture
def dynamic():
    if False:
        i = 10
        return i + 15
    pass

@pytest.fixture
def a(request):
    if False:
        while True:
            i = 10
    request.getfixturevalue('dynamic')

@pytest.fixture
def b(a):
    if False:
        print('Hello World!')
    pass

def test(b, request):
    if False:
        return 10
    assert request.fixturenames == ['b', 'request', 'a', 'dynamic']