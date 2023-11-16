import pytest

def test_foo():
    if False:
        while True:
            i = 10
    assert True

@pytest.mark.parametrize('i', range(3))
def test_bar(i):
    if False:
        print('Hello World!')
    assert True