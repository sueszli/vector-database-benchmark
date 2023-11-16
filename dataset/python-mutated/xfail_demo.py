import pytest
xfail = pytest.mark.xfail

@xfail
def test_hello():
    if False:
        return 10
    assert 0

@xfail(run=False)
def test_hello2():
    if False:
        for i in range(10):
            print('nop')
    assert 0

@xfail("hasattr(os, 'sep')")
def test_hello3():
    if False:
        return 10
    assert 0

@xfail(reason='bug 110')
def test_hello4():
    if False:
        while True:
            i = 10
    assert 0

@xfail('pytest.__version__[0] != "17"')
def test_hello5():
    if False:
        i = 10
        return i + 15
    assert 0

def test_hello6():
    if False:
        print('Hello World!')
    pytest.xfail('reason')

@xfail(raises=IndexError)
def test_hello7():
    if False:
        print('Hello World!')
    x = []
    x[1] = 1