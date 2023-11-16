from .. import SubSomething2

def test_func():
    if False:
        return 10
    assert SubSomething2.calledByTest2() == (42, 2)