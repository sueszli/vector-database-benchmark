from .. import SubSomething1

def test_subfunc1():
    if False:
        return 10
    assert SubSomething1.calledByTest1() == (42, 1)