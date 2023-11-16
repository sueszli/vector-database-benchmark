from builtins import _test_sink

def d1():
    if False:
        while True:
            i = 10
    pass

def d2():
    if False:
        while True:
            i = 10
    pass

class TestC:
    pass

@d1
class TestC_1(TestC):

    def __init__(self, foo, bar, baz):
        if False:
            print('Hello World!')
        _test_sink(foo)
        _test_sink(bar)
        _test_sink(baz)

@d2
class TestC_2(TestC):

    def __init__(self, foo, bar, baz):
        if False:
            print('Hello World!')
        _test_sink(foo)
        _test_sink(bar)
        _test_sink(baz)

@d1
@d2
class TestC_3(TestC):

    def __init__(self, foo, bar, baz):
        if False:
            return 10
        _test_sink(foo)
        _test_sink(bar)
        _test_sink(baz)

def setup():
    if False:
        i = 10
        return i + 15
    TestC_1(0, 0, 0)
    TestC_1(0, 0, 0)
    TestC_1(0, 0, 0)
    TestC_2(0, 0, 0)
    TestC_2(0, 0, 0)
    TestC_2(0, 0, 0)
    TestC_3(0, 0, 0)
    TestC_3(0, 0, 0)
    TestC_3(0, 0, 0)