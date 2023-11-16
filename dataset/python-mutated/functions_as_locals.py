from builtins import _test_sink, _test_source

def foo(arg):
    if False:
        print('Hello World!')
    _test_sink(arg)

def foo_as_local():
    if False:
        print('Hello World!')
    x = _test_source()
    f = foo
    foo(x)
    f(x)

def local_tito(arg):
    if False:
        i = 10
        return i + 15
    f = foo
    f(arg)

class C:

    def m(self, arg):
        if False:
            i = 10
            return i + 15
        _test_sink(arg)

def local_function_with_method_sink(c: C):
    if False:
        return 10
    f = c.m
    x = _test_source()
    c.m(x)
    f(x)

def method_tito(c: C, arg):
    if False:
        i = 10
        return i + 15
    f = c.m
    f(arg)

def barA(arg1: str, arg2: str):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(arg1)

def barB(arg1: str, arg2: int):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(arg2)

def a_or_b():
    if False:
        while True:
            i = 10
    if 1 > 2:
        f = barA
    else:
        f = barB
    f(_test_source(), 0)
    f(0, _test_source())