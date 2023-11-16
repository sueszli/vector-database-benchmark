from builtins import _test_sink, _test_source

def bar():
    if False:
        return 10
    return _test_source()

def qux(arg):
    if False:
        return 10
    _test_sink(arg)

def bad(ok, arg):
    if False:
        print('Hello World!')
    qux(arg)

def some_source():
    if False:
        return 10
    return bar()

def match_flows():
    if False:
        for i in range(10):
            print('nop')
    x = some_source()
    bad(5, x)