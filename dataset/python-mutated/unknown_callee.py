from builtins import _test_sink, _test_source

def test_issue(o):
    if False:
        while True:
            i = 10
    x = _test_source()
    y = o.method(x)
    _test_sink(y)

def test_collapse_source(o):
    if False:
        i = 10
        return i + 15
    x = {'a': _test_source()}
    y = o.method(x)
    _test_sink(y['b'])

def test_sink_collapse(arg, o):
    if False:
        print('Hello World!')
    x = o.method(arg)
    _test_sink(x['a'])

def should_collapse_depth_zero(arg, o):
    if False:
        print('Hello World!')
    return o.method(arg)

def test_collapse_depth():
    if False:
        for i in range(10):
            print('nop')
    x = {'a': _test_source()}
    y = should_collapse_depth_zero(x, 0)
    _test_sink(y['b'])