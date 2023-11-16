from builtins import _test_sink, _test_source
x = _test_source()
_test_sink(x)

def foo(x):
    if False:
        print('Hello World!')
    _test_sink(x)
y = _test_source()
foo(y)