from builtins import _test_sink

def no_if(x):
    if False:
        print('Hello World!')
    _test_sink(x)

def with_if(x):
    if False:
        print('Hello World!')
    if _test_sink(x):
        pass
    pass