from builtins import _test_sink, _test_source

def named_sink(x):
    if False:
        return 10
    _test_sink(x)

def locals_to_sink():
    if False:
        print('Hello World!')
    _test_sink(locals()['x'])
    x = _test_source()
    _test_sink(locals()['x'])
    _test_sink(locals()['y'])
    named_sink(**locals())

def source_parameter_to_sink(x, y):
    if False:
        i = 10
        return i + 15
    _test_sink(locals()['x'])
    _test_sink(locals()['y'])