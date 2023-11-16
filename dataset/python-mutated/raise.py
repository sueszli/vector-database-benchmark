from builtins import _test_sink, _test_source

def forward():
    if False:
        print('Hello World!')
    raise _test_sink(_test_source())

def backward(x):
    if False:
        return 10
    raise _test_sink(x)

def unreachable():
    if False:
        for i in range(10):
            print('nop')
    x = _test_source()
    raise Exception()
    _test_sink(x)

def unreachable_through_function_call_sink():
    if False:
        print('Hello World!')
    x = _test_source()
    no_sink(x)

def no_sink(x):
    if False:
        for i in range(10):
            print('nop')
    raise Exception()
    _test_sink(x)

def no_source():
    if False:
        i = 10
        return i + 15
    raise Exception()
    return _test_source()

def unreachable_through_function_call_source():
    if False:
        return 10
    x = no_source()
    _test_sink(x)

def unreachable_code_do_to_always_exception():
    if False:
        return 10
    no_source()
    y = _test_source()
    _test_sink(y)

def conditional_unreachability(y):
    if False:
        for i in range(10):
            print('nop')
    if y:
        x = _test_source()
        raise Exception()
    else:
        x = 'benign'
    _test_sink(x)