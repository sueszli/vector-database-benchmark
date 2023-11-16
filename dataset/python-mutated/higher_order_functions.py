from builtins import _test_sink, _test_source

def goes_to_sink(arg):
    if False:
        print('Hello World!')
    _test_sink(arg)

def has_tito(arg):
    if False:
        i = 10
        return i + 15
    return arg

def higher_order_function(f, arg):
    if False:
        for i in range(10):
            print('nop')
    f(arg)

def test_higher_order_function():
    if False:
        while True:
            i = 10
    higher_order_function(goes_to_sink, _test_source())

class C:

    def method_to_sink(self, arg):
        if False:
            i = 10
            return i + 15
        _test_sink(arg)

    def self_to_sink(self):
        if False:
            print('Hello World!')
        _test_sink(self)

def higher_order_method(c: C, arg):
    if False:
        return 10
    higher_order_function(c.method_to_sink, arg)

def test_higher_order_method():
    if False:
        for i in range(10):
            print('nop')
    higher_order_method(C(), _test_source())

def test_higher_order_method_self():
    if False:
        i = 10
        return i + 15
    c: C = _test_source()
    higher_order_function(c.self_to_sink)

def higher_order_function_and_sink(f, arg):
    if False:
        i = 10
        return i + 15
    f(arg)
    _test_sink(arg)

def test_higher_order_function_and_sink():
    if False:
        i = 10
        return i + 15
    higher_order_function_and_sink(goes_to_sink, _test_source())

def test_higher_order_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return higher_order_function(has_tito, x)

def apply(f, x):
    if False:
        print('Hello World!')
    return f(x)

def test_apply_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return apply(has_tito, x)

def source_through_tito():
    if False:
        return 10
    x = _test_source()
    y = apply(has_tito, x)
    return y

def test_apply_source():
    if False:
        while True:
            i = 10
    return apply(_test_source, 0)

class Callable:

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return

def callable_class():
    if False:
        i = 10
        return i + 15
    c = Callable(_test_source())
    _test_sink(c)

def sink_args(*args):
    if False:
        for i in range(10):
            print('nop')
    for arg in args:
        _test_sink(arg)

def test_location(x: int, y: Callable, z: int):
    if False:
        while True:
            i = 10
    sink_args(x, y, z)

def conditional_apply(f, g, cond: bool, x: int):
    if False:
        while True:
            i = 10
    if cond:
        return f(x)
    else:
        return g(x)

def safe():
    if False:
        print('Hello World!')
    return 0

def test_conditional_apply_forward():
    if False:
        return 10
    _test_sink(conditional_apply(_test_source, safe, True, 0))
    _test_sink(conditional_apply(_test_source, safe, False, 0))
    _test_sink(conditional_apply(safe, _test_source, True, 0))
    _test_sink(conditional_apply(safe, _test_source, False, 0))

def test_conditional_apply_backward(x):
    if False:
        print('Hello World!')
    conditional_apply(_test_sink, safe, True, x)
    conditional_apply(_test_sink, safe, False, x)
    conditional_apply(safe, _test_sink, True, x)
    conditional_apply(safe, _test_sink, False, x)