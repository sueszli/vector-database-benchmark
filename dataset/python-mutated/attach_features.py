from builtins import _test_sink, _test_source

def source():
    if False:
        for i in range(10):
            print('nop')
    return 0

def source_with_inferred():
    if False:
        while True:
            i = 10
    a = source()
    return a

def inferred_is_propagated():
    if False:
        i = 10
        return i + 15
    return source_with_inferred()

def inferred_sink(taint_left, taint_right, taint_without_feature, untainted):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(taint_left)
    _test_sink(taint_right)
    _test_sink(taint_without_feature)

def sink_is_propagated(argument):
    if False:
        i = 10
        return i + 15
    inferred_sink(argument, None, None, None)

def taint_in_taint_out(arg):
    if False:
        print('Hello World!')
    return arg

def tito_and_sink(arg):
    if False:
        print('Hello World!')
    _test_sink(arg)
    return arg

def tito_is_propagated(arg):
    if False:
        i = 10
        return i + 15
    return taint_in_taint_out(arg)

def attach_without_tito(arg):
    if False:
        for i in range(10):
            print('nop')
    return 0

def no_tito(arg):
    if False:
        i = 10
        return i + 15
    return attach_without_tito(arg)

def modeled_sink_with_optionals(a: int=0, b: int=1) -> None:
    if False:
        print('Hello World!')
    _test_sink(b)

class HasMethods:

    def method_with_optionals(self, a: int=0, b: int=1) -> None:
        if False:
            i = 10
            return i + 15
        _test_sink(b)

def attach_to_returned_sink():
    if False:
        for i in range(10):
            print('nop')
    x = _test_source()
    return x

def attach_to_returned_source():
    if False:
        i = 10
        return i + 15
    return 0

def attach_to_returned_source_2():
    if False:
        print('Hello World!')
    return 0