from builtins import _test_sink, _test_source

def source_field():
    if False:
        return 10
    result = {}
    result.a = _test_source()
    return result

def sink_field(arg):
    if False:
        while True:
            i = 10
    _test_sink(arg.a)

def match_flows():
    if False:
        while True:
            i = 10
    x = source_field()
    sink_field(x)

def star_arg(x, *data, **kwargs):
    if False:
        print('Hello World!')
    sink_field(data[1])

def star_arg_wrapper(x, *data, **kwargs):
    if False:
        i = 10
        return i + 15
    star_arg(x, *data, **kwargs)

def match_star_arg_with_star():
    if False:
        print('Hello World!')
    data = [0, source_field(), 2]
    star_arg_wrapper('a', *data)

def match_star_arg_directly():
    if False:
        while True:
            i = 10
    star_arg_wrapper('a', 'b', source_field(), 'd')

def star_star_arg(x, **kwargs):
    if False:
        i = 10
        return i + 15
    sink_field(kwargs['arg'])

def star_star_arg_wrapper(x, **kwargs):
    if False:
        print('Hello World!')
    star_star_arg(x, **kwargs)

def match_star_star_arg_with_star():
    if False:
        for i in range(10):
            print('nop')
    data = {'a': 0, 'arg': source_field()}
    star_star_arg_wrapper('a', **data)

def match_star_star_arg_directly():
    if False:
        i = 10
        return i + 15
    star_star_arg_wrapper('a', 'b', arg=source_field())

class Foo:

    @property
    def some_source():
        if False:
            for i in range(10):
                print('nop')
        return _test_source()

def refer_to_method_as_field(foo: Foo):
    if False:
        for i in range(10):
            print('nop')
    taint = foo.some_source
    _test_sink(taint)