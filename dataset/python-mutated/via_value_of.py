import enum
from builtins import _test_sink, _test_source

def return_via_parameter_name(parameter=None):
    if False:
        for i in range(10):
            print('nop')
    return 0

class MyEnum(enum.Enum):
    FOO = 1

def test_string_literals():
    if False:
        i = 10
        return i + 15
    return return_via_parameter_name('A')

def test_numerals():
    if False:
        return 10
    return return_via_parameter_name(1)

def test_bool():
    if False:
        i = 10
        return i + 15
    return return_via_parameter_name(False)

def test_enums():
    if False:
        print('Hello World!')
    return return_via_parameter_name(MyEnum.FOO)

def test_missing():
    if False:
        for i in range(10):
            print('nop')
    return return_via_parameter_name()

def meta(parameter):
    if False:
        print('Hello World!')
    return return_via_parameter_name(parameter)

def meta_named(parameter):
    if False:
        i = 10
        return i + 15
    return return_via_parameter_name(parameter=parameter)

def test_via_value_of_does_not_propagate():
    if False:
        return 10
    return meta('Name')

def tito(parameter, other):
    if False:
        i = 10
        return i + 15
    pass

def test_tito():
    if False:
        while True:
            i = 10
    a = tito(_test_source(), 'second')
    return a

def sink_via_value_of(x, y):
    if False:
        return 10
    pass

def test_sink(element):
    if False:
        print('Hello World!')
    return sink_via_value_of(element, 'second')

def test_backwards_tito(parameter):
    if False:
        while True:
            i = 10
    return tito(parameter, 'by_backwards')

def meta_sink(parameter, value):
    if False:
        i = 10
        return i + 15
    sink_via_value_of(parameter, value)

def meta_sink_args(parameter, value):
    if False:
        i = 10
        return i + 15
    sink_via_value_of(*[parameter, value])

def meta_sink_kwargs(parameter, value):
    if False:
        i = 10
        return i + 15
    sink_via_value_of(**{'x': parameter, 'y': value})

def meta_sink_positional_kwargs(parameter, value):
    if False:
        print('Hello World!')
    sink_via_value_of('x', **{'y': value})

def test_sinks_do_not_propagate(parameter):
    if False:
        for i in range(10):
            print('nop')
    meta_sink(parameter, 'not a feature')

def attach_to_source(parameter):
    if False:
        while True:
            i = 10
    return _test_source()

def test_attach_to_source():
    if False:
        for i in range(10):
            print('nop')
    return attach_to_source('attached to source')

def attach_to_sink(parameter, feature):
    if False:
        i = 10
        return i + 15
    _test_sink(parameter)

def test_attach_to_sink(parameter):
    if False:
        print('Hello World!')
    attach_to_sink(parameter, 'attached to sink')

def return_including_name(parameter):
    if False:
        while True:
            i = 10
    return 0

def test_return_including_name():
    if False:
        print('Hello World!')
    return return_including_name('parameter_value')

def return_via_second_parameter(first, second, third=3, fourth=4, fifth=5):
    if False:
        return 10
    return 0

def test_return_second_parameter():
    if False:
        return 10
    return return_via_second_parameter(1, 2)

def test_return_second_parameter_keyword():
    if False:
        print('Hello World!')
    return return_via_second_parameter(second=2, first=1)

def test_args_parameter():
    if False:
        return 10
    args = ['first', 'second']
    return return_via_second_parameter(*args)

def test_kwargs_parameter():
    if False:
        return 10
    kwargs = {'first': '1', 'second': '2'}
    return return_via_second_parameter(**kwargs)

def test_args_kwargs_parameter():
    if False:
        while True:
            i = 10
    args = ['1']
    kwargs = {'second': '2'}
    return return_via_second_parameter(*args, **kwargs)

def test_positional_kwargs_parameter():
    if False:
        return 10
    kwargs = {'second': '2'}
    return return_via_second_parameter('1', **kwargs)

def test_named_kwargs_parameter():
    if False:
        return 10
    kwargs = {'first': '1'}
    return return_via_second_parameter(**kwargs, second='2')

def test_unknown_named_args(b, e):
    if False:
        return 10
    args = [e]
    return return_via_second_parameter(*args, second=b)

def test_unknown_named_kwargs(b, e):
    if False:
        i = 10
        return i + 15
    kwargs = {'fifth': e}
    return return_via_second_parameter(**kwargs, second=b)

def test_unknown_positional_args(a, b, c):
    if False:
        i = 10
        return i + 15
    args = [c]
    return return_via_second_parameter(a, b, *args)

def test_unknown_positional_kwargs(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    kwargs = {'third': c}
    return return_via_second_parameter(a, b, **kwargs)

def test_unknown_positional_named_args1(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    args = [c]
    return return_via_second_parameter(a, *args, second=b)

def test_unknown_positional_named_args2(a, b, c, d):
    if False:
        i = 10
        return i + 15
    args = [d]
    return return_via_second_parameter(a, c, *args, second=b)

def test_unknown_positional_named_kwargs1(a, b, c):
    if False:
        return 10
    kwargs = {'third': c}
    return return_via_second_parameter(a, **kwargs, second=b)

def test_unknown_positional_named_kwargs2(a, b, c, d):
    if False:
        return 10
    kwargs = {'fourth': d}
    return return_via_second_parameter(a, c, **kwargs, second=b)

def test_unknown_named_args_kwargs(a, b, c):
    if False:
        return 10
    args = [a]
    kwargs = {'third': c}
    return return_via_second_parameter(*args, **kwargs, second=b)

def test_unknown_positional_named_args_kwargs1(a, b, c, d, e):
    if False:
        for i in range(10):
            print('nop')
    args = [d]
    kwargs = {'fifth': e}
    return return_via_second_parameter(a, *args, **kwargs, second=b)