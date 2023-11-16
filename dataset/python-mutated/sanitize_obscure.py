from builtins import _test_sink, _test_source
from typing import TypeVar
T = TypeVar('T')

def sanitize_all(x: T) -> T:
    if False:
        while True:
            i = 10
    ...

def test1(x):
    if False:
        while True:
            i = 10
    y = sanitize_all(x)
    _test_sink(y)

def test2():
    if False:
        print('Hello World!')
    x = _test_source()
    y = sanitize_all(x)
    _test_sink(y)

def sanitize_tito(x: T) -> T:
    if False:
        return 10
    ...

def test3(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_tito(x)
    _test_sink(y)

def test4():
    if False:
        i = 10
        return i + 15
    x = _test_source()
    y = sanitize_tito(x)
    _test_sink(y)

def a_source():
    if False:
        while True:
            i = 10
    return

def b_source():
    if False:
        while True:
            i = 10
    return

def a_sink(x):
    if False:
        print('Hello World!')
    return

def b_sink(x):
    if False:
        i = 10
        return i + 15
    return

def sanitize_a_tito(x):
    if False:
        while True:
            i = 10
    ...

def no_propagation_with_sanitize_a_tito():
    if False:
        return 10
    a = a_source()
    b = sanitize_a_tito(a)
    return b

def propagation_of_b_with_sanitize_a_tito():
    if False:
        return 10
    b = b_source()
    tito = sanitize_a_tito(b)
    return tito

def sanitize_a_sink_tito(x):
    if False:
        for i in range(10):
            print('nop')
    ...

def no_propagation_of_a_sink(x):
    if False:
        return 10
    y = sanitize_a_sink_tito(x)
    a_sink(y)

def propagation_of_b_sink(x):
    if False:
        while True:
            i = 10
    y = sanitize_a_sink_tito(x)
    b_sink(y)

def sanitize_a_source_tito(x):
    if False:
        i = 10
        return i + 15
    ...

def no_propagation_of_a_source():
    if False:
        for i in range(10):
            print('nop')
    x = a_source()
    return sanitize_a_source_tito(x)

def propagation_of_b_source():
    if False:
        print('Hello World!')
    x = b_source()
    return sanitize_a_source_tito(x)

def sanitize_parameter_source_a_tito(x, y):
    if False:
        for i in range(10):
            print('nop')
    ...

def no_propagation_of_a_source_via_parameter_tito():
    if False:
        return 10
    x = a_source()
    return sanitize_parameter_source_a_tito(x)

def propagation_of_a_source_via_other_parameter_tito():
    if False:
        print('Hello World!')
    x = a_source()
    return sanitize_parameter_source_a_tito(y=x, x='foo')

def propagation_of_b_source_via_parameter_tito():
    if False:
        for i in range(10):
            print('nop')
    x = b_source()
    return sanitize_parameter_source_a_tito(x)

def sanitize_parameter_sink_a_tito(x, y):
    if False:
        while True:
            i = 10
    ...

def no_propagation_of_a_sink_via_parameter_tito(x):
    if False:
        while True:
            i = 10
    y = sanitize_parameter_sink_a_tito(x)
    a_sink(y)

def propagation_of_a_sink_via_other_parameter_tito(x):
    if False:
        while True:
            i = 10
    y = sanitize_parameter_sink_a_tito('foo', x)
    a_sink(y)

def propagation_of_b_sink_via_parameter_tito(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_parameter_sink_a_tito(x)
    b_sink(y)

def sanitize_return_source_a_tito(x):
    if False:
        return 10
    ...

def no_propagation_of_a_source_via_return_tito():
    if False:
        for i in range(10):
            print('nop')
    x = a_source()
    return sanitize_return_source_a_tito(x)

def propagation_of_b_source_via_return_tito():
    if False:
        i = 10
        return i + 15
    x = b_source()
    return sanitize_return_source_a_tito(x)

def sanitize_return_sink_a_tito(x):
    if False:
        while True:
            i = 10
    ...

def no_propagation_of_a_sink_via_return_tito(x):
    if False:
        for i in range(10):
            print('nop')
    y = sanitize_return_sink_a_tito(x)
    a_sink(y)

def propagation_of_b_sink_via_return_tito(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_return_sink_a_tito(x)
    b_sink(y)

def sanitize_parameter_source_a(x):
    if False:
        while True:
            i = 10
    ...

def no_propagation_of_a_source_via_parameter():
    if False:
        while True:
            i = 10
    x = a_source()
    return sanitize_parameter_source_a(x)

def propagation_of_b_source_via_parameter():
    if False:
        while True:
            i = 10
    x = b_source()
    return sanitize_parameter_source_a(x)

def sanitize_parameter_sink_a(x):
    if False:
        print('Hello World!')
    ...

def no_propagation_of_a_sink_via_parameter(x):
    if False:
        while True:
            i = 10
    y = sanitize_parameter_sink_a(x)
    a_sink(y)

def propagation_of_b_sink_via_parameter(x):
    if False:
        for i in range(10):
            print('nop')
    y = sanitize_parameter_sink_a(x)
    b_sink(y)

def sanitize_return_source_a(x):
    if False:
        return 10
    ...

def no_propagation_of_a_source_via_return():
    if False:
        for i in range(10):
            print('nop')
    x = a_source()
    return sanitize_return_source_a(x)

def propagation_of_b_source_via_return():
    if False:
        print('Hello World!')
    x = b_source()
    return sanitize_return_source_a(x)

def sanitize_return_sink_a(x):
    if False:
        while True:
            i = 10
    ...

def no_propagation_of_a_sink_via_return(x):
    if False:
        print('Hello World!')
    y = sanitize_return_sink_a(x)
    a_sink(y)

def propagation_of_b_sink_via_return(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_return_sink_a(x)
    b_sink(y)

def sanitize_obscure_single_argument(x, y):
    if False:
        for i in range(10):
            print('nop')
    ...

def sanitize_obscure_single_argument_tito(x, y):
    if False:
        while True:
            i = 10
    return sanitize_obscure_single_argument(x, y)