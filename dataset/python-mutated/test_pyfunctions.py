from pyo3_pytests import pyfunctions

def none_py():
    if False:
        i = 10
        return i + 15
    return None

def test_none_py(benchmark):
    if False:
        return 10
    benchmark(none_py)

def test_none_rs(benchmark):
    if False:
        for i in range(10):
            print('nop')
    rust = benchmark(pyfunctions.none)
    py = none_py()
    assert rust == py

def simple_py(a, b=None, *, c=None):
    if False:
        for i in range(10):
            print('nop')
    return (a, b, c)

def test_simple_py(benchmark):
    if False:
        for i in range(10):
            print('nop')
    benchmark(simple_py, 1, 'foo', c={1: 2})

def test_simple_rs(benchmark):
    if False:
        while True:
            i = 10
    rust = benchmark(pyfunctions.simple, 1, 'foo', c={1: 2})
    py = simple_py(1, 'foo', c={1: 2})
    assert rust == py

def simple_args_py(a, b=None, *args, c=None):
    if False:
        while True:
            i = 10
    return (a, b, args, c)

def test_simple_args_py(benchmark):
    if False:
        return 10
    benchmark(simple_args_py, 1, 'foo', 4, 5, 6, c={1: 2})

def test_simple_args_rs(benchmark):
    if False:
        i = 10
        return i + 15
    rust = benchmark(pyfunctions.simple_args, 1, 'foo', 4, 5, 6, c={1: 2})
    py = simple_args_py(1, 'foo', 4, 5, 6, c={1: 2})
    assert rust == py

def simple_kwargs_py(a, b=None, c=None, **kwargs):
    if False:
        print('Hello World!')
    return (a, b, c, kwargs)

def test_simple_kwargs_py(benchmark):
    if False:
        for i in range(10):
            print('nop')
    benchmark(simple_kwargs_py, 1, 'foo', c={1: 2}, bar=4, foo=10)

def test_simple_kwargs_rs(benchmark):
    if False:
        return 10
    rust = benchmark(pyfunctions.simple_kwargs, 1, 'foo', c={1: 2}, bar=4, foo=10)
    py = simple_kwargs_py(1, 'foo', c={1: 2}, bar=4, foo=10)
    assert rust == py

def simple_args_kwargs_py(a, b=None, *args, c=None, **kwargs):
    if False:
        while True:
            i = 10
    return (a, b, args, c, kwargs)

def test_simple_args_kwargs_py(benchmark):
    if False:
        print('Hello World!')
    benchmark(simple_args_kwargs_py, 1, 'foo', 'baz', bar=4, foo=10)

def test_simple_args_kwargs_rs(benchmark):
    if False:
        i = 10
        return i + 15
    rust = benchmark(pyfunctions.simple_args_kwargs, 1, 'foo', 'baz', bar=4, foo=10)
    py = simple_args_kwargs_py(1, 'foo', 'baz', bar=4, foo=10)
    assert rust == py

def args_kwargs_py(*args, **kwargs):
    if False:
        print('Hello World!')
    return (args, kwargs)

def test_args_kwargs_py(benchmark):
    if False:
        print('Hello World!')
    benchmark(args_kwargs_py, 1, 'foo', {1: 2}, bar=4, foo=10)

def test_args_kwargs_rs(benchmark):
    if False:
        i = 10
        return i + 15
    rust = benchmark(pyfunctions.args_kwargs, 1, 'foo', {1: 2}, bar=4, foo=10)
    py = args_kwargs_py(1, 'foo', {1: 2}, bar=4, foo=10)
    assert rust == py