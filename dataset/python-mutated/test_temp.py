import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
WINDOWS = sys.platform[:3] == 'win'
f = lambda x: x ** 2
x = [1, 2, 3, 4, 5]

def test_code_to_tempfile():
    if False:
        i = 10
        return i + 15
    if not WINDOWS:
        pyfile = dump_source(f, alias='_f')
        _f = load_source(pyfile)
        assert _f(4) == f(4)

def test_code_to_stream():
    if False:
        i = 10
        return i + 15
    pyfile = dumpIO_source(f, alias='_f')
    _f = loadIO_source(pyfile)
    assert _f(4) == f(4)

def test_pickle_to_tempfile():
    if False:
        return 10
    if not WINDOWS:
        dumpfile = dump(x)
        _x = load(dumpfile)
        assert _x == x

def test_pickle_to_stream():
    if False:
        while True:
            i = 10
    dumpfile = dumpIO(x)
    _x = loadIO(dumpfile)
    assert _x == x
f = lambda x: x ** 2

def g(x):
    if False:
        i = 10
        return i + 15
    return f(x) - x

def h(x):
    if False:
        print('Hello World!')

    def g(x):
        if False:
            i = 10
            return i + 15
        return x
    return g(x) - x

class Foo(object):

    def bar(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x * x + x
_foo = Foo()

def add(x, y):
    if False:
        return 10
    return x + y
squared = lambda x: x ** 2

class Bar:
    pass
_bar = Bar()

def test_two_arg_functions():
    if False:
        i = 10
        return i + 15
    for obj in [add]:
        pyfile = dumpIO_source(obj, alias='_obj')
        _obj = loadIO_source(pyfile)
        assert _obj(4, 2) == obj(4, 2)

def test_one_arg_functions():
    if False:
        print('Hello World!')
    for obj in [g, h, squared]:
        pyfile = dumpIO_source(obj, alias='_obj')
        _obj = loadIO_source(pyfile)
        assert _obj(4) == obj(4)

def test_the_rest():
    if False:
        return 10
    for obj in [Bar, Foo, Foo.bar, _foo.bar]:
        pyfile = dumpIO_source(obj, alias='_obj')
        _obj = loadIO_source(pyfile)
        assert _obj.__name__ == obj.__name__
if __name__ == '__main__':
    test_code_to_tempfile()
    test_code_to_stream()
    test_pickle_to_tempfile()
    test_pickle_to_stream()
    test_two_arg_functions()
    test_one_arg_functions()
    test_the_rest()