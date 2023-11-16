from dill.detect import baditems, badobjects, badtypes, errors, parent, at, globalvars
from dill import settings
from dill._dill import IS_PYPY
from pickle import PicklingError
import inspect

def test_bad_things():
    if False:
        for i in range(10):
            print('nop')
    f = inspect.currentframe()
    assert baditems(f) == [f]
    assert badobjects(f) is f
    assert badtypes(f) == type(f)
    assert type(errors(f)) is PicklingError if IS_PYPY else TypeError
    d = badtypes(f, 1)
    assert isinstance(d, dict)
    assert list(badobjects(f, 1).keys()) == list(d.keys())
    assert list(errors(f, 1).keys()) == list(d.keys())
    s = set([(err.__class__.__name__, err.args[0]) for err in list(errors(f, 1).values())])
    a = dict(s)
    assert len(s) is len(a)
    n = 1 if IS_PYPY else 2
    assert len(a) is n if 'PicklingError' in a.keys() else n - 1

def test_parent():
    if False:
        while True:
            i = 10
    x = [4, 5, 6, 7]
    listiter = iter(x)
    obj = parent(listiter, list)
    assert obj is x
    if IS_PYPY:
        assert parent(obj, int) is None
    else:
        assert parent(obj, int) is x[-1]
    assert at(id(at)) is at
(a, b, c) = (1, 2, 3)

def squared(x):
    if False:
        for i in range(10):
            print('nop')
    return a + x ** 2

def foo(x):
    if False:
        print('Hello World!')

    def bar(y):
        if False:
            i = 10
            return i + 15
        return squared(x) + y
    return bar

class _class:

    def _method(self):
        if False:
            return 10
        pass

    def ok(self):
        if False:
            while True:
                i = 10
        return True

def test_globals():
    if False:
        print('Hello World!')

    def f():
        if False:
            for i in range(10):
                print('nop')
        a

        def g():
            if False:
                i = 10
                return i + 15
            b

            def h():
                if False:
                    i = 10
                    return i + 15
                c
    assert globalvars(f) == dict(a=1, b=2, c=3)
    res = globalvars(foo, recurse=True)
    assert set(res) == set(['squared', 'a'])
    res = globalvars(foo, recurse=False)
    assert res == {}
    zap = foo(2)
    res = globalvars(zap, recurse=True)
    assert set(res) == set(['squared', 'a'])
    res = globalvars(zap, recurse=False)
    assert set(res) == set(['squared'])
    del zap
    res = globalvars(squared)
    assert set(res) == set(['a'])
bar = [0]

class Foo(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __getstate__(self):
        if False:
            return 10
        bar[0] = bar[0] + 1
        return {}

    def __setstate__(self, data):
        if False:
            return 10
        pass
f = Foo()

def test_getstate():
    if False:
        return 10
    from dill import dumps, loads
    dumps(f)
    b = bar[0]
    dumps(lambda : f, recurse=False)
    assert bar[0] == b
    dumps(lambda : f, recurse=True)
    assert bar[0] == b + 1

def test_deleted():
    if False:
        for i in range(10):
            print('nop')
    from dill import dumps, loads
    from math import sin, pi
    global sin

    def sinc(x):
        if False:
            i = 10
            return i + 15
        return sin(x) / x
    settings['recurse'] = True
    _sinc = dumps(sinc)
    sin = globals().pop('sin')
    sin = 1
    del sin
    sinc_ = loads(_sinc)
    res = sinc_(1)
    from math import sin
    assert sinc(1) == res

def test_lambdify():
    if False:
        i = 10
        return i + 15
    try:
        from sympy import symbols, lambdify
    except ImportError:
        return
    settings['recurse'] = True
    x = symbols('x')
    y = x ** 2
    f = lambdify([x], y)
    z = min
    d = globals()
    globalvars(f, recurse=True, builtin=True)
    assert z is min
    assert d is globals()
if __name__ == '__main__':
    test_bad_things()
    test_parent()
    test_globals()
    test_getstate()
    test_deleted()
    test_lambdify()