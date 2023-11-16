import inspect
from functools import partial
from io import StringIO
import numpy as np
import pytest
from cudf.pandas.fast_slow_proxy import _fast_arg, _FunctionProxy, _slow_arg, _transform_arg, _Unusable, make_final_proxy_type, make_intermediate_proxy_type

@pytest.fixture
def final_proxy():
    if False:
        print('Hello World!')

    class Fast:

        def __init__(self, x):
            if False:
                while True:
                    i = 10
            self.x = x

        def to_slow(self):
            if False:
                i = 10
                return i + 15
            return Slow(self.x)

        @classmethod
        def from_slow(cls, slow):
            if False:
                for i in range(10):
                    print('nop')
            return cls(slow)

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            return self.x == other.x

        def method(self):
            if False:
                print('Hello World!')
            return 'fast method'

    class Slow:

        def __init__(self, x):
            if False:
                print('Hello World!')
            self.x = x

        def __eq__(self, other):
            if False:
                print('Hello World!')
            return self.x == other.x

        def method(self):
            if False:
                print('Hello World!')
            return 'slow method'
    Pxy = make_final_proxy_type('Pxy', Fast, Slow, fast_to_slow=lambda fast: fast.to_slow(), slow_to_fast=lambda slow: Fast.from_slow(slow))
    return (Fast(1), Slow(1), Pxy(1))

@pytest.fixture
def function_proxy():
    if False:
        print('Hello World!')

    def fast_func():
        if False:
            print('Hello World!')
        '\n        Fast doc\n        '
        return 'fast func'

    def slow_func():
        if False:
            return 10
        '\n        Slow doc\n        '
        return 'slow_func'
    return (fast_func, slow_func, _FunctionProxy(fast_func, slow_func))

def test_repr_no_fast_object():
    if False:
        for i in range(10):
            print('nop')

    class Slow:

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return 'slow object'
    Pxy = make_final_proxy_type('Pxy', _Unusable, Slow, fast_to_slow=lambda fast: Slow(), slow_to_fast=lambda slow: _Unusable())
    assert repr(Pxy()) == repr(Slow())

def test_fast_slow_arg_function_basic():
    if False:
        return 10

    def func1():
        if False:
            while True:
                i = 10
        return 1
    assert _fast_arg(func1)() == _slow_arg(func1)() == 1

    def func2(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    assert _fast_arg(func2)(1, 2) == _slow_arg(func2)(1, 2) == 3

def test_fast_slow_arg_function_closure(function_proxy, final_proxy):
    if False:
        i = 10
        return i + 15
    (fast_x, slow_x, x) = function_proxy
    (fast_y, slow_y, y) = final_proxy

    def func():
        if False:
            return 10
        return (x, y.method())
    assert _slow_arg(func)() == (slow_x, slow_y.method())
    assert _fast_arg(func)() == (fast_x, fast_y.method())

def test_fast_slow_arg_function_global(monkeypatch, function_proxy, final_proxy):
    if False:
        i = 10
        return i + 15
    (fast_x, slow_x, x) = function_proxy
    (fast_y, slow_y, y) = final_proxy
    monkeypatch.setitem(globals(), '__x', x)
    monkeypatch.setitem(globals(), '__y', y)

    def func():
        if False:
            i = 10
            return i + 15
        global __x, __y
        return (__x, __y.method())
    assert _slow_arg(func)() == (slow_x, slow_y.method())
    assert _fast_arg(func)() == (fast_x, fast_y.method())

def test_fast_slow_arg_function_np():
    if False:
        return 10
    assert _slow_arg(np.mean) is np.mean
    assert _slow_arg(np.unique) is np.unique
    assert _fast_arg(np.mean) is np.mean
    assert _fast_arg(np.unique) is np.unique

def test_fast_slow_arg_builtins(function_proxy):
    if False:
        while True:
            i = 10
    (_, _, x) = function_proxy

    def func():
        if False:
            print('Hello World!')
        x
        return len([1])
    assert _slow_arg(func)() == 1
    assert _fast_arg(func)() == 1

def test_function_proxy_decorating_super_method():
    if False:
        return 10
    deco = _FunctionProxy(_Unusable(), lambda func: func)

    class Foo:

        @deco
        def method(self):
            if False:
                print('Hello World!')
            super()

@pytest.mark.xfail(reason='Mutually recursive functions are known to be handled incorrectly.')
def test_fast_slow_arg_recursion(final_proxy):
    if False:
        print('Hello World!')
    (fast_x, slow_x, x) = final_proxy

    def foo(n):
        if False:
            while True:
                i = 10
        if n <= 0:
            return x
        else:
            return bar(n - 1)

    def bar(n):
        if False:
            print('Hello World!')
        return foo(n - 1)
    assert _slow_arg(foo)(0) == slow_x
    assert _slow_arg(bar)(1) == slow_x
    assert _slow_arg(foo)(1) == slow_x
    assert _slow_arg(bar)(2) == slow_x
    assert _fast_arg(foo)(0) == fast_x
    assert _fast_arg(bar)(1) == fast_x
    assert _fast_arg(foo)(1) == fast_x
    assert _fast_arg(bar)(2) == fast_x

def test_fallback_with_stringio():
    if False:
        print('Hello World!')

    def slow(s):
        if False:
            while True:
                i = 10
        return s.read()

    def fast(s):
        if False:
            print('Hello World!')
        s.read()
        raise ValueError()
    pxy = _FunctionProxy(fast=fast, slow=slow)
    assert pxy(StringIO('hello')) == 'hello'

def test_access_class():
    if False:
        for i in range(10):
            print('nop')

    def func():
        if False:
            print('Hello World!')
        pass
    pxy = _FunctionProxy(fast=_Unusable(), slow=func)
    pxy.__class__

def test_class_attribute_error(final_proxy, function_proxy):
    if False:
        print('Hello World!')
    (_, _, x) = final_proxy
    (_, _, y) = function_proxy
    with pytest.raises(AttributeError):
        x.foo
    with pytest.raises(AttributeError):
        y.foo
    with pytest.raises(AttributeError):
        y.__abs__

def test_function_proxy_doc(function_proxy):
    if False:
        for i in range(10):
            print('nop')
    (_, slow, pxy) = function_proxy
    assert pxy.__doc__ == slow.__doc__

def test_special_methods():
    if False:
        print('Hello World!')

    class Fast:

        def __abs__(self):
            if False:
                print('Hello World!')
            pass

        def __gt__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class Slow:

        def __gt__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    Pxy = make_final_proxy_type('Pxy', Fast, Slow, fast_to_slow=lambda _: Slow(), slow_to_fast=lambda _: Fast())
    assert not hasattr(Pxy, '__abs__')
    assert not hasattr(Pxy(), '__abs__')
    assert hasattr(Pxy, '__gt__')
    assert hasattr(Pxy(), '__gt__')

@pytest.fixture(scope='module')
def fast_and_intermediate_with_doc():
    if False:
        while True:
            i = 10

    class FastIntermediate:
        """The fast intermediate docstring."""

        def method(self):
            if False:
                print('Hello World!')
            'The fast intermediate method docstring.'

    class Fast:
        """The fast docstring."""

        @property
        def prop(self):
            if False:
                print('Hello World!')
            'The fast property docstring.'

        def method(self):
            if False:
                while True:
                    i = 10
            'The fast method docstring.'

        def intermediate(self):
            if False:
                print('Hello World!')
            'The fast intermediate docstring.'
            return FastIntermediate()
    return (Fast, FastIntermediate)

@pytest.fixture(scope='module')
def slow_and_intermediate_with_doc():
    if False:
        while True:
            i = 10

    class SlowIntermediate:
        """The slow intermediate docstring."""

        def method(self):
            if False:
                return 10
            'The slow intermediate method docstring.'

    class Slow:
        """The slow docstring."""

        @property
        def prop(self):
            if False:
                while True:
                    i = 10
            'The slow property docstring.'

        def method(self):
            if False:
                print('Hello World!')
            'The slow method docstring.'

        def intermediate(self):
            if False:
                while True:
                    i = 10
            'The slow intermediate docstring.'
            return SlowIntermediate()
    return (Slow, SlowIntermediate)

def test_doc(fast_and_intermediate_with_doc, slow_and_intermediate_with_doc):
    if False:
        for i in range(10):
            print('nop')
    (Fast, FastIntermediate) = fast_and_intermediate_with_doc
    (Slow, SlowIntermediate) = slow_and_intermediate_with_doc
    Pxy = make_final_proxy_type('Pxy', Fast, Slow, fast_to_slow=lambda _: Slow(), slow_to_fast=lambda _: Fast())
    IntermediatePxy = make_intermediate_proxy_type('IntermediatePxy', FastIntermediate, SlowIntermediate)
    assert inspect.getdoc(Pxy) == inspect.getdoc(Slow)
    assert inspect.getdoc(Pxy()) == inspect.getdoc(Slow())
    assert inspect.getdoc(Pxy.prop) == inspect.getdoc(Slow.prop)
    assert inspect.getdoc(Pxy().prop) == inspect.getdoc(Slow().prop)
    assert inspect.getdoc(Pxy.method) == inspect.getdoc(Slow.method)
    assert inspect.getdoc(Pxy().method) == inspect.getdoc(Slow().method)
    assert inspect.getdoc(Pxy().intermediate()) == inspect.getdoc(Slow().intermediate())
    assert inspect.getdoc(Pxy().intermediate().method) == inspect.getdoc(Slow().intermediate().method)

def test_dir(fast_and_intermediate_with_doc, slow_and_intermediate_with_doc):
    if False:
        i = 10
        return i + 15
    (Fast, FastIntermediate) = fast_and_intermediate_with_doc
    (Slow, SlowIntermediate) = slow_and_intermediate_with_doc
    Pxy = make_final_proxy_type('Pxy', Fast, Slow, fast_to_slow=lambda _: Slow(), slow_to_fast=lambda _: Fast())
    IntermediatePxy = make_intermediate_proxy_type('IntermediatePxy', FastIntermediate, SlowIntermediate)
    assert dir(Pxy) == dir(Slow)
    assert dir(Pxy()) == dir(Slow())
    assert dir(Pxy.prop) == dir(Slow.prop)
    assert dir(Pxy().prop) == dir(Slow().prop)
    assert dir(Pxy.method) == dir(Slow.method)
    assert dir(Pxy().intermediate()) == dir(Slow().intermediate())

@pytest.mark.xfail
@pytest.mark.parametrize('check', [lambda Pxy, Slow: dir(Pxy().method) == dir(Slow().method), lambda Pxy, Slow: dir(Pxy().intermediate().method) == dir(Slow().intermediate().method)])
def test_dir_bound_method(fast_and_intermediate_with_doc, slow_and_intermediate_with_doc, check):
    if False:
        print('Hello World!')
    'This test will fail because dir for bound methods is currently\n    incorrect, but we have no way to fix it without materializing the slow\n    type, which is unnecessarily expensive.'
    (Fast, FastIntermediate) = fast_and_intermediate_with_doc
    (Slow, SlowIntermediate) = slow_and_intermediate_with_doc
    Pxy = make_final_proxy_type('Pxy', Fast, Slow, fast_to_slow=lambda _: Slow(), slow_to_fast=lambda _: Fast())
    IntermediatePxy = make_intermediate_proxy_type('IntermediatePxy', FastIntermediate, SlowIntermediate)
    assert check(Pxy, Slow)

def test_proxy_binop():
    if False:
        while True:
            i = 10

    class Foo:
        pass

    class Bar:

        def __add__(self, other):
            if False:
                while True:
                    i = 10
            if isinstance(other, Foo):
                return 'sum'
            return NotImplemented

        def __radd__(self, other):
            if False:
                print('Hello World!')
            return self.__add__(other)
    FooProxy = make_final_proxy_type('FooProxy', _Unusable, Foo, fast_to_slow=Foo(), slow_to_fast=_Unusable())
    BarProxy = make_final_proxy_type('BarProxy', _Unusable, Bar, fast_to_slow=Bar(), slow_to_fast=_Unusable())
    assert Foo() + Bar() == 'sum'
    assert Bar() + Foo() == 'sum'
    assert FooProxy() + BarProxy() == 'sum'
    assert BarProxy() + FooProxy() == 'sum'
    assert FooProxy() + Bar() == 'sum'
    assert Bar() + FooProxy() == 'sum'
    assert Foo() + BarProxy() == 'sum'
    assert BarProxy() + Foo() == 'sum'

def tuple_with_attrs(name, fields: list[str], extra_fields: set[str]):
    if False:
        return 10
    args = ', '.join(fields)
    kwargs = ', '.join(sorted(extra_fields))
    code = f'\ndef __new__(cls, {args}, *, {kwargs}):\n    return tuple.__new__(cls, ({args}, ))\n\ndef __init__(self, {args}, *, {kwargs}):\n    for key, val in zip({sorted(extra_fields)}, [{kwargs}]):\n        self.__dict__[key] = val\n\ndef __eq__(self, other):\n    return (\n        type(other) is type(self)\n        and tuple.__eq__(self, other)\n        and all(getattr(self, k) == getattr(other, k) for k in self._fields)\n    )\n\ndef __ne__(self, other):\n    return not (self == other)\n\ndef __getnewargs_ex__(self):\n    return tuple(self), self.__dict__\n'
    namespace = {'__builtins__': {'AttributeError': AttributeError, 'tuple': tuple, 'zip': zip, 'super': super, 'frozenset': frozenset, 'type': type, 'all': all, 'getattr': getattr}}
    exec(code, namespace)
    return type(name, (tuple,), {'_fields': frozenset(extra_fields), '__eq__': namespace['__eq__'], '__getnewargs_ex__': namespace['__getnewargs_ex__'], '__init__': namespace['__init__'], '__ne__': namespace['__ne__'], '__new__': namespace['__new__']})

def test_tuple_with_attrs_transform():
    if False:
        print('Hello World!')
    Bunch = tuple_with_attrs('Bunch', ['a', 'b'], {'c', 'd'})
    Bunch2 = tuple_with_attrs('Bunch', ['a', 'b'], {'c', 'd'})
    a = Bunch(1, 2, c=3, d=4)
    b = (1, 2)
    c = Bunch(1, 2, c=4, d=3)
    d = Bunch2(1, 2, c=3, d=4)
    assert a != c
    assert a != b
    assert b != c
    assert a != d
    transform = partial(_transform_arg, attribute_name='_fsproxy_fast', seen=set())
    aprime = transform(a)
    bprime = transform(b)
    cprime = transform(c)
    dprime = transform(d)
    assert a == aprime and a is not aprime
    assert b == bprime and b is not bprime
    assert c == cprime and c is not cprime
    assert d == dprime and d is not dprime