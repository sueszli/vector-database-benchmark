import pytest
from pytest import raises

def otherfunc(a, b):
    if False:
        i = 10
        return i + 15
    assert a == b

def somefunc(x, y):
    if False:
        for i in range(10):
            print('nop')
    otherfunc(x, y)

def otherfunc_multi(a, b):
    if False:
        while True:
            i = 10
    assert a == b

@pytest.mark.parametrize('param1, param2', [(3, 6)])
def test_generative(param1, param2):
    if False:
        print('Hello World!')
    assert param1 * 2 < param2

class TestFailing:

    def test_simple(self):
        if False:
            return 10

        def f():
            if False:
                print('Hello World!')
            return 42

        def g():
            if False:
                i = 10
                return i + 15
            return 43
        assert f() == g()

    def test_simple_multiline(self):
        if False:
            print('Hello World!')
        otherfunc_multi(42, 6 * 9)

    def test_not(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                return 10
            return 42
        assert not f()

class TestSpecialisedExplanations:

    def test_eq_text(self):
        if False:
            return 10
        assert 'spam' == 'eggs'

    def test_eq_similar_text(self):
        if False:
            i = 10
            return i + 15
        assert 'foo 1 bar' == 'foo 2 bar'

    def test_eq_multiline_text(self):
        if False:
            i = 10
            return i + 15
        assert 'foo\nspam\nbar' == 'foo\neggs\nbar'

    def test_eq_long_text(self):
        if False:
            for i in range(10):
                print('nop')
        a = '1' * 100 + 'a' + '2' * 100
        b = '1' * 100 + 'b' + '2' * 100
        assert a == b

    def test_eq_long_text_multiline(self):
        if False:
            while True:
                i = 10
        a = '1\n' * 100 + 'a' + '2\n' * 100
        b = '1\n' * 100 + 'b' + '2\n' * 100
        assert a == b

    def test_eq_list(self):
        if False:
            print('Hello World!')
        assert [0, 1, 2] == [0, 1, 3]

    def test_eq_list_long(self):
        if False:
            return 10
        a = [0] * 100 + [1] + [3] * 100
        b = [0] * 100 + [2] + [3] * 100
        assert a == b

    def test_eq_dict(self):
        if False:
            for i in range(10):
                print('nop')
        assert {'a': 0, 'b': 1, 'c': 0} == {'a': 0, 'b': 2, 'd': 0}

    def test_eq_set(self):
        if False:
            while True:
                i = 10
        assert {0, 10, 11, 12} == {0, 20, 21}

    def test_eq_longer_list(self):
        if False:
            i = 10
            return i + 15
        assert [1, 2] == [1, 2, 3]

    def test_in_list(self):
        if False:
            i = 10
            return i + 15
        assert 1 in [0, 2, 3, 4, 5]

    def test_not_in_text_multiline(self):
        if False:
            return 10
        text = 'some multiline\ntext\nwhich\nincludes foo\nand a\ntail'
        assert 'foo' not in text

    def test_not_in_text_single(self):
        if False:
            return 10
        text = 'single foo line'
        assert 'foo' not in text

    def test_not_in_text_single_long(self):
        if False:
            i = 10
            return i + 15
        text = 'head ' * 50 + 'foo ' + 'tail ' * 20
        assert 'foo' not in text

    def test_not_in_text_single_long_term(self):
        if False:
            return 10
        text = 'head ' * 50 + 'f' * 70 + 'tail ' * 20
        assert 'f' * 70 not in text

    def test_eq_dataclass(self):
        if False:
            for i in range(10):
                print('nop')
        from dataclasses import dataclass

        @dataclass
        class Foo:
            a: int
            b: str
        left = Foo(1, 'b')
        right = Foo(1, 'c')
        assert left == right

    def test_eq_attrs(self):
        if False:
            i = 10
            return i + 15
        import attr

        @attr.s
        class Foo:
            a = attr.ib()
            b = attr.ib()
        left = Foo(1, 'b')
        right = Foo(1, 'c')
        assert left == right

def test_attribute():
    if False:
        return 10

    class Foo:
        b = 1
    i = Foo()
    assert i.b == 2

def test_attribute_instance():
    if False:
        while True:
            i = 10

    class Foo:
        b = 1
    assert Foo().b == 2

def test_attribute_failure():
    if False:
        while True:
            i = 10

    class Foo:

        def _get_b(self):
            if False:
                while True:
                    i = 10
            raise Exception('Failed to get attrib')
        b = property(_get_b)
    i = Foo()
    assert i.b == 2

def test_attribute_multiple():
    if False:
        for i in range(10):
            print('nop')

    class Foo:
        b = 1

    class Bar:
        b = 2
    assert Foo().b == Bar().b

def globf(x):
    if False:
        for i in range(10):
            print('nop')
    return x + 1

class TestRaises:

    def test_raises(self):
        if False:
            print('Hello World!')
        s = 'qwe'
        raises(TypeError, int, s)

    def test_raises_doesnt(self):
        if False:
            return 10
        raises(OSError, int, '3')

    def test_raise(self):
        if False:
            return 10
        raise ValueError('demo error')

    def test_tupleerror(self):
        if False:
            return 10
        (a, b) = [1]

    def test_reinterpret_fails_with_print_for_the_fun_of_it(self):
        if False:
            print('Hello World!')
        items = [1, 2, 3]
        print(f'items is {items!r}')
        (a, b) = items.pop()

    def test_some_error(self):
        if False:
            return 10
        if namenotexi:
            pass

    def func1(self):
        if False:
            for i in range(10):
                print('nop')
        assert 41 == 42

def test_dynamic_compile_shows_nicely():
    if False:
        i = 10
        return i + 15
    import importlib.util
    import sys
    src = 'def foo():\n assert 1 == 0\n'
    name = 'abc-123'
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    code = compile(src, name, 'exec')
    exec(code, module.__dict__)
    sys.modules[name] = module
    module.foo()

class TestMoreErrors:

    def test_complex_error(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                i = 10
                return i + 15
            return 44

        def g():
            if False:
                i = 10
                return i + 15
            return 43
        somefunc(f(), g())

    def test_z1_unpack_error(self):
        if False:
            return 10
        items = []
        (a, b) = items

    def test_z2_type_error(self):
        if False:
            while True:
                i = 10
        items = 3
        (a, b) = items

    def test_startswith(self):
        if False:
            print('Hello World!')
        s = '123'
        g = '456'
        assert s.startswith(g)

    def test_startswith_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                print('Hello World!')
            return '123'

        def g():
            if False:
                while True:
                    i = 10
            return '456'
        assert f().startswith(g())

    def test_global_func(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(globf(42), float)

    def test_instance(self):
        if False:
            print('Hello World!')
        self.x = 6 * 7
        assert self.x != 42

    def test_compare(self):
        if False:
            print('Hello World!')
        assert globf(10) < 5

    def test_try_finally(self):
        if False:
            while True:
                i = 10
        x = 1
        try:
            assert x == 0
        finally:
            x = 0

class TestCustomAssertMsg:

    def test_single_line(self):
        if False:
            return 10

        class A:
            a = 1
        b = 2
        assert A.a == b, 'A.a appears not to be b'

    def test_multiline(self):
        if False:
            print('Hello World!')

        class A:
            a = 1
        b = 2
        assert A.a == b, 'A.a appears not to be b\nor does not appear to be b\none of those'

    def test_custom_repr(self):
        if False:
            for i in range(10):
                print('nop')

        class JSON:
            a = 1

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return "This is JSON\n{\n  'foo': 'bar'\n}"
        a = JSON()
        b = 2
        assert a.a == b, a