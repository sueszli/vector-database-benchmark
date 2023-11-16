"""Unit tests for the positional only argument syntax specified in PEP 570."""
import dis
import pickle
import unittest
from test.support import check_syntax_error

def global_pos_only_f(a, b, /):
    if False:
        while True:
            i = 10
    return (a, b)

def global_pos_only_and_normal(a, /, b):
    if False:
        for i in range(10):
            print('nop')
    return (a, b)

def global_pos_only_defaults(a=1, /, b=2):
    if False:
        print('Hello World!')
    return (a, b)

class PositionalOnlyTestCase(unittest.TestCase):

    def assertRaisesSyntaxError(self, codestr, regex='invalid syntax'):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(SyntaxError, regex):
            compile(codestr + '\n', '<test>', 'single')

    def test_invalid_syntax_errors(self):
        if False:
            return 10
        check_syntax_error(self, 'def f(a, b = 5, /, c): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'def f(a = 5, b, /, c): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'def f(a = 5, b=1, /, c, *, d=2): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'def f(a = 5, b, /): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'def f(*args, /): pass')
        check_syntax_error(self, 'def f(*args, a, /): pass')
        check_syntax_error(self, 'def f(**kwargs, /): pass')
        check_syntax_error(self, 'def f(/, a = 1): pass')
        check_syntax_error(self, 'def f(/, a): pass')
        check_syntax_error(self, 'def f(/): pass')
        check_syntax_error(self, 'def f(*, a, /): pass')
        check_syntax_error(self, 'def f(*, /, a): pass')
        check_syntax_error(self, 'def f(a, /, a): pass', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'def f(a, /, *, a): pass', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'def f(a, b/2, c): pass')
        check_syntax_error(self, 'def f(a, /, c, /): pass')
        check_syntax_error(self, 'def f(a, /, c, /, d): pass')
        check_syntax_error(self, 'def f(a, /, c, /, d, *, e): pass')
        check_syntax_error(self, 'def f(a, *, c, /, d, e): pass')

    def test_invalid_syntax_errors_async(self):
        if False:
            return 10
        check_syntax_error(self, 'async def f(a, b = 5, /, c): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'async def f(a = 5, b, /, c): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'async def f(a = 5, b=1, /, c, d=2): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'async def f(a = 5, b, /): pass', 'non-default argument follows default argument')
        check_syntax_error(self, 'async def f(*args, /): pass')
        check_syntax_error(self, 'async def f(*args, a, /): pass')
        check_syntax_error(self, 'async def f(**kwargs, /): pass')
        check_syntax_error(self, 'async def f(/, a = 1): pass')
        check_syntax_error(self, 'async def f(/, a): pass')
        check_syntax_error(self, 'async def f(/): pass')
        check_syntax_error(self, 'async def f(*, a, /): pass')
        check_syntax_error(self, 'async def f(*, /, a): pass')
        check_syntax_error(self, 'async def f(a, /, a): pass', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'async def f(a, /, *, a): pass', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'async def f(a, b/2, c): pass')
        check_syntax_error(self, 'async def f(a, /, c, /): pass')
        check_syntax_error(self, 'async def f(a, /, c, /, d): pass')
        check_syntax_error(self, 'async def f(a, /, c, /, d, *, e): pass')
        check_syntax_error(self, 'async def f(a, *, c, /, d, e): pass')

    def test_optional_positional_only_args(self):
        if False:
            i = 10
            return i + 15

        def f(a, b=10, /, c=100):
            if False:
                print('Hello World!')
            return a + b + c
        self.assertEqual(f(1, 2, 3), 6)
        self.assertEqual(f(1, 2, c=3), 6)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'b'"):
            f(1, b=2, c=3)
        self.assertEqual(f(1, 2), 103)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'b'"):
            f(1, b=2)
        self.assertEqual(f(1, c=2), 13)

        def f(a=1, b=10, /, c=100):
            if False:
                for i in range(10):
                    print('nop')
            return a + b + c
        self.assertEqual(f(1, 2, 3), 6)
        self.assertEqual(f(1, 2, c=3), 6)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'b'"):
            f(1, b=2, c=3)
        self.assertEqual(f(1, 2), 103)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'b'"):
            f(1, b=2)
        self.assertEqual(f(1, c=2), 13)

    def test_syntax_for_many_positional_only(self):
        if False:
            print('Hello World!')
        fundef = 'def f(%s, /):\n  pass\n' % ', '.join(('i%d' % i for i in range(300)))
        compile(fundef, '<test>', 'single')

    def test_pos_only_definition(self):
        if False:
            i = 10
            return i + 15

        def f(a, b, c, /, d, e=1, *, f, g=2):
            if False:
                return 10
            pass
        self.assertEqual(5, f.__code__.co_argcount)
        self.assertEqual(3, f.__code__.co_posonlyargcount)
        self.assertEqual((1,), f.__defaults__)

        def f(a, b, c=1, /, d=2, e=3, *, f, g=4):
            if False:
                print('Hello World!')
            pass
        self.assertEqual(5, f.__code__.co_argcount)
        self.assertEqual(3, f.__code__.co_posonlyargcount)
        self.assertEqual((1, 2, 3), f.__defaults__)

    def test_pos_only_call_via_unpacking(self):
        if False:
            while True:
                i = 10

        def f(a, b, /):
            if False:
                i = 10
                return i + 15
            return a + b
        self.assertEqual(f(*[1, 2]), 3)

    def test_use_positional_as_keyword(self):
        if False:
            i = 10
            return i + 15

        def f(a, /):
            if False:
                i = 10
                return i + 15
            pass
        expected = "f\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"
        with self.assertRaisesRegex(TypeError, expected):
            f(a=1)

        def f(a, /, b):
            if False:
                return 10
            pass
        expected = "f\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"
        with self.assertRaisesRegex(TypeError, expected):
            f(a=1, b=2)

        def f(a, b, /):
            if False:
                i = 10
                return i + 15
            pass
        expected = "f\\(\\) got some positional-only arguments passed as keyword arguments: 'a, b'"
        with self.assertRaisesRegex(TypeError, expected):
            f(a=1, b=2)

    def test_positional_only_and_arg_invalid_calls(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, /, c):
            if False:
                return 10
            pass
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'c'"):
            f(1, 2)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 2 required positional arguments: 'b' and 'c'"):
            f(1)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 3 required positional arguments: 'a', 'b', and 'c'"):
            f()
        with self.assertRaisesRegex(TypeError, 'f\\(\\) takes 3 positional arguments but 4 were given'):
            f(1, 2, 3, 4)

    def test_positional_only_and_optional_arg_invalid_calls(self):
        if False:
            return 10

        def f(a, b, /, c=3):
            if False:
                return 10
            pass
        f(1, 2)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'b'"):
            f(1)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 2 required positional arguments: 'a' and 'b'"):
            f()
        with self.assertRaisesRegex(TypeError, 'f\\(\\) takes from 2 to 3 positional arguments but 4 were given'):
            f(1, 2, 3, 4)

    def test_positional_only_and_kwonlyargs_invalid_calls(self):
        if False:
            while True:
                i = 10

        def f(a, b, /, c, *, d, e):
            if False:
                while True:
                    i = 10
            pass
        f(1, 2, 3, d=1, e=2)
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument: 'd'"):
            f(1, 2, 3, e=2)
        with self.assertRaisesRegex(TypeError, "missing 2 required keyword-only arguments: 'd' and 'e'"):
            f(1, 2, 3)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'c'"):
            f(1, 2)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 2 required positional arguments: 'b' and 'c'"):
            f(1)
        with self.assertRaisesRegex(TypeError, " missing 3 required positional arguments: 'a', 'b', and 'c'"):
            f()
        with self.assertRaisesRegex(TypeError, 'f\\(\\) takes 3 positional arguments but 6 positional arguments \\(and 2 keyword-only arguments\\) were given'):
            f(1, 2, 3, 4, 5, 6, d=7, e=8)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got an unexpected keyword argument 'f'"):
            f(1, 2, 3, d=1, e=4, f=56)

    def test_positional_only_invalid_calls(self):
        if False:
            print('Hello World!')

        def f(a, b, /):
            if False:
                print('Hello World!')
            pass
        f(1, 2)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'b'"):
            f(1)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 2 required positional arguments: 'a' and 'b'"):
            f()
        with self.assertRaisesRegex(TypeError, 'f\\(\\) takes 2 positional arguments but 3 were given'):
            f(1, 2, 3)

    def test_positional_only_with_optional_invalid_calls(self):
        if False:
            print('Hello World!')

        def f(a, b=2, /):
            if False:
                for i in range(10):
                    print('nop')
            pass
        f(1)
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'a'"):
            f()
        with self.assertRaisesRegex(TypeError, 'f\\(\\) takes from 1 to 2 positional arguments but 3 were given'):
            f(1, 2, 3)

    def test_no_standard_args_usage(self):
        if False:
            return 10

        def f(a, b, /, *, c):
            if False:
                for i in range(10):
                    print('nop')
            pass
        f(1, 2, c=3)
        with self.assertRaises(TypeError):
            f(1, b=2, c=3)

    def test_change_default_pos_only(self):
        if False:
            while True:
                i = 10

        def f(a, b=2, /, c=3):
            if False:
                i = 10
                return i + 15
            return a + b + c
        self.assertEqual((2, 3), f.__defaults__)
        f.__defaults__ = (1, 2, 3)
        self.assertEqual(f(1, 2, 3), 6)

    def test_lambdas(self):
        if False:
            while True:
                i = 10
        x = lambda a, /, b: a + b
        self.assertEqual(x(1, 2), 3)
        self.assertEqual(x(1, b=2), 3)
        x = lambda a, /, b=2: a + b
        self.assertEqual(x(1), 3)
        x = lambda a, b, /: a + b
        self.assertEqual(x(1, 2), 3)
        x = lambda a, b, /: a + b
        self.assertEqual(x(1, 2), 3)

    def test_invalid_syntax_lambda(self):
        if False:
            while True:
                i = 10
        check_syntax_error(self, 'lambda a, b = 5, /, c: None', 'non-default argument follows default argument')
        check_syntax_error(self, 'lambda a = 5, b, /, c: None', 'non-default argument follows default argument')
        check_syntax_error(self, 'lambda a = 5, b, /: None', 'non-default argument follows default argument')
        check_syntax_error(self, 'lambda *args, /: None')
        check_syntax_error(self, 'lambda *args, a, /: None')
        check_syntax_error(self, 'lambda **kwargs, /: None')
        check_syntax_error(self, 'lambda /, a = 1: None')
        check_syntax_error(self, 'lambda /, a: None')
        check_syntax_error(self, 'lambda /: None')
        check_syntax_error(self, 'lambda *, a, /: None')
        check_syntax_error(self, 'lambda *, /, a: None')
        check_syntax_error(self, 'lambda a, /, a: None', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'lambda a, /, *, a: None', "duplicate argument 'a' in function definition")
        check_syntax_error(self, 'lambda a, /, b, /: None')
        check_syntax_error(self, 'lambda a, /, b, /, c: None')
        check_syntax_error(self, 'lambda a, /, b, /, c, *, d: None')
        check_syntax_error(self, 'lambda a, *, b, /, c: None')

    def test_posonly_methods(self):
        if False:
            i = 10
            return i + 15

        class Example:

            def f(self, a, b, /):
                if False:
                    for i in range(10):
                        print('nop')
                return (a, b)
        self.assertEqual(Example().f(1, 2), (1, 2))
        self.assertEqual(Example.f(Example(), 1, 2), (1, 2))
        self.assertRaises(TypeError, Example.f, 1, 2)
        expected = "f\\(\\) got some positional-only arguments passed as keyword arguments: 'b'"
        with self.assertRaisesRegex(TypeError, expected):
            Example().f(1, b=2)

    def test_module_function(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 2 required positional arguments: 'a' and 'b'"):
            global_pos_only_f()

    def test_closures(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')

            def g(x2, /, y2):
                if False:
                    print('Hello World!')
                return x + y + x2 + y2
            return g
        self.assertEqual(f(1, 2)(3, 4), 10)
        with self.assertRaisesRegex(TypeError, "g\\(\\) missing 1 required positional argument: 'y2'"):
            f(1, 2)(3)
        with self.assertRaisesRegex(TypeError, 'g\\(\\) takes 2 positional arguments but 3 were given'):
            f(1, 2)(3, 4, 5)

        def f(x, /, y):
            if False:
                print('Hello World!')

            def g(x2, y2):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y + x2 + y2
            return g
        self.assertEqual(f(1, 2)(3, 4), 10)

        def f(x, /, y):
            if False:
                print('Hello World!')

            def g(x2, /, y2):
                if False:
                    return 10
                return x + y + x2 + y2
            return g
        self.assertEqual(f(1, 2)(3, 4), 10)
        with self.assertRaisesRegex(TypeError, "g\\(\\) missing 1 required positional argument: 'y2'"):
            f(1, 2)(3)
        with self.assertRaisesRegex(TypeError, 'g\\(\\) takes 2 positional arguments but 3 were given'):
            f(1, 2)(3, 4, 5)

    def test_annotations_in_closures(self):
        if False:
            return 10

        def inner_has_pos_only():
            if False:
                i = 10
                return i + 15

            def f(x: int, /):
                if False:
                    i = 10
                    return i + 15
                ...
            return f
        assert inner_has_pos_only().__annotations__ == {'x': int}

        class Something:

            def method(self):
                if False:
                    while True:
                        i = 10

                def f(x: int, /):
                    if False:
                        print('Hello World!')
                    ...
                return f
        assert Something().method().__annotations__ == {'x': int}

        def multiple_levels():
            if False:
                return 10

            def inner_has_pos_only():
                if False:
                    while True:
                        i = 10

                def f(x: int, /):
                    if False:
                        return 10
                    ...
                return f
            return inner_has_pos_only()
        assert multiple_levels().__annotations__ == {'x': int}

    def test_same_keyword_as_positional_with_kwargs(self):
        if False:
            return 10

        def f(something, /, **kwargs):
            if False:
                i = 10
                return i + 15
            return (something, kwargs)
        self.assertEqual(f(42, something=42), (42, {'something': 42}))
        with self.assertRaisesRegex(TypeError, "f\\(\\) missing 1 required positional argument: 'something'"):
            f(something=42)
        self.assertEqual(f(42), (42, {}))

    def test_mangling(self):
        if False:
            for i in range(10):
                print('nop')

        class X:

            def f(self, __a=42, /):
                if False:
                    for i in range(10):
                        print('nop')
                return __a

            def f2(self, __a=42, /, __b=43):
                if False:
                    while True:
                        i = 10
                return (__a, __b)

            def f3(self, __a=42, /, __b=43, *, __c=44):
                if False:
                    i = 10
                    return i + 15
                return (__a, __b, __c)
        self.assertEqual(X().f(), 42)
        self.assertEqual(X().f2(), (42, 43))
        self.assertEqual(X().f3(), (42, 43, 44))

    def test_too_many_arguments(self):
        if False:
            print('Hello World!')
        fundef = 'def f(%s, /):\n  pass\n' % ', '.join(('i%d' % i for i in range(300)))
        compile(fundef, '<test>', 'single')

    def test_serialization(self):
        if False:
            i = 10
            return i + 15
        pickled_posonly = pickle.dumps(global_pos_only_f)
        pickled_optional = pickle.dumps(global_pos_only_and_normal)
        pickled_defaults = pickle.dumps(global_pos_only_defaults)
        unpickled_posonly = pickle.loads(pickled_posonly)
        unpickled_optional = pickle.loads(pickled_optional)
        unpickled_defaults = pickle.loads(pickled_defaults)
        self.assertEqual(unpickled_posonly(1, 2), (1, 2))
        expected = "global_pos_only_f\\(\\) got some positional-only arguments passed as keyword arguments: 'a, b'"
        with self.assertRaisesRegex(TypeError, expected):
            unpickled_posonly(a=1, b=2)
        self.assertEqual(unpickled_optional(1, 2), (1, 2))
        expected = "global_pos_only_and_normal\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"
        with self.assertRaisesRegex(TypeError, expected):
            unpickled_optional(a=1, b=2)
        self.assertEqual(unpickled_defaults(), (1, 2))
        expected = "global_pos_only_defaults\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"
        with self.assertRaisesRegex(TypeError, expected):
            unpickled_defaults(a=1, b=2)

    def test_async(self):
        if False:
            i = 10
            return i + 15

        async def f(a=1, /, b=2):
            return (a, b)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"):
            f(a=1, b=2)

        def _check_call(*args, **kwargs):
            if False:
                return 10
            try:
                coro = f(*args, **kwargs)
                coro.send(None)
            except StopIteration as e:
                result = e.value
            self.assertEqual(result, (1, 2))
        _check_call(1, 2)
        _check_call(1, b=2)
        _check_call(1)
        _check_call()

    def test_generator(self):
        if False:
            print('Hello World!')

        def f(a=1, /, b=2):
            if False:
                for i in range(10):
                    print('nop')
            yield (a, b)
        with self.assertRaisesRegex(TypeError, "f\\(\\) got some positional-only arguments passed as keyword arguments: 'a'"):
            f(a=1, b=2)
        gen = f(1, 2)
        self.assertEqual(next(gen), (1, 2))
        gen = f(1, b=2)
        self.assertEqual(next(gen), (1, 2))
        gen = f(1)
        self.assertEqual(next(gen), (1, 2))
        gen = f()
        self.assertEqual(next(gen), (1, 2))

    def test_super(self):
        if False:
            i = 10
            return i + 15
        sentinel = object()

        class A:

            def method(self):
                if False:
                    while True:
                        i = 10
                return sentinel

        class C(A):

            def method(self, /):
                if False:
                    print('Hello World!')
                return super().method()
        self.assertEqual(C().method(), sentinel)

    def test_annotations_constant_fold(self):
        if False:
            return 10

        def g():
            if False:
                i = 10
                return i + 15

            def f(x: not int is int, /):
                if False:
                    return 10
                ...
        codes = [(i.opname, i.argval) for i in dis.get_instructions(g)]
        self.assertNotIn(('UNARY_NOT', None), codes)
        self.assertIn(('IS_OP', 1), codes)
if __name__ == '__main__':
    unittest.main()