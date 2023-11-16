"""Tests for detecting free vars in tf.function."""
import functools
import unittest
from absl.testing import parameterized
import numpy as np
from tensorflow.core.function.capture import free_vars_detect
from tensorflow.python.util import tf_decorator

def get_var_name(d):
    if False:
        i = 10
        return i + 15
    return [var.name for var in d]

class GetSelfObjFromClosureTest(parameterized.TestCase):

    def test_single_enclosing_class(self):
        if False:
            return 10

        class Foo:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.val = 1

            def bar(self):
                if False:
                    while True:
                        i = 10
                x = 2

                def fn():
                    if False:
                        while True:
                            i = 10
                    return self.val + x
                return fn
        foo = Foo()
        fn = foo.bar()
        self_obj = free_vars_detect._get_self_obj_from_closure(fn)
        self.assertIs(self_obj, foo)

class FreeVarDetectionTest(parameterized.TestCase):

    def test_func_arg(self):
        if False:
            return 10
        x = 1

        def f(x):
            if False:
                print('Hello World!')
            return x + 1
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertEmpty(func_map)

    def test_func_local_var(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                return 10
            x = 1
            return x + 1
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertEmpty(func_map)

    def test_global_var_int(self):
        if False:
            for i in range(10):
                print('nop')
        x = 1

        def f():
            if False:
                i = 10
                return i + 15
            return x + 1
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_builtin_func(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                print('Hello World!')
            return len(x)
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertEmpty(func_map)

    def test_global_var_dict(self):
        if False:
            for i in range(10):
                print('nop')
        glob = {'a': 1}

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return glob['a'] + 1
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_global_var_dict_w_var_index(self):
        if False:
            i = 10
            return i + 15
        glob = {'a': 1}
        key = 'a'

        def f():
            if False:
                i = 10
                return i + 15
            return glob[key] + 1
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['glob', 'key'])

    def test_duplicate_global_var(self):
        if False:
            print('Hello World!')
        x = 1

        def f():
            if False:
                print('Hello World!')
            return x + x
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['x'])

    @parameterized.named_parameters(('lambda_1', lambda _x: 3), ('lambda_2', lambda _x: 3))
    def test_multiple_lambda_w_same_line_num_and_args(self, fn):
        if False:
            print('Hello World!')
        func_map = free_vars_detect._detect_function_free_vars(fn)
        self.assertEmpty(func_map)

    def test_lambda_wo_free_var(self):
        if False:
            i = 10
            return i + 15
        f = lambda x: x + x
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertEmpty(func_map)

    def test_lambda_w_free_var(self):
        if False:
            return 10
        glob = 1
        f = lambda x: x + glob
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_multi_lambda_w_free_var(self):
        if False:
            while True:
                i = 10
        glob = 1
        g = lambda x: x + glob
        h = lambda : glob + 1

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return g(x) + h()
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertLen(func_map, 3)
        self.assertIn('f', func_map.keys())
        self.assertIn('g', func_map.keys())
        self.assertIn('h', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['g', 'h'])
        free_vars = get_var_name(func_map['g'])
        self.assertSequenceEqual(free_vars, ['glob'])
        free_vars = get_var_name(func_map['h'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_lambda_inline(self):
        if False:
            while True:
                i = 10
        glob = 1

        def f(x):
            if False:
                print('Hello World!')
            return lambda : x + glob
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_glob_numpy_var(self):
        if False:
            i = 10
            return i + 15
        a = 0
        b = np.asarray(1)

        def f():
            if False:
                print('Hello World!')
            c = np.asarray(2)
            res = a + b + c
            return res
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['a', 'b'])

    def test_global_var_in_nested_func(self):
        if False:
            while True:
                i = 10
        x = 1

        def f():
            if False:
                return 10

            def g():
                if False:
                    print('Hello World!')
                return x + 1
            return g()
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        self.assertLen(func_map.keys(), 1)
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_global_var_from_outer_func(self):
        if False:
            return 10
        x = 1

        def g():
            if False:
                print('Hello World!')
            return x + 1

        def f():
            if False:
                while True:
                    i = 10
            return g()
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        self.assertIn('g', func_map.keys())
        self.assertLen(func_map.keys(), 2)
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['g'])
        free_vars = get_var_name(func_map['g'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_method(self):
        if False:
            i = 10
            return i + 15
        x = 1

        class Foo:

            def f(self):
                if False:
                    while True:
                        i = 10
                return x
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 1)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_method_w_method_call(self):
        if False:
            print('Hello World!')
        x = 0

        class Foo:

            def f(self):
                if False:
                    return 10
                return self.g

            def g(self):
                if False:
                    for i in range(10):
                        print('nop')
                return [x]
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['self.g'])
        self.assertIn('Foo.g', func_map.keys())
        free_vars = get_var_name(func_map['Foo.g'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_method_w_self_as_arg(self):
        if False:
            i = 10
            return i + 15
        x = 1

        class Foo:

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return self.g(self)

            def g(self, obj):
                if False:
                    while True:
                        i = 10
                if obj != self:
                    return x
                else:
                    return -x
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['self.g'])
        self.assertIn('Foo.g', func_map.keys())
        free_vars = get_var_name(func_map['Foo.g'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_self_inside_method(self):
        if False:
            i = 10
            return i + 15
        x = 1

        class Foo:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.val = 2

            def bar(self):
                if False:
                    print('Hello World!')

                def tf_func():
                    if False:
                        i = 10
                        return i + 15
                    return self.val + x
                return tf_func
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.bar())
        self.assertLen(func_map.keys(), 1)
        self.assertIn('tf_func', func_map.keys())
        free_vars = get_var_name(func_map['tf_func'])
        self.assertSequenceEqual(free_vars, ['self', 'self.val', 'x'])

    def test_self_inside_function_w_multiple_closures(self):
        if False:
            i = 10
            return i + 15

        class Foo:

            def method(self):
                if False:
                    print('Hello World!')

                class Baz:

                    def baz_str(self):
                        if False:
                            for i in range(10):
                                print('nop')
                        return 'Baz'
                baz = Baz()
                x = 'x'

                class Bar:

                    def bar_str(self):
                        if False:
                            i = 10
                            return i + 15
                        return x + 'Bar'

                    def method(self):
                        if False:
                            for i in range(10):
                                print('nop')

                        def fn():
                            if False:
                                i = 10
                                return i + 15
                            return self.bar_str() + baz.baz_str()
                        return fn
                bar = Bar()
                return bar.method()
        foo = Foo()
        fn = foo.method()
        self.assertLen(fn.__closure__, 2)
        func_map = free_vars_detect._detect_function_free_vars(fn)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('fn', func_map.keys())
        free_vars = get_var_name(func_map['fn'])
        self.assertSequenceEqual(free_vars, ['baz', 'self', 'self.bar_str'])
        self.assertIn('Bar.bar_str', func_map.keys())
        free_vars = get_var_name(func_map['Bar.bar_str'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_method_w_self_attribute(self):
        if False:
            return 10
        x = 0

        class Foo:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 1
                self.y = 2

            def f(self):
                if False:
                    print('Hello World!')
                return self.g + self.x + self.y

            def g(self):
                if False:
                    for i in range(10):
                        print('nop')
                return x
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['self.g', 'self.x', 'self.y'])
        self.assertIn('Foo.g', func_map.keys())
        free_vars = get_var_name(func_map['Foo.g'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_method_w_multiple_attributes(self):
        if False:
            i = 10
            return i + 15
        glob = 'dummy_value'

        class Foo:

            def f(self):
                if False:
                    return 10
                return self.g.h.x.y.z

            def g(self):
                if False:
                    i = 10
                    return i + 15
                return glob
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['self.g'])
        self.assertIn('Foo.g', func_map.keys())
        free_vars = get_var_name(func_map['Foo.g'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_classmethod_decorator(self):
        if False:
            print('Hello World!')
        glob = 1

        class Foo:

            @classmethod
            def f(cls):
                if False:
                    while True:
                        i = 10
                return glob
        func_map = free_vars_detect._detect_function_free_vars(Foo.f)
        self.assertLen(func_map.keys(), 1)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_method_call_classmethod(self):
        if False:
            return 10
        glob = 1

        class Foo:

            def f(self):
                if False:
                    return 10
                return self.g()

            @classmethod
            def g(cls):
                if False:
                    return 10
                return glob
        foo = Foo()
        func_map = free_vars_detect._detect_function_free_vars(foo.f)
        self.assertLen(func_map.keys(), 2)
        self.assertIn('Foo.f', func_map.keys())
        free_vars = get_var_name(func_map['Foo.f'])
        self.assertSequenceEqual(free_vars, ['self.g'])
        self.assertIn('Foo.g', func_map.keys())
        free_vars = get_var_name(func_map['Foo.g'])
        self.assertSequenceEqual(free_vars, ['glob'])

    def test_global_var_from_renamed_outer_func(self):
        if False:
            i = 10
            return i + 15
        x = 1

        def g():
            if False:
                i = 10
                return i + 15
            return x + 1

        def f():
            if False:
                while True:
                    i = 10
            h = g
            return h()
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        self.assertIn('g', func_map.keys())
        self.assertLen(func_map.keys(), 2)
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['g'])
        free_vars = get_var_name(func_map['g'])
        self.assertSequenceEqual(free_vars, ['x'])

    def test_decorated_method_w_self_no_exception(self):
        if False:
            while True:
                i = 10
        'Test this pattern does not raise any exceptions.'

        def dummy_tf_function(func):
            if False:
                i = 10
                return i + 15
            func_map = free_vars_detect._detect_function_free_vars(func)
            self.assertLen(func_map, 1)
            self.assertIn('foo', func_map.keys())
            free_vars = get_var_name(func_map['foo'])
            self.assertSequenceEqual(free_vars, ['dummy_tf_function'])

            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                return func(*args, **kwargs)
            return wrapper
        glob = 1

        class Foo:

            @dummy_tf_function
            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.bar()

            def bar(self):
                if False:
                    print('Hello World!')
                return glob
        _ = Foo()

    @parameterized.parameters((functools.update_wrapper, True), (tf_decorator.make_decorator, False))
    def test_func_w_decorator(self, make_decorator, wrapper_first):
        if False:
            for i in range(10):
                print('nop')
        x = 1

        def decorator_foo(func):
            if False:
                return 10

            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                return func(*args, **kwargs)
            if wrapper_first:
                return make_decorator(wrapper, func)
            else:
                return make_decorator(func, wrapper)

        @decorator_foo
        @decorator_foo
        def f():
            if False:
                return 10

            @decorator_foo
            @decorator_foo
            def g():
                if False:
                    return 10
                return x + 1
            return g()
        func_map = free_vars_detect._detect_function_free_vars(f)
        self.assertIn('f', func_map.keys())
        self.assertLen(func_map.keys(), 2)
        free_vars = get_var_name(func_map['f'])
        self.assertSequenceEqual(free_vars, ['decorator_foo', 'x'])

    @unittest.skip('Feature not implemented')
    def test_global_var_from_arg_func(self):
        if False:
            return 10
        x = 1

        def g():
            if False:
                return 10
            return x + 1

        def f(h):
            if False:
                for i in range(10):
                    print('nop')
            return h()
        _ = f(g)

class GenerateLoggingTest(parameterized.TestCase):

    def _remove_explanation(self, logging_txt):
        if False:
            for i in range(10):
                print('nop')
        free_vars = logging_txt.split('\n')
        self.assertGreater(len(free_vars), 2)
        return '\n'.join(free_vars[2:])

    def test_none_input(self):
        if False:
            i = 10
            return i + 15
        txt = free_vars_detect.generate_free_var_logging(None)
        self.assertIsNone(txt)

    def test_non_function_input(self):
        if False:
            return 10
        x = 1

        class Foo:

            def bar(self):
                if False:
                    return 10
                return x
        foo = Foo()
        txt = free_vars_detect.generate_free_var_logging(foo)
        self.assertIsNone(txt)

    def test_func_wo_source_code(self):
        if False:
            i = 10
            return i + 15
        code = 'def f_exec():\n  return 1'
        exec(code, globals())
        txt = free_vars_detect.generate_free_var_logging(f_exec)
        self.assertIsNone(txt)

    def test_no_free_var(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                while True:
                    i = 10
            return x + 1
        txt = free_vars_detect.generate_free_var_logging(f)
        self.assertIsNone(txt)

    def test_single_func(self):
        if False:
            i = 10
            return i + 15
        x = 1
        y = 2

        def f(a):
            if False:
                print('Hello World!')
            return a + x + y
        txt = free_vars_detect.generate_free_var_logging(f)
        txt = self._remove_explanation(txt)
        self.assertEqual(txt, 'Inside function f(): x, y')

    def test_nested_func(self):
        if False:
            for i in range(10):
                print('nop')
        x = 1
        y = 2

        def g():
            if False:
                while True:
                    i = 10
            return y

        def f():
            if False:
                while True:
                    i = 10
            return g() + x
        txt = free_vars_detect.generate_free_var_logging(f)
        txt = self._remove_explanation(txt)
        lines = txt.split('\n')
        self.assertLen(lines, 2)
        self.assertEqual(lines[0], 'Inside function f(): g, x')
        self.assertEqual(lines[1], 'Inside function g(): y')

    def test_method_w_method_call(self):
        if False:
            i = 10
            return i + 15
        x = 0

        class Foo:

            def f(self):
                if False:
                    print('Hello World!')
                return self.g

            def g(self):
                if False:
                    for i in range(10):
                        print('nop')
                return [x]
        foo = Foo()
        txt = free_vars_detect.generate_free_var_logging(foo.f)
        txt = self._remove_explanation(txt)
        lines = txt.split('\n')
        self.assertLen(lines, 2)
        self.assertEqual(lines[0], 'Inside function Foo.f(): self.g')
        self.assertEqual(lines[1], 'Inside function Foo.g(): x')

    def test_partial_func(self):
        if False:
            print('Hello World!')
        x = 1
        y = 2

        def f(a):
            if False:
                print('Hello World!')
            return a + x + y
        partial_f = functools.partial(f, a=0)
        txt = free_vars_detect.generate_free_var_logging(partial_f)
        txt = self._remove_explanation(txt)
        self.assertEqual(txt, 'Inside function f(): x, y')

    def test_partial_method(self):
        if False:
            return 10
        x = 0

        class Foo:

            def f(self):
                if False:
                    while True:
                        i = 10
                return self.g

            def g(self):
                if False:
                    i = 10
                    return i + 15
                return [x]
            partial_f = functools.partialmethod(f)
        foo = Foo()
        txt = free_vars_detect.generate_free_var_logging(foo.partial_f)
        txt = self._remove_explanation(txt)
        lines = txt.split('\n')
        self.assertLen(lines, 2)
        self.assertEqual(lines[0], 'Inside function Foo.f(): self.g')
        self.assertEqual(lines[1], 'Inside function Foo.g(): x')

    def test_partial_wrapped_partial_func(self):
        if False:
            for i in range(10):
                print('nop')

        def decorator_foo(func):
            if False:
                print('Hello World!')

            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                return func(*args, **kwargs)
            return functools.update_wrapper(wrapper, func)
        x = 1
        y = 2

        def f(a, b):
            if False:
                while True:
                    i = 10
            return a + b + x + y
        f = functools.partial(f, a=0)
        f = decorator_foo(f)
        f = functools.partial(f, b=0)
        txt = free_vars_detect.generate_free_var_logging(f)
        txt = self._remove_explanation(txt)
        self.assertEqual(txt, 'Inside function f(): x, y')

    def test_freevar_threshold(self):
        if False:
            i = 10
            return i + 15
        a = b = c = d = e = 1

        def f():
            if False:
                return 10
            return a + b + c + d + e
        txt = free_vars_detect.generate_free_var_logging(f, var_threshold=3)
        txt = self._remove_explanation(txt)
        self.assertEqual(txt, 'Inside function f(): a, b, c...')

    def test_func_threshold(self):
        if False:
            while True:
                i = 10
        x = 1

        def g():
            if False:
                for i in range(10):
                    print('nop')
            return x

        def h():
            if False:
                for i in range(10):
                    print('nop')
            return x

        def f():
            if False:
                i = 10
                return i + 15
            return g() + h()
        txt = free_vars_detect.generate_free_var_logging(f, fn_threshold=2)
        txt = self._remove_explanation(txt)
        lines = txt.split('\n')
        self.assertLen(lines, 3)
        self.assertEqual(lines[0], 'Inside function f(): g, h')
        self.assertEqual(lines[1], 'Inside function g(): x')
        self.assertEqual(lines[2], '...')

    def test_func_second_call_return_none(self):
        if False:
            i = 10
            return i + 15
        x = 1

        def f():
            if False:
                while True:
                    i = 10
            return x
        logging_txt = free_vars_detect.generate_free_var_logging(f)
        self.assertIsNotNone(logging_txt)
        logging_txt = free_vars_detect.generate_free_var_logging(f)
        self.assertIsNone(logging_txt)
if __name__ == '__main__':
    unittest.main()