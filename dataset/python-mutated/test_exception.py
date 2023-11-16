from torch.testing._internal.common_utils import TestCase
import torch
from torch import nn
'\nTest TorchScript exception handling.\n'

class TestException(TestCase):

    def test_pyop_exception_message(self):
        if False:
            return 10

        class Foo(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv(x)
        foo = Foo()
        with self.assertRaisesRegex(RuntimeError, 'Expected 3D \\(unbatched\\) or 4D \\(batched\\) input to conv2d'):
            foo(torch.ones([123]))

    def test_builtin_error_messsage(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def close_match(x):
                if False:
                    print('Hello World!')
                return x.masked_fill(True)
        with self.assertRaisesRegex(RuntimeError, 'This op may not exist or may not be currently supported in TorchScript'):

            @torch.jit.script
            def unknown_op(x):
                if False:
                    for i in range(10):
                        print('nop')
                torch.set_anomaly_enabled(True)
                return x

    def test_exceptions(self):
        if False:
            return 10
        cu = torch.jit.CompilationUnit('\n            def foo(cond):\n                if bool(cond):\n                    raise ValueError(3)\n                return 1\n        ')
        cu.foo(torch.tensor(0))
        with self.assertRaisesRegex(torch.jit.Error, '3'):
            cu.foo(torch.tensor(1))

        def foo(cond):
            if False:
                while True:
                    i = 10
            a = 3
            if bool(cond):
                raise ArbitraryError(a, 'hi')
                if 1 == 2:
                    raise ArbitraryError
            return a
        with self.assertRaisesRegex(RuntimeError, 'undefined value ArbitraryError'):
            torch.jit.script(foo)

        def exception_as_value():
            if False:
                i = 10
                return i + 15
            a = Exception()
            print(a)
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a value'):
            torch.jit.script(exception_as_value)

        @torch.jit.script
        def foo_no_decl_always_throws():
            if False:
                i = 10
                return i + 15
            raise RuntimeError('Hi')
        output_type = next(foo_no_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == 'NoneType')

        @torch.jit.script
        def foo_decl_always_throws():
            if False:
                return 10
            raise Exception('Hi')
        output_type = next(foo_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == 'Tensor')

        def foo():
            if False:
                return 10
            raise 3 + 4
        with self.assertRaisesRegex(RuntimeError, 'must derive from BaseException'):
            torch.jit.script(foo)

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            if 1 == 1:
                a = 1
            elif 1 == 1:
                raise Exception('Hi')
            else:
                raise Exception('Hi')
            return a
        self.assertEqual(foo(), 1)

        @torch.jit.script
        def tuple_fn():
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('hello', 'goodbye')
        with self.assertRaisesRegex(torch.jit.Error, 'hello, goodbye'):
            tuple_fn()

        @torch.jit.script
        def no_message():
            if False:
                while True:
                    i = 10
            raise RuntimeError
        with self.assertRaisesRegex(torch.jit.Error, 'RuntimeError'):
            no_message()

    def test_assertions(self):
        if False:
            return 10
        cu = torch.jit.CompilationUnit('\n            def foo(cond):\n                assert bool(cond), "hi"\n                return 0\n        ')
        cu.foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, 'AssertionError: hi'):
            cu.foo(torch.tensor(0))

        @torch.jit.script
        def foo(cond):
            if False:
                print('Hello World!')
            assert bool(cond), 'hi'
        foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, 'AssertionError: hi'):
            foo(torch.tensor(0))

    def test_python_op_exception(self):
        if False:
            return 10

        @torch.jit.ignore
        def python_op(x):
            if False:
                while True:
                    i = 10
            raise Exception('bad!')

        @torch.jit.script
        def fn(x):
            if False:
                return 10
            return python_op(x)
        with self.assertRaisesRegex(RuntimeError, 'operation failed in the TorchScript interpreter'):
            fn(torch.tensor(4))

    def test_dict_expansion_raises_error(self):
        if False:
            print('Hello World!')

        def fn(self):
            if False:
                i = 10
                return i + 15
            d = {'foo': 1, 'bar': 2, 'baz': 3}
            return {**d}
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'Dict expansion '):
            torch.jit.script(fn)

    def test_custom_python_exception(self):
        if False:
            i = 10
            return i + 15

        class MyValueError(ValueError):
            pass

        @torch.jit.script
        def fn():
            if False:
                while True:
                    i = 10
            raise MyValueError('test custom exception')
        with self.assertRaisesRegex(torch.jit.Error, 'jit.test_exception.MyValueError: test custom exception'):
            fn()

    def test_custom_python_exception_defined_elsewhere(self):
        if False:
            return 10
        from jit.myexception import MyKeyError

        @torch.jit.script
        def fn():
            if False:
                i = 10
                return i + 15
            raise MyKeyError('This is a user defined key error')
        with self.assertRaisesRegex(torch.jit.Error, 'jit.myexception.MyKeyError: This is a user defined key error'):
            fn()