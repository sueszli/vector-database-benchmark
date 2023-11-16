import unittest
from .common import StaticTestBase
try:
    import cinderjit
except ImportError:
    cinderjit = None

class ElideTypeChecksTests(StaticTestBase):

    @unittest.skipIf(cinderjit is None, 'JIT disabled')
    def test_invoke_function_skips_arg_type_checks(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            from xxclassloader import unsafe_change_type\n\n            class A:\n                pass\n\n            class B:\n                pass\n\n            def g(a: A) -> str:\n                return a.__class__.__name__\n\n            def f() -> str:\n                a = A()\n                # compiler is unaware that this changes the type of `a`,\n                # so it unsafely allows the following call g(a)\n                unsafe_change_type(a, B)\n                return g(a)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertInBytecode(mod.f, 'INVOKE_FUNCTION', ((mod.__name__, 'g'), 1))
            with self.assertRaisesRegex(TypeError, "g expected 'A' for argument a, got 'B'"):
                mod.g(mod.B())
            cinderjit.force_compile(mod.f)
            self.assertEqual(mod.f(), 'B')

    @unittest.skipIf(cinderjit is None, 'JIT disabled')
    def test_invoke_method_skips_arg_type_checks(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            from xxclassloader import unsafe_change_type\n\n            class A:\n                pass\n\n            class B:\n                pass\n\n            class C:\n                def g(self, a: A) -> str:\n                    return a.__class__.__name__\n\n            def f(c: C) -> str:\n                a = A()\n                # compiler is unaware that this changes the type of `a`,\n                # so it unsafely allows the following call C.g(a)\n                unsafe_change_type(a, B)\n                return c.g(a)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertInBytecode(mod.f, 'INVOKE_METHOD', ((mod.__name__, 'C', 'g'), 1))
            with self.assertRaisesRegex(TypeError, "g expected 'A' for argument a, got 'B'"):
                mod.C().g(mod.B())
            try:
                mod.f(mod.C())
            except TypeError:
                pass
            self.assertEqual(mod.f(mod.C()), 'B')

    def test_elide_check_with_one_optional(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            from typing import Optional\n            def foo() -> int:\n                def bar(g: Optional[str] = None) -> int:\n                    return int(g or "42")\n                return bar()\n        '
        with self.in_module(codestr) as mod:
            f = mod.foo
            self.assertEqual(f(), 42)

    def test_type_error_raised_when_eliding_defaults(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            from typing import Optional\n            def foo(f: int, g: Optional[str] = None) -> int:\n                return int(g or "42")\n        '
        with self.in_module(codestr) as mod:
            f = mod.foo
            with self.assertRaises(TypeError):
                f('1')