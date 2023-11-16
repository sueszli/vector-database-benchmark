from compiler.pycodegen import CinderCodeGenerator
from .common import StaticTestBase

class SuperTests(StaticTestBase):

    def test_dynamic_base_class(self):
        if False:
            return 10
        nonstatic = '\n            class A:\n                x = 1\n        '
        with self.in_module(nonstatic, code_gen=CinderCodeGenerator) as nonstatic_mod:
            codestr = f'\n                from {nonstatic_mod.__name__} import A\n\n                class B(A):\n                    x = 2\n\n                    def foo(self):\n                        return super().x\n            '
            with self.in_strict_module(codestr) as mod:
                self.assertInBytecode(mod.B.foo, 'LOAD_ATTR_SUPER')
                self.assertEqual(mod.B().foo(), 1)

    def test_method_in_parent_class(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class A:\n            def f(self):\n                return 4\n\n        class B(A):\n            def g(self):\n                return super().f()\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertInBytecode(mod.B.g, 'INVOKE_FUNCTION', ((mod.__name__, 'A', 'f'), 1))
            self.assertEqual(mod.foo(), 4)

    def test_method_in_parents_parent_class(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class AA:\n            def f(self):\n                return 4\n\n        class A(AA):\n            def g(self):\n                return 8\n\n        class B(A):\n            def g(self):\n                return super().f()\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertInBytecode(mod.B.g, 'INVOKE_FUNCTION', ((mod.__name__, 'AA', 'f'), 1))
            self.assertEqual(mod.foo(), 4)

    def test_super_call_with_parameters(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class A:\n            def f(self):\n                return 4\n\n        class B(A):\n            def f(self):\n                return 5\n\n        class C(B):\n            def g(self):\n                return super(B, self).f()\n\n        def foo():\n            return C().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.C.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_property_in_parent_class(self):
        if False:
            print('Hello World!')
        codestr = '\n        class A:\n            @property\n            def f(self):\n                return 4\n\n        class B(A):\n            def g(self):\n                return super().f\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_property_in_parents_parent_class(self):
        if False:
            print('Hello World!')
        codestr = '\n        class AA:\n            @property\n            def f(self):\n                return 4\n\n        class A(AA):\n            pass\n\n        class B(A):\n            def g(self):\n                return super().f\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_attr_in_parent_class(self):
        if False:
            return 10
        codestr = '\n        class A:\n            f = 4\n\n        class B(A):\n            def g(self):\n                return super().f\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_attr_in_parent_class(self):
        if False:
            print('Hello World!')
        codestr = '\n        class A:\n            f = 4\n\n        class B(A):\n            def g(self):\n                return super().f\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_attr_in_parents_parent_class(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class AA:\n            f = 4\n\n        class A(AA):\n            pass\n\n        class B(A):\n            def g(self):\n                return super().f\n\n        def foo():\n            return B().g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')
            self.assertEqual(mod.foo(), 4)

    def test_unsupported_class_nested_in_function(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def foo():\n\n            class B:\n                def g(self):\n                    return super().f()\n\n            return B\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.foo().g, 'INVOKE_FUNCTION')

    def test_unsupported_class_nested_in_class(self):
        if False:
            while True:
                i = 10
        codestr = '\n        class A:\n            class B:\n                def g(self):\n                    return super().f()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.A.B.g, 'INVOKE_FUNCTION')

    def test_unsupported_class_nested_in_funcdef(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class A:\n            def g(self):\n                def f():\n                    return super().bar()\n                return f()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.A.g, 'INVOKE_FUNCTION')

    def test_unsupported_case_falls_back_to_dynamic(self):
        if False:
            return 10
        codestr = '\n        class A(Exception):\n            pass\n\n        class B(A):\n            def g(self):\n                return super().__init__()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.B.g, 'INVOKE_FUNCTION')