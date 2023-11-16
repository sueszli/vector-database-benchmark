import unittest
from compiler.consts import CO_STATICALLY_COMPILED
from compiler.pycodegen import CinderCodeGenerator
from unittest import skip
from .common import StaticTestBase

class NonStaticInheritanceTests(StaticTestBase):

    def test_static_return_is_resolved_with_multiple_levels_of_inheritance(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                def foobar(self, x: int) -> int:\n                    return x\n                def f(self) -> int:\n                    return self.foobar(1)\n        '
        with self.in_strict_module(codestr, name='mymod', enable_patching=True) as mod:
            C = mod.C

            class D(C):

                def foobar(self, x: int) -> int:
                    if False:
                        while True:
                            i = 10
                    return x + 1

            class E(D):

                def foobar(self, x: int) -> int:
                    if False:
                        i = 10
                        return i + 15
                    return x + 2
            self.assertEqual(D().f(), 2)
            self.assertEqual(E().f(), 3)

    def test_multiple_inheritance_initialization(self):
        if False:
            i = 10
            return i + 15
        "Primarily testing that when we have multiple inheritance that\n        we safely initialize all of our v-tables.  Previously we could\n        init B2 while initializing the bases for DM, and then we wouldn't\n        initialize the classes derived from it."
        codestr = '\n            class C:\n                def foobar(self, x: int) -> int:\n                    return x\n                def f(self) -> int:\n                    return self.foobar(1)\n                def g(self): pass\n\n            def f(x: C):\n                return x.f()\n        '
        with self.in_strict_module(codestr, name='mymod', enable_patching=True, freeze=False) as mod:
            C = mod.C
            f = mod.f

            class B1(C):

                def f(self):
                    if False:
                        return 10
                    return 10

            class B2(C):

                def f(self):
                    if False:
                        return 10
                    return 20

            class D(B2):

                def f(self):
                    if False:
                        i = 10
                        return i + 15
                    return 30

            class DM(B2, B1):
                pass
            C.g = 42
            self.assertEqual(f(B1()), 10)
            self.assertEqual(f(B2()), 20)
            self.assertEqual(f(D()), 30)
            self.assertEqual(f(DM()), 20)

    def test_multiple_inheritance_initialization_invoke_only(self):
        if False:
            return 10
        "Primarily testing that when we have multiple inheritance that\n        we safely initialize all of our v-tables.  Previously we could\n        init B2 while initializing the bases for DM, and then we wouldn't\n        initialize the classes derived from it."
        codestr = '\n            class C:\n                def foobar(self, x: int) -> int:\n                    return x\n                def f(self) -> int:\n                    return self.foobar(1)\n                def g(self): pass\n\n            def f(x: C):\n                return x.f()\n        '
        with self.in_strict_module(codestr, name='mymod', enable_patching=True) as mod:
            C = mod.C
            f = mod.f

            class B1(C):

                def f(self):
                    if False:
                        while True:
                            i = 10
                    return 10

            class B2(C):

                def f(self):
                    if False:
                        i = 10
                        return i + 15
                    return 20

            class D(B2):

                def f(self):
                    if False:
                        return 10
                    return 30

            class DM(B2, B1):
                pass
            self.assertEqual(f(C()), 1)
            self.assertEqual(f(B1()), 10)
            self.assertEqual(f(B2()), 20)
            self.assertEqual(f(D()), 30)
            self.assertEqual(f(DM()), 20)

    def test_inherit_abc(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from abc import ABC\n\n            class C(ABC):\n                @property\n                def f(self) -> int:\n                    return 42\n\n                def g(self) -> int:\n                    return self.f\n        '
        with self.in_module(codestr) as mod:
            C = mod.C
            a = C()
            self.assertEqual(a.g(), 42)

    def test_static_decorator_non_static_class(self):
        if False:
            return 10
        codestr = '\n            def mydec(f):\n                def wrapper(*args, **kwargs):\n                    return f(*args, **kwargs)\n                return wrapper\n\n            class B:\n                def g(self): pass\n\n            def f(x: B):\n                return x.g()\n        '
        with self.in_module(codestr) as mod:
            mydec = mod.mydec
            B = mod.B
            f = mod.f
            f(B())

            class D(B):

                @mydec
                def f(self):
                    if False:
                        i = 10
                        return i + 15
                    pass
            self.assertEqual(D().f(), None)
            D.f = lambda self: 42
            self.assertEqual(f(B()), None)
            self.assertEqual(f(D()), None)
            self.assertEqual(D().f(), 42)

    def test_nonstatic_multiple_inheritance_invoke(self):
        if False:
            print('Hello World!')
        'multiple inheritance from non-static classes should\n        result in only static classes in the v-table'
        codestr = "\n        def f(x: str):\n            return x.encode('utf8')\n        "

        class C:
            pass

        class D(C, str):
            pass
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(D('abc')), b'abc')

    def test_nonstatic_multiple_inheritance_invoke_static_base(self):
        if False:
            while True:
                i = 10
        codestr = '\n        class B:\n            def f(self):\n                return 42\n\n        def f(x: B):\n            return x.f()\n        '

        class C:

            def f(self):
                if False:
                    return 10
                return 'abc'
        with self.in_module(codestr) as mod:

            class D(C, mod.B):
                pass
            self.assertEqual(mod.f(D()), 'abc')

    def test_nonstatic_multiple_inheritance_invoke_static_base_2(self):
        if False:
            return 10
        codestr = '\n        class B:\n            def f(self):\n                return 42\n\n        def f(x: B):\n            return x.f()\n        '

        class C:

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 'abc'
        with self.in_module(codestr) as mod:

            class D(C, mod.B):

                def f(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    return 'foo'
            self.assertEqual(mod.f(D()), 'foo')

    def test_no_inherit_multiple_static_bases(self):
        if False:
            print('Hello World!')
        codestr = '\n            class A:\n                pass\n\n            class B:\n                pass\n        '
        with self.in_module(codestr) as mod:
            with self.assertRaisesRegex(TypeError, 'multiple bases have instance lay-out conflict'):

                class C(mod.A, mod.B):
                    pass

    def test_no_inherit_multiple_static_bases_indirect(self):
        if False:
            print('Hello World!')
        codestr = '\n            class A:\n                pass\n\n            class B:\n                pass\n        '
        with self.in_module(codestr) as mod:

            class C(mod.B):
                pass
            with self.assertRaisesRegex(TypeError, 'multiple bases have instance lay-out conflict'):

                class D(C, mod.A):
                    pass

    def test_no_inherit_static_and_builtin(self):
        if False:
            return 10
        codestr = '\n            class A:\n                pass\n        '
        with self.in_module(codestr) as mod:
            with self.assertRaisesRegex(TypeError, 'multiple bases have instance lay-out conflict'):

                class C(mod.A, str):
                    pass

    def test_mutate_sub_sub_class(self):
        if False:
            for i in range(10):
                print('nop')
        "patching non-static class through multiple levels\n        of inheritance shouldn't crash"
        codestr = '\n        class B:\n            def __init__(self): pass\n            def f(self):\n                return 42\n\n        def f(b: B):\n            return b.f()\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(mod.B()), 42)

            class D1(mod.B):

                def __init__(self):
                    if False:
                        print('Hello World!')
                    pass

            class D2(D1):

                def __init__(self):
                    if False:
                        return 10
                    pass
            D1.__init__ = lambda self: None
            D2.__init__ = lambda self: None
            self.assertEqual(mod.f(D1()), 42)
            self.assertEqual(mod.f(D2()), 42)

    def test_invoke_class_method_dynamic_base(self):
        if False:
            for i in range(10):
                print('nop')
        bases = '\n        class B1: pass\n        '
        codestr = '\n        from bases import B1\n        class D(B1):\n            @classmethod\n            def f(cls):\n                return cls.g()\n\n            @classmethod\n            def g(cls):\n                return 42\n\n        def f():\n            return D.f()\n        '
        with self.in_module(bases, name='bases', code_gen=CinderCodeGenerator), self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), 42)

    def test_no_inherit_static_through_nonstatic(self):
        if False:
            return 10
        base = '\n            class A:\n                pass\n        '
        nonstatic = '\n            from base import A\n\n            class B(A):\n                pass\n        '
        static = '\n            from nonstatic import B\n\n            class C(B):\n                pass\n        '
        with self.in_module(base, name='base'), self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator):
            with self.assertRaisesRegex(TypeError, "Static compiler cannot verify that static type 'C' is a valid override of static base 'A' because intervening base 'B' is non-static"):
                self.run_code(static)

    def test_nonstatic_derived_method_in_static_class(self):
        if False:
            print('Hello World!')
        nonstatic = '\n            def decorate(f):\n                def foo(*args, **kwargs):\n                    return f(*args, **kwargs)\n                return foo\n        '
        static = '\n            from nonstatic import decorate\n            class C:\n               def f(self):\n                   return 1\n\n            class D(C):\n               @decorate\n               def f(self):\n                   return 2\n\n            def invoke_f(c: C):\n                return c.f()\n\n            def invoke_d_f():\n                d = D()\n                return d.f()\n        '
        with self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator), self.in_module(static) as mod:
            self.assertEqual(mod.D.f.__code__.co_flags & CO_STATICALLY_COMPILED, 0)
            self.assertEqual(mod.invoke_f(mod.C()), 1)
            self.assertEqual(mod.invoke_f(mod.D()), 2)
            self.assertEqual(mod.invoke_d_f(), 1)
            self.assertInBytecode(mod.invoke_d_f, 'INVOKE_FUNCTION', ((mod.__name__, 'C', 'f'), 1))

    def test_nonstatic_override_init_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        nonstatic = "\n            from static import B\n\n            class B2(B):\n                def __init_subclass__(self):\n                    # don't call super\n                    pass\n\n            class D(B2):\n                x = 100\n                def __init__(self):\n                    pass\n\n        "
        static = '\n\n            class B:\n                x: int = 42\n                def get_x(self):\n                    return self.x\n                def set_x(self, value):\n                    self.x = value\n\n        '
        with self.in_module(static, name='static') as mod, self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator) as nonstatic_mod:
            self.assertInBytecode(mod.B.get_x, 'INVOKE_METHOD')
            self.assertInBytecode(mod.B.set_x, 'INVOKE_METHOD')
            d = nonstatic_mod.D()
            self.assertRaises(TypeError, d.set_x, 100)
            self.assertEqual(d.get_x(), 100)

    def test_nonstatic_override_init_subclass_inst(self):
        if False:
            print('Hello World!')
        nonstatic = "\n            from static import B\n\n            class B2(B):\n                def __init_subclass__(self):\n                    # don't call super\n                    pass\n\n            class D(B2):\n                def __init__(self):\n                    self.x = 100\n\n        "
        static = '\n            class B:\n                x: int = 42\n                def get_x(self):\n                    return self.x\n                def set_x(self, value):\n                    self.x = value\n\n        '
        with self.in_module(static, name='static') as mod, self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator) as nonstatic_mod:
            self.assertInBytecode(mod.B.get_x, 'INVOKE_METHOD')
            self.assertInBytecode(mod.B.set_x, 'INVOKE_METHOD')
            d = nonstatic_mod.D()
            d.set_x(200)
            self.assertEqual(d.get_x(), 200)
            self.assertEqual(d.__dict__, {})
            self.assertEqual(mod.B.x, 42)

    def test_nonstatic_call_base_init(self):
        if False:
            i = 10
            return i + 15
        nonstatic = '\n            class B:\n                def __init_subclass__(cls):\n                    cls.foo = 42\n\n        '
        static = '\n            from nonstatic import B\n            class D(B):\n                pass\n\n        '
        with self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator) as nonstatic_mod, self.in_module(static) as mod:
            self.assertEqual(mod.D.foo, 42)

    def test_nonstatic_call_base_init_other_super(self):
        if False:
            return 10
        nonstatic = '\n            class B:\n                def __init_subclass__(cls):\n                    cls.foo = 42\n\n        '
        static = '\n            from nonstatic import B\n            class D(B):\n                def __init__(self):\n                    return super().__init__()\n\n\n        '
        with self.in_module(nonstatic, name='nonstatic', code_gen=CinderCodeGenerator) as nonstatic_mod, self.in_module(static) as mod:
            self.assertEqual(mod.D.foo, 42)
if __name__ == '__main__':
    unittest.main()