import builtins
from cinder import cached_property
from compiler.pycodegen import PythonCodeGenerator
from unittest import skip
from unittest.mock import Mock, patch
from .common import StaticTestBase

class SlotsWithDefaultTests(StaticTestBase):

    def test_access_from_instance_and_class(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 42\n\n        def f():\n            c = C()\n            return (C.x, c.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertNotInBytecode(mod.f, 'LOAD_FIELD')
            self.assertEqual(mod.f(), (42, 42))

    def test_nonstatic_access_from_instance_and_class(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 42\n        '
        with self.in_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 42)
            self.assertEqual(C().x, 42)

    def test_write_from_instance(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            x: int = 42\n\n        def f():\n            c = C()\n            c.x = 21\n            return (C.x, c.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertNotInBytecode(mod.f, 'LOAD_FIELD')
            self.assertEqual(mod.f(), (42, 21))

    def test_nonstatic_write_from_instance(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 42\n        '
        with self.in_module(codestr) as mod:
            C = mod.C
            c = C()
            c.x = 21
            self.assertEqual(C.x, 42)
            self.assertEqual(c.x, 21)

    def test_write_from_class(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 42\n\n        def f():\n            c = C()\n            C.x = 21\n            return (C.x, c.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertNotInBytecode(mod.f, 'LOAD_FIELD')
            self.assertEqual(mod.f(), (21, 21))

    def test_nonstatic_write_from_class(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            x: int = 42\n        '
        with self.in_module(codestr) as mod:
            C = mod.C
            c = C()
            C.x = 21
            self.assertEqual(C.x, 21)
            self.assertEqual(c.x, 21)

    def test_write_to_class_after_instance(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 42\n\n        def f():\n            c = C()\n            c.x = 36 # This write will get clobbered when the class gets patched below.\n            C.x = 21\n            return (C.x, c.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertNotInBytecode(mod.f, 'LOAD_FIELD')
            self.assertEqual(mod.f(), (21, 21))

    def test_inheritance(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 42\n\n        class D(C):\n            pass\n\n        def f():\n            d = D()\n            return (D.x, d.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(), (42, 42))

    def test_inheritance_with_override(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            x: int = 1\n\n        class D(C):\n            x: int = 3\n\n        def f():\n            c = C()\n            c.x = 2\n            d = D()\n            d.x = 4\n            return (C.x, c.x, D.x, d.x)\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(), (1, 2, 3, 4))

    def test_custom_descriptor_override_preserved(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 42\n\n        def f(c: C):\n            return c.x\n        '
        with self.in_module(codestr) as mod:

            class descr:

                def __get__(self, inst, ctx) -> int:
                    if False:
                        return 10
                    return 21

            class D(mod.C):
                x: int = descr()
            self.assertEqual(mod.f(D()), 21)

    def test_call(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 1\n\n        class D(C):\n            pass\n\n        def f(c: C):\n            return c.x\n        '
        with self.in_module(codestr) as mod:
            d = mod.D()
            self.assertEqual(mod.f(d), 1)
            d.x = 2
            self.assertEqual(mod.f(d), 2)

    def test_typed_descriptor_default_value_type_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            x: int = 1\n        '
        with self.in_module(codestr) as mod:
            c = mod.C()
            with self.assertRaisesRegex(TypeError, "expected 'int', got 'str' for attribute 'x'"):
                c.x = 'A'

    def test_typed_descriptor_default_value_patching_type_error(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            x: int = 1\n        '
        with self.in_module(codestr) as mod:
            with self.assertRaisesRegex(TypeError, 'Cannot assign a str, because C.x is expected to be a int'):
                mod.C.x = 'A'

    def test_nonstatic_inheritance_reads_allowed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n           return (type(c).x, c.x)\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                x: int = 2
            self.assertEqual(mod.f(D()), (2, 2))

    def test_nonstatic_inheritance_writes_allowed(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            initial_x = c.x\n            c.x = 2\n            return (initial_x, c.x, c.__class__.x)\n\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                x: int = 3
            self.assertEqual(mod.f(D()), (3, 2, 3))

    def test_nonstatic_inheritance_writes_allowed_init_subclass_override(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 1\n            def __init_subclass__(cls):\n                cls.foo = 42\n\n        def f(c: C):\n            initial_x = c.x\n            c.x = 2\n            return (initial_x, c.x, c.__class__.x)\n\n        '
        m = self.compile(codestr)
        with self.in_module(codestr) as mod:

            class D(mod.C):
                x: int = 3
            self.assertEqual(mod.f(D()), (3, 2, 3))
            self.assertEqual(D.foo, 42)

    def test_static_property_override(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 1\n            def get_x(self):\n                return self.x\n\n        class D(C):\n            @property\n            def x(self) -> int:\n                return 2\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.C().get_x(), 1)
            self.assertEqual(mod.D().get_x(), 2)

    def test_static_property_override_bad_type(self) -> None:
        if False:
            return 10
        codestr = "\n        class C:\n            x: int = 1\n            def get_x(self):\n                return self.x.val\n\n        class D(C):\n            @property\n            def x(self) -> str:\n                return 'abc'\n        "
        self.type_error(codestr, 'Cannot change type of inherited attribute')

    def test_static_property_override_no_type(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = "\n        class X:\n            def __init__(self, val: int):\n                self.val = val\n\n            def f(self):\n                pass\n\n        class C:\n            x: X = X(1)\n            def get_x(self):\n                return self.x.val\n\n        class D(C):\n            @property\n            def x(self):\n                return 'abc'\n        "
        self.type_error(codestr, 'Cannot change type of inherited attribute')

    def test_override_property_with_slot(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x\n\n        class D(C):\n            x: int = 1\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.C().get_x(), 2)
            self.assertEqual(mod.D().get_x(), 1)

    def test_override_property_with_slot_non_static(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                x: int = 1
            self.assertEqual(mod.C().get_x(), 2)
            self.assertEqual(D().get_x(), 1)

    def test_override_property_with_slot_no_value(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x\n\n        class D(C):\n            x: int\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.C().get_x(), 2)
            with self.assertRaises(AttributeError):
                mod.D().get_x()

    def test_override_property_with_slot_no_value_non_static(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x\n\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                x: int
            self.assertEqual(mod.C().get_x(), 2)
            self.assertEqual(D().get_x(), 2)

    def test_override_property_with_slot_non_static_slots(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x\n\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                __slots__ = 'x'
            self.assertEqual(mod.C().get_x(), 2)
            with self.assertRaises(AttributeError):
                D().get_x()

    def test_override_property_with_slot_bad_type(self) -> None:
        if False:
            return 10
        codestr = "\n        class C:\n            @property\n            def x(self) -> int:\n                return 2\n            def get_x(self):\n                return self.x.val\n\n        class D(C):\n            x: str = 'abc'\n        "
        self.type_error(codestr, 'Cannot change type of inherited attribute')

    def test_nonstatic_property_override(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            return c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                @property
                def x(self) -> int:
                    if False:
                        i = 10
                        return i + 15
                    return 2
            self.assertEqual(mod.f(mod.C()), (1, 1))
            self.assertEqual(mod.f(D()), (2, D.x))

    def test_nonstatic_property_override_setter(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            c.x = 123\n            return c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                @property
                def x(self) -> int:
                    if False:
                        i = 10
                        return i + 15
                    return 2
            self.assertEqual(mod.f(mod.C()), (123, 1))
            with self.assertRaisesRegex(AttributeError, "can't set attribute"):
                mod.f(D())

    def test_nonstatic_cached_property_override(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            return c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    self.hit_count = 0

                @cached_property
                def x(self) -> int:
                    if False:
                        for i in range(10):
                            print('nop')
                    self.hit_count += 1
                    return 2
            self.assertEqual(mod.f(mod.C()), (1, 1))
            d = D()
            self.assertEqual(d.hit_count, 0)
            self.assertEqual(mod.f(d), (2, D.x))
            self.assertEqual(d.hit_count, 1)
            self.assertEqual(mod.f(d), (2, D.x))
            self.assertEqual(d.hit_count, 1)

    def test_nonstatic_cached_property_override_type_error(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            return c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                @cached_property
                def x(self) -> str:
                    if False:
                        return 10
                    return 'A'
            self.assertEqual(mod.f(mod.C()), (1, 1))
            d = D()
            with self.assertRaisesRegex(TypeError, 'unexpected return type from D.x, expected int, got str'):
                mod.f(d)

    def test_nonstatic_cached_property_override_setter(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            c.x = 123\n            return c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                @cached_property
                def x(self) -> int:
                    if False:
                        i = 10
                        return i + 15
                    return 2
            self.assertEqual(mod.f(mod.C()), (123, 1))
            with self.assertRaisesRegex(TypeError, "'cached_property' doesn't support __set__"):
                self.assertEqual(mod.f(D()), (2, D.x))

    def test_override_with_slot_without_default(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 1\n\n        class D(C):\n            def __init__(self):\n                self.x = 3\n\n        def f(c: C):\n            r1 = c.x\n            r2 = c.__class__.x\n            c.x = 42\n            return r1, r2, c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(mod.C()), (1, 1, 42, 1))
            self.assertEqual(mod.f(mod.D()), (3, 1, 42, 1))

    def test_override_with_nonstatic_slot(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C):\n            r1 = c.x\n            r2 = c.__class__.x\n            c.x = 42\n            return r1, r2, c.x, c.__class__.x\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):

                def __init__(self):
                    if False:
                        print('Hello World!')
                    self.x = 3
            self.assertEqual(mod.f(mod.C()), (1, 1, 42, 1))
            self.assertEqual(mod.f(D()), (3, 1, 42, 1))

    def test_class_patching_allowed(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C) -> int:\n            return c.x\n        '
        with self.in_module(codestr, enable_patching=True) as mod:
            mod.C.x = 42
            c = mod.C()
            self.assertEqual(mod.f(c), 42)

    def test_class_patching_wrong_type(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C) -> int:\n            return c.x\n        '
        with self.in_module(codestr, enable_patching=True) as mod:
            with self.assertRaisesRegex(TypeError, 'Cannot assign a MagicMock, because C.x is expected to be a int'), patch(f'{mod.__name__}.C.x', return_value=1) as mock:
                c = mod.C()

    def test_instance_patching_allowed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C) -> int:\n            return c.x\n\n        def g(c: C):\n            c.x = 3\n        '
        with self.in_module(codestr, enable_patching=True) as mod:
            c = mod.C()
            with patch.object(c, 'x', 2):
                self.assertEqual(c.x, 2)
                self.assertEqual(mod.C.x, 1)
                self.assertEqual(mod.f(c), 2)
                mod.g(c)
                self.assertEqual(mod.f(c), 3)
                self.assertEqual(c.x, 3)

    def test_instance_patching_wrong_type(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            x: int = 1\n\n        def f(c: C) -> int:\n            return c.x\n\n        def g(c: C):\n            c.x = 3\n        '
        with self.in_module(codestr, enable_patching=True) as mod:
            c = mod.C()
            with self.assertRaisesRegex(TypeError, "expected 'int', got 'str'"):
                c.x = ''

    def test_type_descriptor_of_dynamic_type(self) -> None:
        if False:
            i = 10
            return i + 15
        non_static = '\n        class SomeType:\n            pass\n        '
        with self.in_module(non_static, code_gen=PythonCodeGenerator) as nonstatic_mod:
            static = f'\n                from dataclasses import dataclass\n                from {nonstatic_mod.__name__} import SomeType\n\n                class C:\n                    dynamic_field: SomeType = SomeType()\n            '
            with self.in_strict_module(static) as static_mod:
                c = static_mod.C()
                ST = nonstatic_mod.SomeType()
                self.assertNotEqual(c.dynamic_field, ST)
                c.dynamic_field = ST
                self.assertEqual(c.dynamic_field, ST)
                self.assertNotEqual(static_mod.C.dynamic_field, ST)

    def test_slot_assigned_conditionally(self):
        if False:
            while True:
                i = 10
        codestr = '\n        class Parent:\n            x: bool = False\n\n        class Child(Parent):\n\n            def __init__(self, flag: bool):\n                if flag:\n                    self.x = True\n        '
        with self.in_module(codestr) as mod:
            c1 = mod.Child(True)
            self.assertTrue(c1.x)
            c1 = mod.Child(False)
            self.assertFalse(c1.x)