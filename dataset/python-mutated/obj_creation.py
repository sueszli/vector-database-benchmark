from __static__ import chkdict, chklist, int64
import inspect
import unittest
from cinder import freeze_type
from compiler.errors import TypedSyntaxError
from inspect import CO_SUPPRESS_JIT
from re import escape
from unittest import skip
from .common import StaticTestBase

class StaticObjCreationTests(StaticTestBase):

    def test_new_and_init(self):
        if False:
            return 10
        codestr = '\n            class C:\n                def __new__(cls, a):\n                    return object.__new__(cls)\n                def __init__(self, a):\n                    self.a = a\n\n            X = 0\n            def g() -> int:\n                global X\n                X += 1\n                return 1\n\n            def f() -> C:\n                return C(g())\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            f()
            self.assertEqual(mod.X, 1)

    def test_object_init_and_new(self):
        if False:
            return 10
        codestr = '\n            class C:\n                pass\n\n            def f(x: int) -> C:\n                return C(x)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, escape('<module>.C() takes no arguments')):
            self.compile(codestr)

    def test_init(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n\n                def __init__(self, a: int) -> None:\n                    self.value = a\n\n            def f(x: int) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42).value, 42)

    def test_init_primitive(self):
        if False:
            return 10
        codestr = '\n            from __static__ import int64\n            class C:\n\n                def __init__(self, a: int64) -> None:\n                    self.value: int64 = a\n\n            def f(x: int64) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            init = mod.C.__init__
            self.assertInBytecode(init, 'LOAD_LOCAL')
            self.assertInBytecode(init, 'STORE_FIELD')
            self.assertEqual(f(42).value, 42)

    def test_new_primitive(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import int64\n            class C:\n                value: int64\n                def __new__(cls, a: int64) -> "C":\n                    res: C = object.__new__(cls)\n                    res.value = a\n                    return res\n\n            def f(x: int64) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            init = mod.C.__new__
            self.assertInBytecode(init, 'LOAD_LOCAL')
            self.assertInBytecode(init, 'STORE_FIELD')
            self.assertEqual(f(42).value, 42)

    def test_init_frozen_type(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n\n                def __init__(self, a: int) -> None:\n                    self.value = a\n\n            def f(x: int) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            C = mod.C
            freeze_type(C)
            f = mod.f
            self.assertEqual(f(42).value, 42)

    def test_init_unknown_base(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from re import Scanner\n            class C(Scanner):\n                pass\n\n            def f(x: int) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'CALL_FUNCTION')

    def test_init_wrong_type(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n\n                def __init__(self, a: int) -> None:\n                    self.value = a\n\n            def f(x: str) -> C:\n                return C(x)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "type mismatch: str received for positional arg 'a', expected int"):
            self.compile(codestr)

    def test_init_extra_arg(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n\n                def __init__(self, a: int) -> None:\n                    self.value = a\n\n            def f(x: int) -> C:\n                return C(x, 42)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, escape('Mismatched number of args for function <module>.C.__init__. Expected 2, got 3')):
            self.compile(codestr)

    def test_new(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> "C":\n                    res = object.__new__(cls)\n                    res.value = a\n                    return res\n\n            def f(x: int) -> C:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42).value, 42)

    def test_new_wrong_type(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> "C":\n                    res = object.__new__(cls)\n                    res.value = a\n                    return res\n\n            def f(x: str) -> C:\n                return C(x)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "type mismatch: str received for positional arg 'a', expected int"):
            self.compile(codestr)

    def test_new_object(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> object:\n                    res = object.__new__(cls)\n                    res.value = a\n                    return res\n                def __init__(self, a: int):\n                    self.value = 100\n\n            def f(x: int) -> object:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42).value, 100)

    def test_new_dynamic(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int):\n                    res = object.__new__(cls)\n                    res.value = a\n                    return res\n                def __init__(self, a: int):\n                    self.value = 100\n\n            def f(x: int) -> object:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42).value, 100)

    def test_new_odd_ret_type(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> int:\n                    return 42\n\n            def f(x: int) -> int:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42), 42)

    def test_new_odd_ret_type_no_init(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> int:\n                    return 42\n                def __init__(self, *args) -> None:\n                    raise Exception("no way")\n\n            def f(x: int) -> int:\n                return C(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(42), 42)

    def test_new_odd_ret_type_error(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                value: int\n                def __new__(cls, a: int) -> int:\n                    return 42\n\n            def f(x: int) -> str:\n                return C(x)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'return type must be str, not int'):
            self.compile(codestr)

    def test_class_init_kw(self):
        if False:
            return 10
        codestr = "\n            class C:\n                def __init__(self, x: str):\n                    self.x: str = x\n\n            def f():\n                x = C(x='abc')\n                return x\n        "
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertNotInBytecode(f, 'CALL_FUNCTION_KW', 1)
            self.assertInBytecode(f, 'TP_ALLOC')
            self.assertInBytecode(f, 'INVOKE_FUNCTION')
            c = f()
            self.assertEqual(c.x, 'abc')

    def test_type_subclass(self):
        if False:
            while True:
                i = 10
        codestr = "\n            class C(type):\n                pass\n\n            def f() -> C:\n                return C('foo', (), {})\n        "
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C
            self.assertEqual(type(f()), C)

    def test_object_new(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C(object):\n                pass\n\n            def f() -> C:\n                return object.__new__(C)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C
            self.assertEqual(type(f()), C)

    def test_object_new_wrong_type(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C(object):\n                pass\n\n            def f() -> C:\n                return object.__new__(object)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'return type must be <module>.C, not object'):
            self.compile(codestr)

    def test_bool_call(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            def f(x) -> bool:\n                return bool(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('builtins', 'bool', '!', '__new__'), 2))
            self.assertEqual(f(42), True)
            self.assertEqual(f(0), False)

    def test_bool_accepts_union_types(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from typing import Optional\n\n            def f(x: Optional[int]) -> bool:\n                return bool(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertFalse(f(None))
            self.assertTrue(f(12))

    def test_list_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C(list):\n                pass\n\n            def f() -> C:\n                return C()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), [])
            self.assertInBytecode(f, 'TP_ALLOC')

    def test_list_subclass_iterable(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = "\n            class C(list):\n                pass\n\n            def f() -> C:\n                return C('abc')\n        "
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), ['a', 'b', 'c'])
            self.assertInBytecode(f, 'TP_ALLOC')

    def test_checkeddict_new(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedDict\n\n            def f() -> CheckedDict[str, int]:\n                return CheckedDict[str, int]()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), {})
            self.assertInBytecode(f, 'TP_ALLOC', ('__static__', 'chkdict', (('builtins', 'str'), ('builtins', 'int')), '!'))
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'str'), ('builtins', 'int')), '!', '__init__'), 2))

    def test_checkeddict_new_2(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedDict\n\n            def f() -> CheckedDict[str, int]:\n                return CheckedDict[str, int]({})\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), {})
            self.assertInBytecode(f, 'TP_ALLOC', ('__static__', 'chkdict', (('builtins', 'str'), ('builtins', 'int')), '!'))
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'str'), ('builtins', 'int')), '!', '__init__'), 2))

    def test_super_init_no_obj_invoke(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self):\n                    super().__init__()\n        '
        with self.in_module(codestr) as mod:
            f = mod.C.__init__
            self.assertNotInBytecode(f, 'INVOKE_METHOD')

    def test_super_init_no_load_attr_super(self):
        if False:
            print('Hello World!')
        codestr = '\n            x = super\n\n            class B:\n                def __init__(self, a):\n                    pass\n\n\n            class D(B):\n                def __init__(self):\n                    # force a non-optimizable super\n                    try:\n                        super(1, 2, 3).__init__(a=2)\n                    except:\n                        pass\n                    # and then use the aliased super, we still\n                    # have __class__ available\n                    x().__init__(a=2)\n\n            def f():\n                return D()\n        '
        code = self.compile(codestr)
        with self.in_module(codestr) as mod:
            f = mod.f
            D = mod.D
            self.assertTrue(D.__init__.__code__.co_flags & CO_SUPPRESS_JIT)
            self.assertTrue(isinstance(f(), D))

    def test_invoke_with_freevars(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                def __init__(self) -> None:\n                    super().__init__()\n\n\n            def f() -> C:\n                return C()\n        '
        code = self.compile(codestr)
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C
            freeze_type(C)
            self.assertInBytecode(f, 'INVOKE_FUNCTION')
            self.assertTrue(isinstance(f(), C))

    def test_super_redefined_uses_opt(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            super = super\n\n            class C:\n                def __init__(self):\n                    super().__init__()\n        '
        with self.in_module(codestr) as mod:
            init = mod.C.__init__
            self.assertInBytecode(init, 'LOAD_METHOD_SUPER')

    def test_generic_unknown_type_dict(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedDict\n            def make_C():\n                class C: pass\n                return C\n            C = make_C()\n            d = CheckedDict[str, C]({})\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(type(mod.d), chkdict[str, object])

    def test_generic_unknown_type_list(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedList\n            def make_C():\n                class C: pass\n                return C\n            C = make_C()\n            l = CheckedList[C]([])\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(type(mod.l), chklist[object])

    def test_class_method_call(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n            class B:\n                def __init__(self, a):\n                    self.a = a\n\n                @classmethod\n                def f(cls, *args):\n                    return cls(42)\n\n            class D:\n                def __init__(self, a, b):\n                    self.a = a\n                    self.b = b\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.B.f, 'CALL_FUNCTION', 1)
            self.assertNotInBytecode(mod.B.f, 'TP_ALLOC')