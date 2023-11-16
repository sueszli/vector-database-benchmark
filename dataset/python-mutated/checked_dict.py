from __static__ import chkdict
from compiler.static.types import FAST_LEN_DICT, TypedSyntaxError
from unittest import skip, skipIf
from .common import StaticTestBase, type_mismatch
try:
    import cinderjit
except ImportError:
    cinderjit = None

class CheckedDictTests(StaticTestBase):

    def test_invoke_chkdict_method(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        from __static__ import CheckedDict\n        def dict_maker() -> CheckedDict[int, int]:\n            return CheckedDict[int, int]({2:2})\n        def func():\n            a = dict_maker()\n            return a.keys()\n\n        '
        with self.in_module(codestr) as mod:
            f = mod.func
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'int'), ('builtins', 'int')), '!', 'keys'), 1))
            self.assertEqual(list(f()), [2])
            self.assert_jitted(f)

    def test_generic_method_ret_type(self):
        if False:
            i = 10
            return i + 15
        codestr = "\n            from __static__ import CheckedDict\n\n            from typing import Optional\n            MAP: CheckedDict[str, Optional[str]] = CheckedDict[str, Optional[str]]({'abc': 'foo', 'bar': None})\n            def f(x: str) -> Optional[str]:\n                return MAP.get(x)\n        "
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'str'), ('builtins', 'str', '?')), '!', 'get'), 3))
            self.assertEqual(f('abc'), 'foo')
            self.assertEqual(f('bar'), None)

    def test_compile_nested_dict(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = CheckedDict[B, int]({B():42, D():42})\n                y = CheckedDict[int, CheckedDict[B, int]]({42: x})\n                return y\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            self.assertEqual(type(test()), chkdict[int, chkdict[B, int]])

    def test_compile_dict_setdefault(self):
        if False:
            print('Hello World!')
        codestr = "\n            from __static__ import CheckedDict\n            def testfunc():\n                x = CheckedDict[int, str]({42: 'abc', })\n                x.setdefault(100, 43)\n        "
        with self.assertRaisesRegex(TypedSyntaxError, 'Literal\\[43\\] received for positional arg 2, expected Optional\\[str\\]'):
            self.compile(codestr, modname='foo')

    def test_compile_dict_get(self):
        if False:
            return 10
        codestr = "\n            from __static__ import CheckedDict\n            def testfunc():\n                x = CheckedDict[int, str]({42: 'abc', })\n                x.get(42, 42)\n        "
        with self.assertRaisesRegex(TypedSyntaxError, 'Literal\\[42\\] received for positional arg 2, expected Optional\\[str\\]'):
            self.compile(codestr, modname='foo')
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = CheckedDict[B, int]({B():42, D():42})\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            self.assertEqual(type(test()), chkdict[B, int])

    def test_chkdict_literal(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n            def testfunc():\n                x: CheckedDict[int,str]  = {}\n                return x\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(type(f()), chkdict[int, str])

    def test_compile_dict_get_typed(self):
        if False:
            while True:
                i = 10
        codestr = "\n            from __static__ import CheckedDict\n            def testfunc():\n                x = CheckedDict[int, str]({42: 'abc', })\n                y: str | None = x.get(42)\n        "
        self.compile(codestr)

    def test_compile_dict_setdefault_typed(self):
        if False:
            return 10
        codestr = "\n            from __static__ import CheckedDict\n            def testfunc():\n                x = CheckedDict[int, str]({42: 'abc', })\n                y: str | None = x.setdefault(100, 'foo')\n        "
        self.compile(codestr)

    def test_compile_dict_setitem(self):
        if False:
            i = 10
            return i + 15
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[int, str]({1:'abc'})\n                x.__setitem__(2, 'def')\n                return x\n        "
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            x = test()
            self.assertInBytecode(test, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'int'), ('builtins', 'str')), '!', '__setitem__'), 3))
            self.assertEqual(x, {1: 'abc', 2: 'def'})

    def test_compile_dict_setitem_subscr(self):
        if False:
            print('Hello World!')
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[int, str]({1:'abc'})\n                x[2] = 'def'\n                return x\n        "
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            x = test()
            self.assertInBytecode(test, 'INVOKE_FUNCTION', (('__static__', 'chkdict', (('builtins', 'int'), ('builtins', 'str')), '!', '__setitem__'), 3))
            self.assertEqual(x, {1: 'abc', 2: 'def'})

    def test_compile_generic_dict_getitem_bad_type(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[str, int]({"abc": 42})\n                return x[42]\n        '
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('Literal[42]', 'str')):
            self.compile(codestr, modname='foo')

    def test_compile_generic_dict_setitem_bad_type(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[str, int]({"abc": 42})\n                x[42] = 42\n        '
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('Literal[42]', 'str')):
            self.compile(codestr, modname='foo')

    def test_compile_generic_dict_setitem_bad_type_2(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[str, int]({"abc": 42})\n                x["foo"] = "abc"\n        '
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('str', 'int')):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_shadowcode(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = CheckedDict[B, int]({B():42, D():42})\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            for i in range(200):
                self.assertEqual(type(test()), chkdict[B, int])

    def test_compile_checked_dict_optional(self):
        if False:
            i = 10
            return i + 15
        codestr = "\n            from __static__ import CheckedDict\n            from typing import Optional\n\n            def testfunc():\n                x = CheckedDict[str, str | None]({\n                    'x': None,\n                    'y': 'z'\n                })\n                return x\n        "
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            x = f()
            x['z'] = None
            self.assertEqual(type(x), chkdict[str, str | None])

    def test_compile_checked_dict_bad_annotation(self):
        if False:
            print('Hello World!')
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x: 42 = CheckedDict[str, str]({'abc':'abc'})\n                return x\n        "
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), chkdict[str, str])

    def test_compile_checked_dict_ann_differs(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x: CheckedDict[int, int] = CheckedDict[str, str]({'abc':'abc'})\n                return x\n        "
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('chkdict[str, str]', 'chkdict[int, int]')):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_ann_differs_2(self):
        if False:
            print('Hello World!')
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x: int = CheckedDict[str, str]({'abc':'abc'})\n                return x\n        "
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('chkdict[str, str]', 'int')):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_opt_out_by_default(self):
        if False:
            print('Hello World!')
        codestr = '\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = {B():42, D():42}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), dict)

    def test_compile_checked_dict_opt_in(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__.compiler_flags import checked_dicts\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = {B():42, D():42}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            self.assertEqual(type(test()), chkdict[B, int])

    def test_compile_checked_dict_explicit_dict(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import pydict\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x: pydict = {B():42, D():42}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), dict)

    def test_compile_checked_dict_reversed(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x = CheckedDict[B, int]({D():42, B():42})\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            self.assertEqual(type(test()), chkdict[B, int])

    def test_compile_checked_dict_type_specified(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x: CheckedDict[B, int] = CheckedDict[B, int]({D():42})\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            self.assertEqual(type(test()), chkdict[B, int])

    def test_compile_checked_dict_with_annotation_comprehension(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x: CheckedDict[int, object] = {int(i): object() for i in range(1, 5)}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), chkdict[int, object])

    def test_compile_checked_dict_with_annotation(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n\n            def testfunc():\n                x: CheckedDict[B, int] = {B():42}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            B = mod.B
            test()
            self.assertEqual(type(test()), chkdict[B, int])

    def test_compile_checked_dict_with_annotation_wrong_value_type(self):
        if False:
            i = 10
            return i + 15
        codestr = "\n            from __static__ import CheckedDict\n\n            class B: pass\n\n            def testfunc():\n                x: CheckedDict[B, int] = {B():'hi'}\n                return x\n        "
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('chkdict[foo.B, str]', 'chkdict[foo.B, int]')):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_with_annotation_wrong_key_type(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedDict\n\n            class B: pass\n\n            def testfunc():\n                x: CheckedDict[B, int] = {object():42}\n                return x\n        '
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('chkdict[object, Literal[42]]', 'chkdict[foo.B, int]')):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_wrong_unknown_type(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            def f(x: int):\n                return x\n\n            def testfunc(iter):\n                return f({x:42 for x in iter})\n\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "dict received for positional arg 'x', expected int"):
            self.compile(codestr, modname='foo')

    def test_compile_checked_dict_explicit_dict_as_dict(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import pydict as dict\n            class B: pass\n            class D(B): pass\n\n            def testfunc():\n                x: dict = {B():42, D():42}\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), dict)

    def test_compile_checked_dict_from_dict_call(self):
        if False:
            return 10
        codestr = '\n            from __static__.compiler_flags import checked_dicts\n\n            def testfunc():\n                x = dict(x=42)\n                return x\n        '
        with self.assertRaisesRegex(TypeError, "cannot create '__static__.chkdict\\[K, V\\]' instances"):
            with self.in_module(codestr) as mod:
                test = mod.testfunc
                test()

    def test_compile_checked_dict_from_dict_call_2(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__.compiler_flags import checked_dicts\n\n            def testfunc():\n                x = dict[str, int](x=42)\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), chkdict[str, int])

    def test_compile_checked_dict_from_dict_call_3(self):
        if False:
            return 10
        codestr = '\n            from __future__ import annotations\n            from __static__.compiler_flags import checked_dicts\n\n            def testfunc():\n                x = dict[str, int](x=42)\n                return x\n        '
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertEqual(type(test()), chkdict[str, int])

    def test_compile_checked_dict_len(self):
        if False:
            print('Hello World!')
        codestr = "\n            from __static__ import CheckedDict\n\n            def testfunc():\n                x = CheckedDict[int, str]({1:'abc'})\n                return len(x)\n        "
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertInBytecode(test, 'FAST_LEN', FAST_LEN_DICT)
            if cinderjit is not None:
                cinderjit.get_and_clear_runtime_stats()
            self.assertEqual(test(), 1)
            if cinderjit is not None:
                stats = cinderjit.get_and_clear_runtime_stats().get('deopt')
                self.assertFalse(stats)

    def test_compile_checked_dict_clen(self):
        if False:
            return 10
        codestr = "\n            from __static__ import CheckedDict, clen, int64\n\n            def testfunc() -> int64:\n                x = CheckedDict[int, str]({1:'abc'})\n                return clen(x)\n        "
        with self.in_module(codestr) as mod:
            test = mod.testfunc
            self.assertInBytecode(test, 'FAST_LEN', FAST_LEN_DICT)
            if cinderjit is not None:
                cinderjit.get_and_clear_runtime_stats()
            self.assertEqual(test(), 1)
            if cinderjit is not None:
                stats = cinderjit.get_and_clear_runtime_stats().get('deopt')
                self.assertFalse(stats)

    def test_compile_checked_dict_create_with_dictcomp(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict, clen, int64\n\n            def testfunc() -> None:\n                x = CheckedDict[int, str]({int(i): int(i) for i in\n                               range(1, 5)})\n        '
        with self.assertRaisesRegex(TypedSyntaxError, type_mismatch('chkdict[int, int]', 'chkdict[int, str]')):
            self.compile(codestr)

    def test_chkdict_float_is_dynamic(self):
        if False:
            print('Hello World!')
        codestr = '\n        from __static__ import CheckedDict\n\n        def main():\n            d = CheckedDict[float, str]({2.0: "hello", 2.3: "foobar"})\n            reveal_type(d)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "reveal_type\\(d\\): 'Exact\\[chkdict\\[dynamic, str\\]\\]'"):
            self.compile(codestr)

    def test_build_checked_dict_cached(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        from __static__ import CheckedDict\n\n        def f() -> str:\n            d: CheckedDict[float, str] = {2.0: "hello", 2.3: "foobar"}\n            return d[2.0]\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.f, 'BUILD_CHECKED_MAP')
            for i in range(50):
                self.assertEqual(mod.f(), 'hello')
            self.assertEqual(mod.f(), 'hello')