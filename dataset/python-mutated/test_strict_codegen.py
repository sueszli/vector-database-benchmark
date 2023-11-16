from __future__ import annotations
import sys
import unittest
from typing import Any
from .common import StrictTestBase, StrictTestWithCheckerBase
try:
    import cinderjit
except ImportError:
    cinderjit = None

class StrictCompilationTests(StrictTestBase):

    def test_strictmod_freeze_type(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            x = 1\n        '
        code = self.compile(codestr)
        self.assertInBytecode(code, 'LOAD_GLOBAL', '<freeze-type>')
        self.assertInBytecode(code, 'STORE_GLOBAL', '<classes>')
        with self.with_freeze_type_setting(True), self.in_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            with self.assertRaises(TypeError):
                C.x = 2
            self.assertEqual(C.x, 1)

    def test_strictmod_freeze_set_false(self):
        if False:
            return 10
        codestr = '\n        class C:\n            x = 1\n        '
        code = self.compile(codestr)
        with self.with_freeze_type_setting(False), self.in_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            C.x = 2
            self.assertEqual(C.x, 2)

    def test_strictmod_class_in_function(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def f():\n            class C:\n                x = 1\n            return C\n        '
        with self.with_freeze_type_setting(True), self.in_module(codestr) as mod:
            f = mod.f
            C = f()
            self.assertEqual(C.x, 1)
            code = f.__code__
            self.assertInBytecode(code, 'SETUP_FINALLY')
            self.assertInBytecode(code, 'STORE_FAST', '<classes>')

    def test_strictmod_freeze_class_in_function(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def f():\n            class C:\n                x = 1\n            return C\n        '
        with self.with_freeze_type_setting(True), self.in_module(codestr) as mod:
            f = mod.f
            C = f()
            self.assertEqual(C.x, 1)
            with self.assertRaises(TypeError):
                C.x = 2
            self.assertEqual(C.x, 1)

    def test_strictmod_class_not_in_function(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            pass\n        def f():\n            return C\n        '
        code = self.compile(codestr)
        self.assertNotInBytecode(code, 'SETUP_FINALLY')
        self.assertInBytecode(code, 'STORE_GLOBAL', '<classes>')

    def test_strictmod_fixed_modules_typing(self):
        if False:
            while True:
                i = 10
        codestr = '\n        from typing import final\n\n        @final\n        class C:\n            x = 1\n        '
        code = self.compile(codestr)
        self.assertInBytecode(code, 'STORE_GLOBAL', 'final')
        with self.with_freeze_type_setting(True), self.in_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            with self.assertRaises(TypeError):
                C.x = 2
            self.assertEqual(C.x, 1)

class StrictBuiltinCompilationTests(StrictTestWithCheckerBase):

    def test_deps_run(self) -> None:
        if False:
            i = 10
            return i + 15
        'other things which interact with dependencies need to run'
        called = False

        def side_effect(x: List[object]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal called
            called = True
            self.assertEqual(x, [42])
            x.append(23)
        code = '\n            x = []\n            y = list(x)\n            x.append(42)\n            side_effect(x)\n        '
        mod = self.compile_and_run(code, builtins={'side_effect': side_effect})
        self.assertEqual(mod.y, [])
        self.assertTrue(called)

    def test_deps_run_2(self) -> None:
        if False:
            while True:
                i = 10
        'other things which interact with dependencies need to run'
        called = False

        def side_effect(x: List[object]) -> None:
            if False:
                i = 10
                return i + 15
            nonlocal called
            called = True
            self.assertEqual(x, [42])
            x.append(23)
        code = '\n            x = []\n            y = list(x)\n            x.append(42)\n            side_effect(x)\n            y = list(x)\n        '
        mod = self.compile_and_run(code, builtins={'side_effect': side_effect})
        self.assertEqual(mod.y, [42, 23])
        self.assertTrue(called)

    def test_deps_not_run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "independent pieces of code don't cause others to run"
        called = False

        def side_effect(x: object) -> None:
            if False:
                print('Hello World!')
            nonlocal called
            called = True
        code = '\n            x = []\n            y = 2\n            side_effect(x)\n        '
        mod = self.compile_and_run(code, builtins={'side_effect': side_effect})
        self.assertEqual(mod.y, 2)
        self.assertEqual(called, True)

    def test_builtins(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            x = 1\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min})
        self.assertEqual(mod.f(), 0)
        code = '\n            x = 1\n            min = 3\n            del min\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': max})
        self.assertEqual(mod.f(), 1)
        code = '\n            x = 1\n            def f():\n                return min(x, 0)\n\n            x = globals()\n        '
        mod = self.compile_and_run(code, builtins={'min': min, globals: None})
        self.assertNotIn('min', mod.x)

    def test_del_shadowed_builtin(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            min = None\n            x = 1\n            del min\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'NameError': NameError})
        self.assertEqual(mod.f(), 0)
        code = '\n            min = None\n            del min\n            x = 1\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': max})
        self.assertEqual(mod.f(), 1)

    def test_del_shadowed_and_call_globals(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            min = 2\n            del min\n            x = globals()\n        '
        mod = self.compile_and_run(code)
        self.assertNotIn('min', mod.x)
        self.assertNotIn('<assigned:min>', mod.x)

    def test_cant_assign(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            x = 1\n            def f():\n                return x\n        '
        mod = self.compile_and_run(code)
        with self.assertRaises(AttributeError):
            mod.x = 42

    def test_deleted(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            x = 1\n            del x\n        '
        mod = self.compile_and_run(code)
        with self.assertRaises(AttributeError):
            mod.x

    def test_deleted_mixed_global_non_name(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            x = 1\n            y = {2:3, 4:2}\n            del x, y[2]\n        '
        mod = self.compile_and_run(code)
        with self.assertRaises(AttributeError):
            mod.x
        self.assertEqual(mod.y, {4: 2})

    def test_deleted_mixed_global_non_global(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            x = 1\n            def f():\n                global x\n                y = 2\n                del x, y\n                return y\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.x, 1)
        with self.assertRaises(UnboundLocalError):
            mod.f()

    def test_deleted_non_global(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            y = {2:3, 4:2}\n            del y[2]\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.y, {4: 2})

    def test_deleted_accessed_on_call(self) -> None:
        if False:
            return 10
        code = '\n            x = 1\n            del x\n            def f ():\n                a = x\n        '
        with self.assertRaisesRegex(NameError, "name 'x' is not defined"):
            mod = self.compile_and_run(code)
            mod.f()

    def test_closure(self) -> None:
        if False:
            print('Hello World!')
        code = '\n            abc = 42\n            def x():\n                abc = 100\n                def inner():\n                    return abc\n                return inner\n\n            a = x()() # should be 100\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.a, 100)

    def test_nonlocal_alias(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            abc = 42\n            def x():\n                abc = 100\n                def inner():\n                    global abc\n                    return abc\n                return inner\n\n            a = x()() # should be 42\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.a, 42)

    def test_nonlocal_alias_called_from_mod(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            abc = 42\n            def x():\n                abc = 100\n                def inner():\n                    global abc\n                    del abc\n                return inner\n\n            x()()\n        '
        mod = self.compile_and_run(code)
        self.assertFalse(hasattr(mod, 'abc'))

    def test_nonlocal_alias_multi_func(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            def abc():\n                return 100\n\n            def x():\n                def abc():\n                    return 42\n                def inner():\n                    global abc\n                    return abc\n                return inner\n\n            a = x()()() # should be 100\n        '
        mod = self.compile_and_run(code)
        self.assertEqual('abc', mod.x()().__name__)
        self.assertEqual('abc', mod.x()().__qualname__)
        self.assertEqual(mod.a, 100)

    def test_nonlocal_alias_prop(self) -> None:
        if False:
            return 10
        code = '\n            from __strict__ import strict_slots\n            @strict_slots\n            class C:\n                x = 1\n\n            def x():\n                @strict_slots\n                class C:\n                    x = 2\n                def inner():\n                    global C\n                    return C\n                return inner\n\n            a = x()().x\n        '
        mod = self.compile_and_run(code)
        self.assertEqual('C', mod.x()().__name__)
        self.assertEqual('C', mod.x()().__qualname__)
        self.assertEqual(mod.a, 1)

    def test_global_assign(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            abc = 42\n            def modify(new_value):\n                global abc\n                abc = new_value\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.abc, 42)
        mod.modify(100)
        self.assertEqual(mod.abc, 100)

    def test_global_delete(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            abc = 42\n            def f():\n                global abc\n                del abc\n\n            f()\n        '
        mod = self.compile_and_run(code)
        self.assertFalse(hasattr(mod, 'abc'))

    def test_call_globals(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            abc = 42\n            x = globals()\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.x['abc'], 42)
        self.assertEqual(mod.x['__name__'], '<module>')

    def test_shadow_del_globals(self) -> None:
        if False:
            return 10
        're-assigning to a deleted globals should restore our globals helper'
        code = '\n            globals = 2\n            abc = 42\n            del globals\n            x = globals()\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.x['abc'], 42)
        self.assertEqual(mod.x['__name__'], '<module>')

    def test_vars(self) -> None:
        if False:
            print('Hello World!')
        code = '\n            abc = 42\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(vars(mod)['abc'], 42)

    def test_double_def(self) -> None:
        if False:
            return 10
        code = '\n            x = 1\n            def f():\n                return x\n\n            def f():\n                return 42\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.f(), 42)

    def test_exec(self) -> None:
        if False:
            return 10
        code = "\n            y = []\n            def f():\n                x = []\n                exec('x.append(42); y.append(100)')\n                return x, y\n        "
        mod = self.compile_and_run(code)
        self.assertEqual(mod.f(), ([42], [100]))
        code = "\n            y = []\n            def f():\n                x = []\n                exec('x.append(42); y.append(100)', {'x': [], 'y': []})\n                return x, y\n        "
        mod = self.compile_and_run(code)
        self.assertEqual(mod.f(), ([], []))
        code = "\n            x = 1\n            def f():\n                exec('global x; x = 2')\n            f()\n        "
        mod = self.compile_and_run(code)
        self.assertEqual(mod.x, 1)

    def test_eval(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = "\n            y = 42\n            def f():\n                x = 100\n                return eval('x, y')\n        "
        mod = self.compile_and_run(code)
        self.assertEqual(mod.f(), (100, 42))
        code = "\n            y = 42\n            def f():\n                x = 100\n                return eval('x, y', {'x':23, 'y':5})\n        "
        mod = self.compile_and_run(code)
        self.assertEqual(mod.f(), (23, 5))

    def test_define_dunder_globals(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            __globals__ = 42\n        '
        mod = self.compile_and_run(code)
        self.assertEqual(mod.__globals__, 42)

    def test_shadow_via_for(self) -> None:
        if False:
            print('Hello World!')
        code = '\n            for min in [1,2,3]:\n                pass\n            x = 1\n            del min\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'NameError': NameError})
        self.assertEqual(mod.f(), 0)

    def test_del_shadowed_via_tuple(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            (min, max) = None, None\n            x = 1\n            del min\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'max': max, 'min': min})
        self.assertEqual(mod.f(), 0)

    def test_del_shadowed_via_list(self) -> None:
        if False:
            print('Hello World!')
        code = '\n            (min, max) = None, None\n            x = 1\n            del min\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'max': max, 'min': min})
        self.assertEqual(mod.f(), 0)

    def test_list_comp_aliased_builtin(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            min = 1\n            del min\n            y = [min for x in [1,2,3]]\n            x = 1\n            def f():\n                return y[0](x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min})
        self.assertEqual(mod.f(), 0)

    def test_set_comp_aliased_builtin(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\n            min = 1\n            del min\n            y = {min for x in [1,2,3]}\n            x = 1\n            def f():\n                return next(iter(y))(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'iter': iter, 'next': next})
        self.assertEqual(mod.f(), 0)

    def test_gen_comp_aliased_builtin(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\n            min = 1\n            del min\n            y = (min for x in [1,2,3])\n            x = 1\n            def f():\n                return next(iter(y))(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'iter': iter, 'next': next})
        self.assertEqual(mod.f(), 0)

    def test_dict_comp_aliased_builtin(self) -> None:
        if False:
            return 10
        code = '\n            min = 1\n            del min\n            y = {min:x for x in [1,2,3]}\n            x = 1\n            def f():\n                return next(iter(y))(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'iter': iter, 'next': next})
        self.assertEqual(mod.f(), 0)

    def test_try_except_alias_builtin(self) -> None:
        if False:
            print('Hello World!')
        code = '\n            try:\n                raise Exception()\n            except Exception as min:\n                pass\n            x = 1\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'Exception': Exception})
        self.assertEqual(mod.f(), 0)

    def test_try_except_alias_builtin_2(self) -> None:
        if False:
            while True:
                i = 10
        code = '\n            try:\n                raise Exception()\n            except Exception as min:\n                pass\n            except TypeError as min:\n                pass\n            x = 1\n            def f():\n                return min(x, 0)\n        '
        mod = self.compile_and_run(code, builtins={'min': min, 'Exception': Exception, 'TypeError': TypeError})
        self.assertEqual(mod.f(), 0)

    def test_try_except_alias_builtin_check_exc(self) -> None:
        if False:
            i = 10
            return i + 15
        code = "\n            try:\n                raise Exception()\n            except Exception as min:\n                if type(min) is not Exception:\n                    raise Exception('wrong exception type!')\n            x = 1\n            def f():\n                return min(x, 0)\n        "
        mod = self.compile_and_run(code, builtins={'min': min, 'Exception': Exception, 'type': type})
        self.assertEqual(mod.f(), 0)

class StrictCheckedCompilationTests(StrictTestWithCheckerBase):

    def test_strictmod_freeze_type(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        import __strict__\n        class C:\n            x = 1\n        '
        with self.with_freeze_type_setting(True), self.in_checked_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            with self.assertRaises(TypeError):
                C.x = 2
            self.assertEqual(C.x, 1)

    def test_strictmod_mutable(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        import __strict__\n        from __strict__ import mutable\n\n        @mutable\n        class C:\n            x = 1\n        '
        code = self.check_and_compile(codestr)
        self.assertInBytecode(code, 'STORE_GLOBAL', 'mutable')
        with self.with_freeze_type_setting(True), self.in_checked_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            C.x = 2
            self.assertEqual(C.x, 2)

    def test_strictmod_mutable_noanalyze(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        import __strict__\n        from __strict__ import mutable, allow_side_effects\n\n        @mutable\n        class C:\n            x = 1\n        '
        with self.with_freeze_type_setting(True), self.in_module(codestr) as mod:
            C = mod.C
            self.assertEqual(C.x, 1)
            C.x = 2
            self.assertEqual(C.x, 2)

    def test_strictmod_cached_property(self):
        if False:
            return 10
        codestr = '\n        import __strict__\n        from __strict__ import strict_slots, _mark_cached_property, mutable\n        def dec(x):\n            _mark_cached_property(x, False, dec)\n            class C:\n                def __get__(self, inst, ctx):\n                    return x(inst)\n\n            return C()\n\n        @mutable\n        @strict_slots\n        class C:\n            @dec\n            def f(self):\n                return 1\n        '
        with self.with_freeze_type_setting(True), self.in_checked_module(codestr) as mod:
            C = mod.C
            c = C()
            self.assertEqual(c.f, 1)
            self.assertEqual(c.f, 1)
if __name__ == '__main__':
    unittest.main()