from __future__ import annotations
from textwrap import dedent
from typing import final, List, Sequence, Tuple
from cinderx.strictmodule import StrictModuleLoader
from .common import StrictTestBase

@final
class OwnershipTests(StrictTestBase):

    def check_multiple_modules(self, modules: Sequence[Tuple[str, str]]) -> Sequence[List[Tuple[str, str, int, int]]]:
        if False:
            return 10
        checker = StrictModuleLoader([], '', [], [], True)
        checker.set_force_strict(True)
        errors = []
        for (code, name) in modules:
            m = checker.check_source(dedent(code), f'{name}.py', name, [])
            errors.append(list(m.errors))
        return errors

    def assertError(self, modules: Sequence[Tuple[str, str]], expected: str, **kwargs: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        errors = self.check_multiple_modules(modules)
        for mod_errors in errors:
            for error in mod_errors:
                if expected in error[0]:
                    return
        err_strings = [[e[0] for e in errs] for errs in errors]
        self.assertFalse(True, f'Expected: {expected}\nActual: {err_strings}')

    def test_list_modify(self) -> None:
        if False:
            while True:
                i = 10
        code1 = '\n            l1 = [1, 2, 3]\n        '
        code2 = '\n            from m1 import l1\n            l1[0] = 2\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], '[1,2,3] from module m1 is modified by m2')

    def test_list_append(self) -> None:
        if False:
            while True:
                i = 10
        code1 = '\n            l1 = [1, 2, 3]\n        '
        code2 = '\n            from m1 import l1\n            l1.append(4)\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], '[1,2,3] from module m1 is modified by m2')

    def test_dict_modify(self) -> None:
        if False:
            return 10
        code1 = '\n            d1 = {1: 2, 3: 4}\n        '
        code2 = '\n            from m1 import d1\n            d1[5] = 6\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], '{1: 2, 3: 4} from module m1 is modified by m2')

    def test_func_modify(self) -> None:
        if False:
            return 10
        code1 = '\n            d1 = {1: 2, 3: 4}\n        '
        code2 = '\n            def f(value):\n                value[5] = 1\n        '
        code3 = '\n            from m1 import d1\n            from m2 import f\n            f(d1)\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2'), (code3, 'm3')], '{1: 2, 3: 4} from module m1 is modified by m3')

    def test_decorator_modify(self) -> None:
        if False:
            return 10
        code1 = '\n            state = [0]\n            def dec(func):\n                state[0] = state[0] + 1\n                return func\n        '
        code2 = '\n            from m1 import dec\n            @dec\n            def g():\n                pass\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], '[0] from module m1 is modified by m2')

    def test_decorator_ok(self) -> None:
        if False:
            return 10
        code1 = '\n            def dec(cls):\n                cls.x = 1\n                return cls\n        '
        code2 = '\n            from m1 import dec\n            @dec\n            class C:\n                x: int = 0\n        '
        self.check_multiple_modules([(code1, 'm1'), (code2, 'm2')])

    def test_dict_ok(self) -> None:
        if False:
            i = 10
            return i + 15
        code1 = '\n            def f():\n                return {1: 2, 3: 4}\n        '
        code2 = '\n            from m1 import f\n            x = f()\n            x[5] = 6\n        '
        self.check_multiple_modules([(code1, 'm1'), (code2, 'm2')])

    def test_property_side_effect(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code1 = '\n            l = []\n            class C:\n                @property\n                def l(self):\n                    l.append(1)\n                    return l\n        '
        code2 = '\n            from m1 import C\n            c = C()\n            c.l\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], '[] from module m1 is modified by m2')

    def test_func_dunder_dict_modification(self) -> None:
        if False:
            return 10
        code1 = '\n            def f():\n                pass\n        '
        code2 = '\n            from m1 import f\n\n            f.__dict__["foo"] = 1\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], 'function.__dict__ from module m1 is modified by m2')

    def test_func_dunder_dict_keys(self) -> None:
        if False:
            return 10
        code1 = '\n            def f():\n                pass\n            f.foo = "bar"\n        '
        code2 = '\n            from m1 import f\n            x = f.__dict__\n            x["foo"] = "baz"\n\n        '
        self.assertError([(code1, 'm1'), (code2, 'm2')], 'function.__dict__ from module m1 is modified by m2')