from compiler.pycodegen import PythonCodeGenerator
from compiler.static.types import ParamStyle
from .common import StaticTestBase

class PerfLintTests(StaticTestBase):

    def test_two_starargs(self) -> None:
        if False:
            return 10
        codestr = '\n        def f(x: int, y: int, z: int) -> int:\n            return x + y + z\n\n        a = [1, 2]\n        b = [3]\n        f(*a, *b)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Multiple *args prevents more efficient static call', at='f(*a, *b)'))

    def test_positional_after_starargs(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        def f(x: int, y: int, z: int) -> int:\n            return x + y + z\n\n        a = [1, 2]\n        f(*a, 3)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Positional arg after *args prevents more efficient static call', at='f(*a, 3)'))

    def test_multiple_kwargs(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def f(x: int, y: int, z: int) -> int:\n            return x + y + z\n\n        a = {{"x": 1, "y": 2}}\n        b = {{"z": 3}}\n        f(**a, **b)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Multiple **kwargs prevents more efficient static call', at='f(**a, **b)'))

    def test_starargs_and_default(self) -> None:
        if False:
            return 10
        codestr = '\n        def f(x: int, y: int, z: int = 0) -> int:\n            return x + y + z\n\n        a = [3]\n        f(1, 2, *a)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Passing *args to function with default values prevents more efficient static call', at='f(1, 2, *a)'))

    def test_kwonly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def f(*, x: int = 0) -> int:\n            return x\n\n        f(1)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Keyword-only args in called function prevents more efficient static call', at='f(1)'))

    def test_load_attr_dynamic(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        def foo():\n            return 42\n        a = foo()\n        a.b\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match("Define the object's class in a Static Python module for more efficient attribute load", at='a.b'))

    def test_load_attr_dynamic_base(self) -> None:
        if False:
            return 10
        codestr = '\n        def foo():\n            return 42\n        B = foo()\n        class C(B):\n            pass\n\n        def func():\n            c = C()\n            c.a\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Make the base class of <module>.C that defines attribute a static for more efficient attribute load', at='c.a'))

    def test_store_attr_dynamic(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def foo():\n            return 0\n        a, c = foo(), foo()\n        a.b = c\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match("Define the object's class in a Static Python module for more efficient attribute store", at='a.b = c'))

    def test_store_attr_dynamic_base(self) -> None:
        if False:
            return 10
        codestr = '\n        def foo():\n            return 0\n        B = foo()\n        class C(B):\n            pass\n\n        def f():\n            c = C()\n            c.a = 1\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Make the base class of <module>.C that defines attribute a static for more efficient attribute store', at='c.a'))

    def test_nonfinal_property_load(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            @property\n            def a(self) -> int:\n                return 0\n\n        def func(c: C):\n            c.a\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Getter for property a can be overridden. Make method or class final for more efficient property load', at='c.a'))

    def test_property_setter_no_warning(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            @property\n            def a(self) -> int:\n                return 0\n\n            @a.setter\n            def a(self, value: int) -> None:\n                pass\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_nonfinal_property_store(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            @property\n            def a(self) -> int:\n                return 0\n\n            @a.setter\n            def a(self, value: int) -> None:\n                pass\n\n        def func():\n            c = C()\n            c.a = 1\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Setter for property a can be overridden. Make method or class final for more efficient property store', at='c.a = 1'))

    def test_nonfinal_method_call(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            def add1(self, n: int) -> int:\n                return n + 1\n\n        def foo(c: C) -> None:\n            c.add1(10)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Method add1 can be overridden. Make method or class final for more efficient call', at='c.add1(10)'))

    def test_final_class_method_call(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        @final\n        class C:\n            def add1(self, n: int) -> int:\n                return n + 1\n\n        c = C()\n        c.add1(10)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_final_method_method_call(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            @final\n            def add1(self, n: int) -> int:\n                return n + 1\n\n        c = C()\n        c.add1(10)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_nonfinal_classmethod_call(self) -> None:
        if False:
            return 10
        codestr = '\n        class C:\n            @classmethod\n            def add1(cls, n: int) -> int:\n                return n + 1\n\n            @classmethod\n            def add2(cls, n: int) -> int:\n                return cls.add1(n) + 1\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('Method add1 can be overridden. Make method or class final for more efficient call', at='cls.add1(n)'))

    def test_final_class_classmethod_call(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        @final\n        class C:\n            @classmethod\n            def add1(cls, n: int) -> int:\n                return n + 1\n\n        C.add1(10)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_final_method_classmethod_call(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        class C:\n            @final\n            @classmethod\n            def add1(cls, n: int) -> int:\n                return n + 1\n\n        C.add1(10)\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_dataclass_dynamic_base(self) -> None:
        if False:
            return 10
        codestr = '\n        class SuperClass:\n            pass\n        '
        with self.in_module(codestr, code_gen=PythonCodeGenerator) as nonstatic_mod:
            codestr = f'\n            from dataclasses import dataclass\n            from {nonstatic_mod.__name__} import SuperClass\n\n            @dataclass\n            class C(SuperClass):\n                x: int\n            '
        errors = self.perf_lint(codestr)
        errors.check_warnings(errors.match('C has a dynamic base', at='dataclass'))

    def test_self_missing_annotation_no_warning(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        @final\n        class C:\n            def foo(self) -> int:\n                return 42\n\n        C().foo()\n        '
        errors = self.perf_lint(codestr)
        errors.check_warnings()

    def test_missing_arg_annotation(self) -> None:
        if False:
            print('Hello World!')
        for (style, args) in ((ParamStyle.NORMAL, 'missing'), (ParamStyle.POSONLY, 'missing, /'), (ParamStyle.KWONLY, '*, missing')):
            with self.subTest(param_style=style.name):
                codestr = f'\n                def add1({args}) -> int:\n                    return missing + 1\n                '
                errors = self.perf_lint(codestr)
                errors.check_warnings(errors.match('Missing type annotation', at='missing'))