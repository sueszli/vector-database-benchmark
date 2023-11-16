"""Tests for handling PYI code."""
from pytype.tests import test_base
from pytype.tests import test_utils

class PYITest(test_base.BaseTest):
    """Tests for PYI."""

    def test_module_parameter(self):
        if False:
            return 10
        'This test that types.ModuleType works.'
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        import types\n        def f(x: types.ModuleType = ...) -> None: ...\n      ')
            self.Check('\n        import os\n        import mod\n\n        mod.f(os)\n        ', pythonpath=[d.path])

    def test_optional(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def f(x: int = ...) -> None: ...\n      ')
            ty = self.Infer('\n        import mod\n        def f():\n          return mod.f()\n        def g():\n          return mod.f(3)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import mod\n        def f() -> NoneType: ...\n        def g() -> NoneType: ...\n      ')

    def test_solve(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def f(node: int, *args, **kwargs) -> str: ...\n      ')
            ty = self.Infer('\n        import mod\n        def g(x):\n          return mod.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import mod\n        def g(x) -> str: ...\n      ')

    def test_typing(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        from typing import Any, IO, List, Optional\n        def split(s: Optional[int]) -> List[str]: ...\n      ')
            ty = self.Infer('\n        import mod\n        def g(x):\n          return mod.split(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import mod\n        from typing import List\n        def g(x) -> List[str]: ...\n      ')

    def test_classes(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('classes.pyi', '\n        class A:\n          def foo(self) -> A: ...\n        class B(A):\n          pass\n      ')
            ty = self.Infer('\n        import classes\n        x = classes.B().foo()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import classes\n        x = ...  # type: classes.A\n      ')

    def test_empty_module(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('vague.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            ty = self.Infer('\n        import vague\n        x = vague.foo + vague.bar\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import vague\n        from typing import Any\n        x = ...  # type: Any\n      ')

    def test_decorators(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('decorated.pyi', '\n        class A:\n          @staticmethod\n          def u(a, b) -> int: ...\n          @classmethod\n          def v(cls, a, b) -> int: ...\n          def w(self, a, b) -> int: ...\n      ')
            ty = self.Infer('\n        import decorated\n        u = decorated.A.u(1, 2)\n        v = decorated.A.v(1, 2)\n        a = decorated.A()\n        x = a.u(1, 2)\n        y = a.v(1, 2)\n        z = a.w(1, 2)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import decorated\n        a = ...  # type: decorated.A\n        u = ...  # type: int\n        v = ...  # type: int\n        x = ...  # type: int\n        y = ...  # type: int\n        z = ...  # type: int\n      ')

    def test_pass_pyi_classmethod(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A:\n          @classmethod\n          def v(cls) -> float: ...\n          def w(self, x: classmethod) -> int: ...\n      ')
            ty = self.Infer('\n        import a\n        u = a.A().w(a.A.v)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        u = ...  # type: int\n      ')

    def test_optional_parameters(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def parse(source, filename = ..., mode = ..., *args, **kwargs) -> int: ...\n      ')
            ty = self.Infer('\n        import a\n        u = a.parse("True")\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        u = ...  # type: int\n      ')

    def test_optimize(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any\n        class Bar(dict[Any, int]): ...\n      ')
            ty = self.Infer("\n      import a\n      def f(foo, bar):\n        return __any_object__[1]\n      def g():\n        out = f('foo', 'bar')\n        out = out.split()\n      ", pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Any\n        def f(foo, bar) -> Any: ...\n        def g() -> NoneType: ...\n      ')

    def test_iterable(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Iterable\n        def f(l: Iterable[int]) -> int: ...\n      ')
            ty = self.Infer('\n        import a\n        u = a.f([1, 2, 3])\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        u = ...  # type: int\n      ')

    def test_object(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def make_object() -> object: ...\n      ')
            ty = self.Infer('\n        import a\n        def f(x=None):\n          x = a.make_object()\n          z = x - __any_object__  # type: ignore\n          z + __any_object__\n          return True\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f(x=...) -> bool: ...\n      ')

    def test_callable(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        from typing import Callable\n        def process_function(func: Callable[..., Any]) -> None: ...\n      ')
            ty = self.Infer('\n        import foo\n        def bar():\n          pass\n        x = foo.process_function(bar)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import foo\n        from typing import Any\n        def bar() -> Any: ...   # 'Any' because deep=False\n        x = ...  # type: NoneType\n      ")

    def test_hex(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x = hex(4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: str\n    ')

    def test_base_class(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Generic, TypeVar\n        S = TypeVar('S')\n        T = TypeVar('T')\n        class A(Generic[S]):\n          def bar(self, s: S) -> S: ...\n        class B(Generic[T], A[T]): ...\n        class C(A[int]): ...\n        class D:\n          def baz(self) -> int: ...\n      ")
            ty = self.Infer('\n        import foo\n        def f(x):\n          return x.bar("foo")\n        def g(x):\n          return x.bar(3)\n        def h(x):\n          return x.baz()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x) -> Any: ...\n        def g(x) -> Any: ...\n        def h(x) -> Any: ...\n      ')

    def test_old_style_class_object_match(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        def f(x) -> Any: ...\n        class Foo: pass\n      ')
            ty = self.Infer('\n        import foo\n        def g():\n          return foo.f(foo.Foo())\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def g() -> Any: ...\n      ')

    def test_identity(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def f(x: T) -> T: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f(3)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: int\n      ')

    def test_import_function_template(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d1:
            d1.create_file('foo.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def f(x: T) -> T: ...\n      ')
            with test_utils.Tempdir() as d2:
                d2.create_file('bar.pyi', '\n          import foo\n          f = foo.f\n        ')
                ty = self.Infer('\n          import bar\n          x = bar.f("")\n        ', pythonpath=[d1.path, d2.path])
                self.assertTypesMatchPytd(ty, '\n          import bar\n          x = ...  # type: str\n        ')

    def test_multiple_getattr(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            (ty, errors) = self.InferWithErrors('\n        from foo import *\n        from bar import *  # Nonsense import generates a top-level __getattr__  # import-error[e]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            self.assertErrorRegexes(errors, {'e': 'bar'})

    def test_pyi_list_item(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        lst = ...  # type: list\n        def f(x: int) -> str: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.f(a.lst[0])\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: str\n      ')

    def test_keyword_only_args(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any\n        def foo(x: str, *y: Any, z: complex = ...) -> int: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.foo("foo %d %d", 3, 3)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n      ')

    def test_posarg(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def get_pos(x: T, *args: int, z: int, **kws: int) -> T: ...\n      ')
            ty = self.Infer('\n        import a\n        v = a.get_pos("foo", 3, 4, z=5)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        v = ...  # type: str\n      ')

    def test_kwonly_arg(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def get_kwonly(x: int, *args: int, z: T, **kwargs: int) -> T: ...\n      ')
            ty = self.Infer('\n        import a\n        v = a.get_kwonly(3, 4, z=5j)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        v = ...  # type: complex\n      ')

    def test_starargs(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Dict, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        def foo(a: K, *b, c: V, **d) -> Dict[K, V]: ...\n      ')
            (ty, errors) = self.InferWithErrors('\n        import foo\n        a = foo.foo(*tuple(), **dict())  # missing-parameter[e1]\n        b = foo.foo(*(1,), **{"c": 3j})\n        c = foo.foo(*(1,))  # missing-parameter[e2]\n        d = foo.foo(*(), **{"d": 3j})  # missing-parameter[e3]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Dict\n        a = ...  # type: Any\n        b = ...  # type: Dict[int, complex]\n        c = ...  # type: Any\n        d = ...  # type: Any\n      ')
            self.assertErrorRegexes(errors, {'e1': '\\ba\\b', 'e2': '\\bc\\b', 'e3': '\\ba\\b'})

    def test_union_with_superclass(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A1(): pass\n        class A2(A1): pass\n        class A3(A2): pass\n      ')
            ty = self.Infer("\n        import a\n        def f(x):\n          # Constrain the type of x so it doesn't pull everything into our pytd\n          x = x + 16\n          if x:\n            return a.A1()\n          else:\n            return a.A3()\n      ", pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f(x) -> a.A1: ...\n      ')

    def test_builtins_module(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        import builtins\n        x = ...  # type: builtins.int\n      ')
            ty = self.Infer('\n        import a\n        x = a.x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n      ')

    def test_frozenset(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any, FrozenSet, Set\n        x = ...  # type: FrozenSet[str]\n        y = ...  # type: Set[Any]\n      ')
            ty = self.Infer('\n        import a\n        x = a.x - a.x\n        y = a.x - a.y\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import FrozenSet\n        import a\n        x = ...  # type: FrozenSet[str]\n        y = ...  # type: FrozenSet[str]\n      ')

    def test_raises(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(raises): ...\n      ')
            self.Check('import foo', pythonpath=[d.path])

    def test_typevar_conflict(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, Sequence\n        class A(List[int], Sequence[str]): ...\n      ')
            (ty, _) = self.InferWithErrors('\n        import foo  # pyi-error\n        x = [] + foo.A()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        foo = ...  # type: Any\n        x = ...  # type: list\n      ')

    def test_same_typevar_name(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class MySupportsAbs(Generic[T]): ...\n        class MyContextManager(Generic[T]):\n          def __enter__(self) -> T: ...\n        class Foo(MySupportsAbs[float], MyContextManager[Foo]): ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.Foo().__enter__()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        v = ...  # type: foo.Foo\n      ')

    def test_type_param_in_mutation(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        T2 = TypeVar("T2")\n        class Bar(Generic[T]):\n          def bar(self, x:T2):\n            self = Bar[T2]\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.Bar()\n        x.bar(10)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: foo.Bar[int]\n      ')

    def test_bad_type_param_in_mutation(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        T2 = TypeVar("T2")\n        class Bar(Generic[T]):\n          def quux(self, x: T2): ...\n          def bar(self):\n            self = Bar[T2]\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo  # pyi-error[e]\n        x = foo.Bar()\n        x.bar()\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'T2'})

    def test_star_import(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        x = ...  # type: int\n        T = TypeVar("T")\n        class A: ...\n        def f(x: T) -> T: ...\n        B = A\n      ')
            d.create_file('bar.pyi', '\n        from foo import *\n      ')
            self.Check('\n        import bar\n        bar.x\n        bar.T\n        bar.A\n        bar.f\n        bar.B\n      ', pythonpath=[d.path])

    def test_star_import_value(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        T = TypeVar("T")\n        def f(x: T) -> T: ...\n        class Foo: pass\n      ')
            d.create_file('bar.pyi', '\n        from foo import *\n      ')
            ty = self.Infer('\n        import bar\n        v1 = bar.Foo()\n        v2 = bar.f("")\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import bar\n        v1 = ...  # type: foo.Foo\n        v2 = ...  # type: str\n      ')

    def test_star_import_getattr(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            d.create_file('bar.pyi', '\n        from foo import *\n      ')
            self.Check('\n        import bar\n        bar.rumpelstiltskin\n      ', pythonpath=[d.path])

    def test_alias(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: Foo): ...\n        g = f\n        class Foo: ...\n      ')
            self.Check('import foo', pythonpath=[d.path])

    def test_custom_binary_operator(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          def __sub__(self, other) -> str: ...\n        class Bar(Foo):\n          def __rsub__(self, other) -> int: ...\n      ')
            self.Check('\n        import foo\n        (foo.Foo() - foo.Bar()).real\n      ', pythonpath=[d.path])

    def test_parameterized_any(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        x = ...  # type: Any\n        y = ...  # type: x[Any]\n      ')
            self.Check('\n        import foo\n      ', pythonpath=[d.path])

    def test_parameterized_external_any(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        x = ...  # type: Any\n      ')
            d.create_file('bar.pyi', '\n        import foo\n        from typing import Any\n        x = ...  # type: foo.x[Any]\n      ')
            self.Check('\n        import bar\n      ', pythonpath=[d.path])

    def test_parameterized_alias(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        x = ...  # type: Any\n      ')
            d.create_file('bar.pyi', '\n        import foo\n        from typing import Any\n        x = foo.x[Any]\n      ')
            self.Check('\n        import bar\n      ', pythonpath=[d.path])

    def test_anything_constant(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        Foo = ...  # type: Any\n      ')
            d.create_file('bar.pyi', '\n        import foo\n        def f(x: foo.Foo) -> None: ...\n      ')
            self.Check('\n        import bar\n        bar.f(42)\n      ', pythonpath=[d.path])

    def test_alias_staticmethod(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          @staticmethod\n          def t(a: str) -> None: ...\n      ')
            ty = self.Infer('\n        import foo\n        ta = foo.A.t\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Callable\n        ta = ...  # type: Callable[[str], None]\n        ')

    def test_alias_constant(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          const = ...  # type: int\n        Const = Foo.const\n      ')
            ty = self.Infer('\n        import foo\n        Const = foo.Const\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        Const = ...  # type: int\n      ')

    def test_alias_method(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          def f(self) -> int: ...\n        Func = Foo.f\n      ')
            ty = self.Infer('\n        import foo\n        Func = foo.Func\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def Func(self) -> int: ...\n      ')

    def test_alias_aliases(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          a1 = const\n          a2 = f\n          const = ...  # type: int\n          def f(self) -> int: ...\n        Const = Foo.a1\n        Func = Foo.a2\n      ')
            ty = self.Infer('\n        import foo\n        Const = foo.Const\n        Func = foo.Func\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        Const = ...  # type: int\n        def Func(self) -> int: ...\n      ')

    def test_generic_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Sequence, MutableSequence, TypeVar\n        TFoo = TypeVar("TFoo", bound=Foo)\n        class Foo(Sequence[Foo]):\n          def __getitem__(self: TFoo, i: int) -> TFoo: ...\n        class Bar(Foo, MutableSequence[Bar]): ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.Bar()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        v = ...  # type: foo.Bar\n      ')

    def test_dot_import(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo/a.pyi', 'class A: ...')
            d.create_file('foo/b.pyi', '\n        from . import a\n        X = a.A\n      ')
            ty = self.Infer('\n        from foo import b\n        a = b.X()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import b\n        a = ...  # type: foo.a.A\n      ')

    def test_dot_dot_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo/a.pyi', 'class A: ...')
            d.create_file('foo/bar/b.pyi', '\n        from .. import a\n        X = a.A\n      ')
            ty = self.Infer('\n        from foo.bar import b\n        a = b.X()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo.bar import b\n        a = ...  # type: foo.a.A\n      ')

    def test_typing_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import typing as _typing\n        def f(x: _typing.Tuple[str, str]) -> None: ...\n      ')
            self.Check('import foo', pythonpath=[d.path])

    def test_parameterize_builtin_tuple(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: tuple[int]) -> tuple[int, int]: ...\n      ')
            (ty, _) = self.InferWithErrors('\n        import foo\n        foo.f((0, 0))  # wrong-arg-types\n        x = foo.f((0,))\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Tuple\n        x: Tuple[int, int]\n      ')

    def test_implicit_mutation(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Generic, TypeVar\n        T = TypeVar('T')\n        class Foo(Generic[T]):\n          def __init__(self, x: T) -> None: ...\n      ")
            ty = self.Infer('\n        import foo\n        x = foo.Foo(x=0)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x: foo.Foo[int]\n      ')

    def test_import_typevar_for_property(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('_typeshed.pyi', "\n        from typing import TypeVar\n        Self = TypeVar('Self')\n      ")
            d.create_file('foo.pyi', '\n        from _typeshed import Self\n        class Foo:\n          @property\n          def foo(self: Self) -> Self: ...\n      ')
            self.Check('\n        import foo\n        assert_type(foo.Foo().foo, foo.Foo)\n      ', pythonpath=[d.path])

    def test_bad_annotation(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('bad.pyi', '\n        def f() -> None: ...\n        class X:\n          x: f\n      ')
            self.CheckWithErrors('\n        import bad  # pyi-error\n      ', pythonpath=[d.path])

    def test_nonexistent_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('bad.pyi', '\n        import nonexistent\n        x = nonexistent.x\n      ')
            err = self.CheckWithErrors('\n        import bad  # pyi-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorSequences(err, {'e': ["Couldn't import pyi", 'nonexistent', 'referenced from', 'bad']})
if __name__ == '__main__':
    test_base.main()