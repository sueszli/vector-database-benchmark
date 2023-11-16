"""Tests for the analysis phase matcher (match_var_against_type)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class MatchTest(test_base.BaseTest):
    """Tests for matching types."""

    def test_type_against_callable(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Callable\n        def f(x: Callable) -> str: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f():\n          return foo.f(int)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> str: ...\n      ')

    def test_match_static(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      s = {1}\n      def f(x):\n        # set.intersection is a static method:\n        return s.intersection(x)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Set\n      s = ...  # type: Set[int]\n\n      def f(x) -> Set[int]: ...\n    ')

    def test_generic_hierarchy(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Iterable\n        def f(x: Iterable[str]) -> str: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.f(["a", "b", "c"])\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: str\n      ')

    def test_generic(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, Iterable\n        K = TypeVar("K")\n        V = TypeVar("V")\n        Q = TypeVar("Q")\n        class A(Iterable[V], Generic[K, V]): ...\n        class B(A[K, V]):\n          def __init__(self):\n            self = B[bool, str]\n        def f(x: Iterable[Q]) -> Q: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.f(a.B())\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: str\n      ')

    def test_match_identity_function(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def f(x: T) -> T: ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.f(__any_object__)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import foo\n        v = ...  # type: Any\n      ')

    def test_callable_return(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Callable, TypeVar\n        T = TypeVar("T")\n        def foo(func: Callable[[], T]) -> T: ...\n      ')
            self.Check('\n        import foo\n        class Foo:\n          def __init__(self):\n            self.x = 42\n        foo.foo(Foo).x\n      ', pythonpath=[d.path])

    def test_callable_union_return(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Callable, TypeVar, Union\n        T1 = TypeVar("T1")\n        T2 = TypeVar("T2")\n        def foo(func: Callable[[], T1]) -> Union[T1, T2]: ...\n      ')
            self.Check('\n        import foo\n        class Foo:\n          def __init__(self):\n            self.x = 42\n        v = foo.foo(Foo)\n        if isinstance(v, Foo):\n          v.x\n      ', pythonpath=[d.path])

    def test_any_base_class(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        class Foo(Any): pass\n        class Bar: pass\n        def f(x: Bar) -> None: ...\n      ')
            self.Check('\n        import foo\n        foo.f(foo.Foo())\n      ', pythonpath=[d.path])

    def test_maybe_parameterized(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections.abc\n      class Foo(collections.abc.MutableMapping):\n        pass\n      def f(x: Foo):\n        dict.__delitem__(x, __any_object__)  # pytype: disable=wrong-arg-types\n    ')
if __name__ == '__main__':
    test_base.main()