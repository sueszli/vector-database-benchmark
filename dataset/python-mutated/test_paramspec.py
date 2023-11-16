"""PEP 612 tests."""
from pytype.tests import test_base
from pytype.tests import test_utils

@test_utils.skipBeforePy((3, 10), 'ParamSpec is new in 3.10')
class ParamSpecTest(test_base.BaseTest):
    """Tests for ParamSpec."""

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import ParamSpec\n      P = ParamSpec("P")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import ParamSpec\n      P = ParamSpec("P")\n    ')

    def test_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'P = ParamSpec("P")')
            ty = self.Infer('\n        from a import P\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import ParamSpec\n        P = ParamSpec("P")\n      ')

    def test_invalid(self):
        if False:
            while True:
                i = 10
        (ty, errors) = self.InferWithErrors('\n      from typing import ParamSpec\n      T = ParamSpec()  # invalid-typevar[e1]\n      T = ParamSpec("T")  # ok\n      T = ParamSpec(42)  # invalid-typevar[e2]\n      T = ParamSpec(str())  # invalid-typevar[e3]\n      T = ParamSpec("T", str, int if __random__ else float)  # invalid-typevar[e4]\n      T = ParamSpec("T", 0, float)  # invalid-typevar[e5]\n      T = ParamSpec("T", str)  # invalid-typevar[e6]\n      # pytype: disable=not-supported-yet\n      S = ParamSpec("S", covariant=False)  # ok\n      T = ParamSpec("T", covariant=False)  # duplicate ok\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import ParamSpec\n      S = ParamSpec("S")\n      T = ParamSpec("T")\n    ')
        self.assertErrorRegexes(errors, {'e1': 'wrong arguments', 'e2': 'Expected.*str.*Actual.*int', 'e3': 'constant str', 'e4': 'constraint.*Must be constant', 'e5': 'Expected.*_1:.*type.*Actual.*_1: int', 'e6': '0 or more than 1'})

    def test_print_args(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import ParamSpec\n      S = ParamSpec("S", bound=float, covariant=True)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import ParamSpec\n      S = ParamSpec("S", bound=float)\n    ')

    def test_paramspec_in_def(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import Callable, ParamSpec\n      P = ParamSpec("P")\n\n      def f(x: Callable[P, int]) -> Callable[P, int]:\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, ParamSpec\n      P = ParamSpec("P")\n\n      def f(x: Callable[P, int]) -> Callable[P, int]: ...\n    ')

    def test_concatenate_in_def(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Callable, Concatenate, ParamSpec\n      P = ParamSpec("P")\n\n      def f(x: Callable[Concatenate[int, P], int]) -> Callable[P, int]:\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, Concatenate, ParamSpec\n      P = ParamSpec("P")\n\n      def f(x: Callable[Concatenate[int, P], int]) -> Callable[P, int]: ...\n    ')

    def test_drop_param(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Callable, Concatenate, ParamSpec\n\n      P = ParamSpec("P")\n\n      def f(x: Callable[Concatenate[int, P], int], y: int) -> Callable[P, int]:\n        return lambda k: x(y, k)\n\n      def g(x: int, y: str) -> int:\n        return 42\n\n      a = f(g, 1)\n      assert_type(a, Callable[[str], int])\n    ')

    def test_add_param(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable, Concatenate, ParamSpec\n\n      P = ParamSpec("P")\n\n      def f(x: Callable[P, int]) -> Callable[Concatenate[int, P], int]:\n        return lambda p, q: x(q)\n\n      def g(x: str) -> int:\n        return 42\n\n      a = f(g)\n      assert_type(a, Callable[[int, str], int])\n    ')

    def test_change_return_type(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Callable, Concatenate, ParamSpec\n\n      P = ParamSpec("P")\n\n      def f(x: Callable[P, int]) -> Callable[P, str]:\n        return lambda p: str(x(p))\n\n      def g(x: int) -> int:\n        return 42\n\n      a = f(g)\n      assert_type(a, Callable[[int], str])\n    ')

    def test_typevar(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Callable, Concatenate, List, ParamSpec, TypeVar\n\n      P = ParamSpec("P")\n      T = TypeVar(\'T\')\n\n      def f(x: Callable[P, T]) -> Callable[P, List[T]]:\n        def inner(p):\n          return [x(p)]\n        return inner\n\n      def g(x: int) -> int:\n        return 42\n\n      def h(x: bool) -> str:\n        return \'42\'\n\n      a = f(g)\n      assert_type(a, Callable[[int], List[int]])\n      b = f(h)\n      assert_type(b, Callable[[bool], List[str]])\n    ')

    def test_args_and_kwargs(self):
        if False:
            return 10
        self.Check('\n      from typing import ParamSpec, Callable, TypeVar\n\n      P = ParamSpec("P")\n      T = TypeVar("T")\n\n      def decorator(f: Callable[P, T]) -> Callable[P, T]:\n        def foo(*args: P.args, **kwargs: P.kwargs) -> T:\n          return f(*args, **kwargs)\n        return foo\n\n      def g(x: int, y: str) -> bool:\n        return False\n\n      a = decorator(g)\n      b = a(1, \'2\')\n      assert_type(b, bool)\n    ')
_DECORATOR_PYI = '\n  from typing import TypeVar, ParamSpec, Callable, List\n\n  T = TypeVar("T")\n  P = ParamSpec("P")\n\n  def decorator(fn: Callable[P, T]) -> Callable[P, List[T]]: ...\n'

@test_utils.skipBeforePy((3, 10), 'ParamSpec is new in 3.10')
class PyiParamSpecTest(test_base.BaseTest):
    """Tests for ParamSpec imported from pyi files."""

    def test_decorator(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', _DECORATOR_PYI)]):
            (ty, _) = self.InferWithErrors("\n        import foo\n\n        class A:\n          pass\n\n        @foo.decorator\n        def h(a: A, b: str) -> int:\n          return 10\n\n        p = h(A(), b='2')\n        q = h(1, 2)  # wrong-arg-types\n      ")
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import List, Any\n\n      p: List[int]\n      q: Any\n\n      class A: ...\n\n      def h(a: A, b: str) -> List[int]: ...\n   ')

    def test_method_decoration(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', _DECORATOR_PYI)]):
            (ty, _) = self.InferWithErrors('\n        import foo\n\n        class A:\n          pass\n\n        class B:\n          @foo.decorator\n          def h(a: A, b: str) -> int:\n            return 10\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import List, Any\n\n      class A: ...\n\n      class B:\n        def h(a: A, b: str) -> List[int]: ...\n   ')

    def test_multiple_decorators(self):
        if False:
            for i in range(10):
                print('nop')
        "Check that we don't cache the decorator type params."
        with self.DepTree([('foo.pyi', _DECORATOR_PYI)]):
            self.Check('\n        import foo\n\n        @foo.decorator\n        def f(x) -> str:\n          return "a"\n\n        @foo.decorator\n        def g() -> int:\n          return 42\n\n        def h() -> list[str]:\n          return f(10)\n\n        def k() -> list[int]:\n          return g()\n      ')

    def test_imported_paramspec(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', _DECORATOR_PYI)]):
            (ty, _) = self.InferWithErrors("\n        from foo import decorator\n\n        class A:\n          pass\n\n        @decorator\n        def h(a: A, b: str) -> int:\n          return 10\n\n        p = h(A(), b='2')\n        q = h(1, 2)  # wrong-arg-types\n      ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, List, ParamSpec, TypeVar, Any\n\n      p: List[int]\n      q: Any\n\n      P = ParamSpec('P')\n      T = TypeVar('T')\n\n      class A: ...\n\n      def decorator(fn: Callable[P, T]) -> Callable[P, List[T]]: ...\n      def h(a: A, b: str) -> List[int]: ...\n   ")

    def test_concatenate(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      from typing import TypeVar, ParamSpec, Concatenate, Callable\n\n      T = TypeVar("T")\n      P = ParamSpec("P")\n\n      def change_arg(fn: Callable[Concatenate[int, P], T]) -> Callable[Concatenate[str, P], T]: ...\n      def drop_arg(fn: Callable[Concatenate[int, P], T]) -> Callable[P, T]: ...\n      def add_arg(fn: Callable[P, T]) -> Callable[Concatenate[int, P], T]: ...\n      def mismatched(fn: Callable[Concatenate[str, P], T]) -> Callable[Concatenate[str, P], T]: ...\n    ')]):
            (ty, err) = self.InferWithErrors('\n        import foo\n\n        @foo.change_arg\n        def f(a: int, b: str) -> int:\n          return 10\n\n        @foo.drop_arg\n        def g(a: int, b: str) -> int:\n          return 10\n\n        @foo.add_arg\n        def h(a: int, b: str) -> int:\n          return 10\n\n        @foo.mismatched\n        def k(a: int, b: str) -> int:  # wrong-arg-types[e]\n          return 10\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import Any\n\n      k: Any\n\n      def f(_0: str, b: str) -> int: ...\n      def g(b: str) -> int: ...\n      def h(_0: int, /, a: int, b: str) -> int: ...\n   ')
        self.assertErrorSequences(err, {'e': ['Expected', 'fn: Callable[Concatenate[str, P], Any]', 'Actual', 'fn: Callable[[int, str], int]']})

    def test_overloaded_argument(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      from typing import TypeVar, ParamSpec, Callable, List\n\n      T = TypeVar("T")\n      P = ParamSpec("P")\n\n      def decorator(fn: Callable[P, T]) -> Callable[P, List[T]]: ...\n\n      @overload\n      def f(x: str) -> int: ...\n      @overload\n      def f(x: str, *, y: int = 0) -> int: ...\n    ')]):
            (ty, _) = self.InferWithErrors('\n        import foo\n\n        f = foo.decorator(foo.f)\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import List, overload\n\n      @overload\n      def f(x: str) -> List[int]: ...\n      @overload\n      def f(x: str, *, y: int = ...) -> List[int]: ...\n   ')

    def test_starargs(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', _DECORATOR_PYI)]):
            ty = self.Infer('\n        import foo\n\n        class A:\n          pass\n\n        class B:\n          @foo.decorator\n          def h(a: A, b: str, *args, **kwargs) -> int:\n            return 10\n\n        @foo.decorator\n        def s(*args) -> int:\n          return 10\n\n        @foo.decorator\n        def k(**kwargs) -> int:\n          return 10\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import List, Any\n\n      class A: ...\n\n      class B:\n        def h(a: A, b: str, *args, **kwargs) -> List[int]: ...\n\n      def s(*args) -> List[int]: ...\n      def k(**kwargs) -> List[int]: ...\n   ')

    def test_callable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', '\n      from typing import TypeVar, ParamSpec, Concatenate, Callable\n\n      T = TypeVar("T")\n      P = ParamSpec("P")\n\n      def add_arg(fn: Callable[P, T]) -> Callable[Concatenate[int, P], T]: ...\n    ')]):
            self.Check("\n        import foo\n        from typing import Callable, List\n\n        def f(method: Callable[[int, str], bool]):\n          a = foo.add_arg(method)\n          b = a(1, 2, '3')\n          assert_type(b, bool)\n      ")

    def test_match_callable(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', "\n      from typing import Any, Callable, ParamSpec\n      P = ParamSpec('P')\n      def f(x: Callable[P, Any]) -> Callable[P, Any]: ...\n    ")]):
            self.Check('\n        import foo\n\n        # Any function should match `Callable[P, Any]`.\n        def f0():\n          pass\n        def f1(x):\n          pass\n        def f2(x1, x2):\n          pass\n        foo.f(f0)\n        foo.f(f1)\n        foo.f(f2)\n\n        class C0:\n          def __call__(self):\n            pass\n        class C1:\n          def __call__(self, x1):\n            pass\n        class C2:\n          def __call__(self, x1, x2):\n            pass\n\n        # Any class object should match.\n        foo.f(C0)\n\n        # Any class instance with a `__call__` method should match.\n        foo.f(C0())\n        foo.f(C1())\n        foo.f(C2())\n      ')

    def test_callable_class_inference(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', "\n      from typing import Any, Callable, ParamSpec\n      P = ParamSpec('P')\n      def f(x: Callable[P, Any]) -> Callable[P, Any]: ...\n    ")]):
            ty = self.Infer('\n        import foo\n        class C:\n          def __call__(self, x: int, y) -> str:\n            return str(x)\n        f = foo.f(C())\n      ')
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        class C:\n          def __call__(self, x: int, y) -> str: ...\n        def f(x: int, y) -> Any: ...\n      ')

class ContextlibTest(test_base.BaseTest):
    """Test some more complex uses of contextlib."""

    def test_wrapper(self):
        if False:
            return 10
        self.Check('\n      import contextlib\n      import functools\n\n      from typing import Callable, ContextManager, Iterator, TypeVar\n\n      T = TypeVar("T")\n\n      class Builder:\n        def __init__(self, exit_stack: contextlib.ExitStack):\n          self._stack = exit_stack\n\n        def _enter_context(self, manager: ContextManager[T]) -> T:\n          return self._stack.enter_context(manager)\n\n      def context_manager(func: Callable[..., Iterator[T]]) -> Callable[..., T]:\n        cm_func = contextlib.contextmanager(func)\n\n        @functools.wraps(cm_func)\n        def _context_manager_wrap(self: Builder, *args, **kwargs):\n          return self._enter_context(cm_func(self, *args, **kwargs))\n\n        return _context_manager_wrap\n      ')
if __name__ == '__main__':
    test_base.main()