"""Tests for --quick."""
from pytype.tests import test_base
from pytype.tests import test_utils

def make_quick(func):
    if False:
        i = 10
        return i + 15

    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['quick'] = True
        return func(*args, **kwargs)
    return wrapper

class QuickTest(test_base.BaseTest):
    """Tests for --quick."""

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        for method in ('Check', 'CheckWithErrors', 'Infer', 'InferWithErrors'):
            setattr(cls, method, make_quick(getattr(cls, method)))

    def test_multiple_returns(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def add(x: int, y: int) -> int: ...\n        def add(x: int,  y: float) -> float: ...\n      ')
            self.Check('\n        import foo\n        def f1():\n          f2()\n        def f2() -> int:\n          return foo.add(42, f3())\n        def f3():\n          return 42\n      ', pythonpath=[d.path])

    def test_multiple_returns_container(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        def concat(x: int, y: int) -> Tuple[int, int]: ...\n        def concat(x: int, y: float) -> Tuple[int, float]: ...\n      ')
            self.Check('\n        from typing import Tuple\n        import foo\n        def f1():\n          f2()\n        def f2() -> Tuple[int, int]:\n          return foo.concat(42, f3())\n        def f3():\n          return 42\n      ', pythonpath=[d.path])

    def test_noreturn(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import NoReturn\n\n      class A:\n        pass\n\n      class B:\n        def _raise_notimplemented(self) -> NoReturn:\n          raise NotImplementedError()\n        def f(self, x):\n          if __random__:\n            outputs = 42\n          else:\n            self._raise_notimplemented()\n          return outputs\n        def g(self):\n          outputs = self.f(A())\n    ')

    def test_use_return_annotation(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __init__(self):\n          self.x = 3\n      class Bar:\n        def __init__(self):\n          self.f()\n        def f(self):\n          assert_type(self.g().x, int)\n        def g(self) -> Foo:\n          return Foo()\n    ')

    def test_use_return_annotation_with_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import List, TypeVar\n      T = TypeVar('T')\n      class Foo:\n        def __init__(self):\n          x = self.f()\n          assert_type(x, List[int])\n        def f(self):\n          return self.g(0)\n        def g(self, x: T) -> List[T]:\n          return [x]\n    ")

    def test_use_return_annotation_on_new(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __new__(cls) -> "Foo":\n          self = cls()\n          self.x = __any_object__\n          return self\n        def __init__(self):\n          self.y = 0\n      def f():\n        foo = Foo()\n        assert_type(foo.x, "Any")\n        assert_type(foo.y, "int")\n    ')

    def test_async(self):
        if False:
            print('Hello World!')
        self.Check('\n      async def f1() -> None:\n        await f2()\n      async def f2() -> None:\n        await f3()\n      async def f3() -> None:\n        pass\n    ')

    def test_typevar_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Sequence, TypeVar\n\n      class TestClass(int):\n        def __init__(self):\n          pass\n\n      _T = TypeVar('_T', bound=int)\n      def transform(t: _T) -> _T:\n        return t\n\n      def last_after_transform(t: Sequence[TestClass]) -> TestClass:\n        arr = [transform(val) for val in t]\n        return arr.pop(0)\n    ")

    def test_type_of_typevar(self):
        if False:
            return 10
        self.Check("\n      from typing import Type, TypeVar\n      T = TypeVar('T', str, int)\n      def f(x: Type[T]) -> T:\n        return x()\n      def g(x: Type[T]) -> T:\n        return f(x)\n      def h():\n        return g(int)\n    ")
if __name__ == '__main__':
    test_base.main()