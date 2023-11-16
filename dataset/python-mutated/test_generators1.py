"""Tests for generators."""
from pytype.tests import test_base

class GeneratorTest(test_base.BaseTest):
    """Tests for iterators, generators, coroutines, and yield."""

    def test_next(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f():\n        return next(i for i in [1,2,3])\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_list(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      y = list(x for x in [1, 2, 3])\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      y = ...  # type: List[int]\n    ')

    def test_reuse(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      y = list(x for x in [1, 2, 3])\n      z = list(x for x in [1, 2, 3])\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      y = ...  # type: List[int]\n      z = ...  # type: List[int]\n    ')

    def test_next_with_default(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        return next((i for i in [1,2,3]), None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f() -> Union[int, NoneType]: ...\n    ')

    def test_iter_match(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def bar(self):\n          for x in __any_object__:\n            return x\n        def __iter__(self):\n          return generator()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      class Foo:\n        def bar(self) -> Any: ...\n        def __iter__(self) -> Generator[nothing, nothing, nothing]: ...\n    ')

    def test_coroutine_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def foo(self):\n        yield 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def foo(self) -> Generator[int, Any, None]: ...\n    ')

    def test_iteration_of_getitem(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __getitem__(self, key):\n          return "hello"\n\n      def foo(self):\n        for x in Foo():\n          return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Foo:\n        def __getitem__(self, key) -> str: ...\n      def foo(self) -> Union[None, str]: ...\n    ')

    def test_unpacking_of_getitem(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        def __getitem__(self, pos):\n          if pos < 3:\n            return pos\n          else:\n            raise StopIteration\n      x, y, z = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, TypeVar\n      _T0 = TypeVar("_T0")\n      class Foo:\n        def __getitem__(self, pos: _T0) -> _T0: ...\n      x = ...  # type: int\n      y = ...  # type: int\n      z = ...  # type: int\n    ')

    def test_none_check(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        x = None if __random__ else 42\n        if x:\n          yield x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def f() -> Generator[int, Any, None]: ...\n    ')

    def test_yield_type(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import Generator\n      def f(x):\n        if x == 1:\n          yield 1\n        else:\n          yield "1"\n\n      x = f(2)\n      y = f(1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator, Union\n      def f(x) -> Generator[Union[int, str], Any, None]: ...\n      x = ...  # type: Generator[str, Any, None]\n      y = ...  # type: Generator[int, Any, None]\n    ')
if __name__ == '__main__':
    test_base.main()