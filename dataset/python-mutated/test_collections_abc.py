"""Tests for collections.abc."""
from pytype.tests import test_base
from pytype.tests import test_utils

class CollectionsABCTest(test_base.BaseTest):
    """Tests for collections.abc."""

    def test_mapping(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n      class Foo(collections.abc.Mapping):\n        pass\n    ')

    def test_bytestring(self):
        if False:
            print('Hello World!')
        'Check that we handle type aliases.'
        self.Check('\n      import collections\n      x: collections.abc.ByteString\n    ')

    def test_callable(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from collections.abc import Callable\n      f: Callable[[str], str] = lambda x: x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      Callable: type\n      f: typing.Callable[[str], str]\n    ')

    def test_pyi_callable(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from collections.abc import Callable\n        def f() -> Callable[[], float]: ...\n      ')
            (ty, _) = self.InferWithErrors('\n        import foo\n        func = foo.f()\n        func(0.0)  # wrong-arg-count\n        x = func()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Callable\n        func: Callable[[], float]\n        x: float\n      ')

    def test_generator(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from collections.abc import Generator\n      def f() -> Generator[int, None, None]:\n        yield 0\n    ')

    def test_set(self):
        if False:
            print('Hello World!')
        self.Check('\n      from collections.abc import Set\n      def f() -> Set[int]:\n        return frozenset([0])\n    ')
if __name__ == '__main__':
    test_base.main()