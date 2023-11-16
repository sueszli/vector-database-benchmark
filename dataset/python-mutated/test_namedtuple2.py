"""Tests for the namedtuple implementation in collections_overlay.py."""
from pytype.tests import test_base

class NamedtupleTests(test_base.BaseTest):
    """Tests for collections.namedtuple."""

    def test_namedtuple_match(self):
        if False:
            return 10
        self.Check('\n        import collections\n        from typing import Any, Dict\n\n        X = collections.namedtuple("X", ["a"])\n\n        def GetRefillSeekerRanks() -> Dict[str, X]:\n          return {"hello": X(__any_object__)}\n        ')

    def test_namedtuple_different_name(self):
        if False:
            return 10
        with self.DepTree([('foo.py', '\n      import collections\n      X1 = collections.namedtuple("X", ["a", "b"])\n      X2 = collections.namedtuple("X", ["c", "d"])\n    ')]):
            self.Check('\n        import foo\n        def f() -> foo.X2:\n          return foo.X2(0, 0)\n      ')

    def test_namedtuple_inheritance(self):
        if False:
            print('Hello World!')
        self.Check("\n      import collections\n      class Base(collections.namedtuple('Base', ['x', 'y'])):\n        pass\n      class Foo(Base):\n        def __new__(cls, **kwargs):\n          return super().__new__(cls, **kwargs)\n      def f(x: Foo):\n        pass\n      def g(x: Foo):\n        return f(x)\n    ")

    def test_namedtuple_inheritance_expensive(self):
        if False:
            return 10
        self.Check("\n      import collections\n      class Foo(collections.namedtuple('_Foo', ['x', 'y'])):\n        pass\n      def f() -> Foo:\n        x1 = __any_object__ or None\n        x2 = __any_object__ or False\n        x3 = __any_object__ or False\n        x4 = __any_object__ or False\n        y1 = __any_object__ or None\n        y2 = __any_object__ or False\n        y3 = __any_object__ or False\n        return Foo((x1, x2, x3, x4), (y1, y2, y3))\n    ")

class NamedtupleTestsPy3(test_base.BaseTest):
    """Tests for collections.namedtuple in Python 3."""

    def test_bad_call(self):
        if False:
            i = 10
            return i + 15
        'The last two arguments are kwonly in 3.6.'
        self.InferWithErrors('\n        import collections\n        collections.namedtuple()  # missing-parameter\n        collections.namedtuple("_")  # missing-parameter\n        collections.namedtuple("_", "", True)  # wrong-arg-count\n        collections.namedtuple("_", "", True, True)  # wrong-arg-count\n        collections.namedtuple("_", "", True, True, True)  # wrong-arg-count\n    ')

    def test_nested_namedtuple(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import NamedTuple\n      class Bar:\n        class Foo(NamedTuple):\n          x: int\n        foo = Foo(x=0)\n    ')

    def test_namedtuple_defaults(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import collections\n      X = collections.namedtuple('X', ['a', 'b'], defaults=[0])\n      X('a')\n      X('a', 'b')\n    ")

    def test_variable_annotations(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import collections\n      class X(collections.namedtuple('X', ['a', 'b'])):\n        a: int\n        b: str\n    ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: int\n        b: str\n    ')
if __name__ == '__main__':
    test_base.main()