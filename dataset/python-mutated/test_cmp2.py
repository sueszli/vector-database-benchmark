"""Test comparison operators."""
from pytype.tests import test_base

class InstanceUnequalityTest(test_base.BaseTest):

    def test_is(self):
        if False:
            while True:
                i = 10
        'SomeType is not be the same as AnotherType.'
        self.Check('\n      from typing import Optional\n      def f(x: Optional[str]) -> NoneType:\n        if x is None:\n          return x\n        else:\n          return None\n      ')

class ContainsFallbackTest(test_base.BaseTest):
    """Tests the __contains__ -> __iter__ -> __getitem__ fallbacks."""

    def test_overload_contains(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class F:\n        def __contains__(self, x: int):\n          if not isinstance(x, int):\n            raise TypeError("__contains__ only takes int")\n          return True\n      1 in F()\n      "not int" in F()  # unsupported-operands\n    ')

    def test_fallback_iter(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class F:\n        def __iter__(self):\n          pass\n      1 in F()\n      "not int" in F()\n    ')

    def test_fallback_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class F:\n        def __getitem__(self, key):\n          pass\n      1 in F()\n      "not int" in F()\n    ')

class NotImplementedTest(test_base.BaseTest):
    """Tests handling of the NotImplemented builtin."""

    def test_return_annotation(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __eq__(self, other) -> bool:\n          if isinstance(other, Foo):\n            return id(self) == id(other)\n          else:\n            return NotImplemented\n    ')

    def test_infer_return_type(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        def __eq__(self, other):\n          if isinstance(other, Foo):\n            return id(self) == id(other)\n          else:\n            return NotImplemented\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __eq__(self, other) -> bool: ...\n    ')

class CmpErrorTest(test_base.BaseTest):
    """Tests comparisons with type errors."""

    def test_compare_types(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors("\n      res = (1).__class__ < ''.__class__  # unsupported-operands\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      res: Any\n    ')

    def test_failed_override(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import datetime\n      a = datetime.timedelta(0)\n      b = bool(a > 0)  # unsupported-operands\n    ')

    def test_compare_primitives(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      100 < 'a'  # unsupported-operands\n      'a' <= 1.0  # unsupported-operands\n      10 < 10.0\n      10.0 >= 10\n      def f(x: int, y: str) -> bool:\n        return x < y  # unsupported-operands\n    ")

class MetaclassTest(test_base.BaseTest):
    """Tests comparisons on class objects with a custom metaclass."""

    def test_compare_types(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class Meta(type):\n        def __gt__(self, other):\n          return True\n          # return self.__name__ > other.__name__\n\n      class A(metaclass=Meta): pass\n      class B(metaclass=Meta): pass\n\n      print(A > B)  # missing-parameter\n    ')
if __name__ == '__main__':
    test_base.main()