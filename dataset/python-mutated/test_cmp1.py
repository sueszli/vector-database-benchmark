"""Test comparison operators."""
from pytype.tests import test_base

class InTest(test_base.BaseTest):
    """Test for "x in y". Also test overloading of this operator."""

    def test_concrete(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      def f(x, y):\n        return x in y  # unsupported-operands[e]\n      f(1, [1])\n      f(1, [2])\n      f("x", "x")\n      f("y", "x")\n      f("y", (1,))\n      f("y", object())\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')
        self.assertErrorRegexes(errors, {'e': "'in'.*object"})

    def test_deep(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x, y):\n        return x in y\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')

    def test_overloaded(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __contains__(self, x):\n          return 3j\n      def f():\n        return Foo() in []\n      def g():\n        # The result of __contains__ is coerced to a bool.\n        return 3 in Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __contains__(self, x) -> complex: ...\n      def f() -> bool: ...\n      def g() -> bool: ...\n    ')

    def test_none(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      x = None\n      if "" in x:  # unsupported-operands[e1]\n        del x[""]  # unsupported-operands[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': "'in'.*None", 'e2': 'item deletion.*None'})

class NotInTest(test_base.BaseTest):
    """Test for "x not in y". Also test overloading of this operator."""

    def test_concrete(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors('\n      def f(x, y):\n        return x not in y  # unsupported-operands[e]\n      f(1, [1])\n      f(1, [2])\n      f("x", "x")\n      f("y", "x")\n      f("y", (1,))\n      f("y", object())\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')
        self.assertErrorRegexes(errors, {'e': "'in'.*object"})

    def test_overloaded(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __contains__(self, x):\n          return 3j\n      def f():\n        return Foo() not in []\n      def g():\n        # The result of __contains__ is coerced to a bool.\n        return 3 not in Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __contains__(self, x) -> complex: ...\n      def f() -> bool: ...\n      def g() -> bool: ...\n    ')

    def test_none(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      x = None\n      if "" not in x:  # unsupported-operands[e1]\n        x[""] = 42  # unsupported-operands[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': "'in'.*None", 'e2': 'item assignment.*None'})

class IsTest(test_base.BaseTest):
    """Test for "x is y". This operator can't be overloaded."""

    def test_concrete(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y):\n        return x is y\n      f(1, 2)\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')

    def test_deep(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        return x is y\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')

class IsNotTest(test_base.BaseTest):
    """Test for "x is not y". This operator can't be overloaded."""

    def test_concrete(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        return x is not y\n      f(1, 2)\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')

    def test_deep(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y):\n        return x is y\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x, y) -> bool: ...')

    def test_class_new(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        def __new__(cls, *args, **kwargs):\n          assert(cls is not Foo)\n          return object.__new__(cls)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Type, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      class Foo:\n        def __new__(cls: Type[_TFoo], *args, **kwargs) -> _TFoo: ...\n    ")

    def test_class_factory(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        @classmethod\n        def factory(cls, *args, **kwargs):\n          assert(cls is not Foo)\n          return object.__new__(cls)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Type, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      class Foo:\n        @classmethod\n        def factory(cls: Type[_TFoo], *args, **kwargs) -> _TFoo: ...\n    ")

class CmpTest(test_base.BaseTest):
    """Test for comparisons. Also test overloading."""
    OPS = ['<', '<=', '>', '>=']

    def _test_concrete(self, op):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors(f'\n      def f(x, y):\n        return x {op} y  # unsupported-operands[e]\n      f(1, 2)\n      f(1, "a")  # <- error raised from here but in line 2\n      f(object(), "x")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: int, y: int) -> bool: ...\n      @overload\n      def f(x: object, y: str) -> bool: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'Types.*int.*str'})
        self.assertErrorRegexes(errors, {'e': 'Called from.*line 4'})

    def test_concrete(self):
        if False:
            print('Hello World!')
        for op in self.OPS:
            self._test_concrete(op)

    def test_literal(self):
        if False:
            while True:
                i = 10
        for op in self.OPS:
            errors = self.CheckWithErrors(f"\n        '1' {op} 2 # unsupported-operands[e]\n      ")
            self.assertErrorRegexes(errors, {'e': 'Types.*str.*int'})

    def test_overloaded(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __lt__(self, x):\n          return 3j\n      def f():\n        return Foo() < 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __lt__(self, x) -> complex: ...\n      def f() -> complex: ...\n    ')

class EqTest(test_base.BaseTest):
    """Test for "x == y". Also test overloading."""

    def test_concrete(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        return x == y\n      f(1, 2)\n      f(1, "a")\n      f(object(), "x")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: int, y: int) -> bool: ...\n      @overload\n      def f(x: int, y: str) -> bool: ...\n      @overload\n      def f(x: object, y: str) -> bool: ...\n    ')

    def test_overloaded(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __eq__(self, x):\n          return 3j\n      def f():\n        return Foo() == 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __eq__(self, x) -> complex: ...\n      def f() -> complex: ...\n    ')

    def test_class(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        return x.__class__ == y.__class__\n      f(1, 2)\n      f(1, "a")\n      f(object(), "x")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: int, y: int) -> bool: ...\n      @overload\n      def f(x: int, y: str) -> bool: ...\n      @overload\n      def f(x: object, y: str) -> bool: ...\n    ')

    def test_primitive_against_unknown(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      v = None  # type: int\n      v == __any_object__\n    ')

class NeTest(test_base.BaseTest):
    """Test for "x != y". Also test overloading."""

    def test_concrete(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y):\n        return x != y\n      f(1, 2)\n      f(1, "a")\n      f(object(), "x")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: int, y: int) -> bool: ...\n      @overload\n      def f(x: int, y: str) -> bool: ...\n      @overload\n      def f(x: object, y: str) -> bool: ...\n    ')

    def test_overloaded(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __ne__(self, x):\n          return 3j\n      def f():\n        return Foo() != 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __ne__(self, x) -> complex: ...\n      def f() -> complex: ...\n    ')

class InstanceUnequalityTest(test_base.BaseTest):

    def test_iterator_contains(self):
        if False:
            while True:
                i = 10
        self.Check('\n      1 in iter((1, 2))\n      1 not in iter((1, 2))\n    ')
if __name__ == '__main__':
    test_base.main()