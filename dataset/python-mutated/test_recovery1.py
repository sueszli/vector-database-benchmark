"""Tests for recovering after errors."""
from pytype.tests import test_base

class RecoveryTests(test_base.BaseTest):
    """Tests for recovering after errors.

  The type inferencer can warn about bad code, but it should never blow up.
  These tests check that we don't faceplant when we encounter difficult code.
  """

    def test_bad_subtract(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        t = 0.0\n        return t - ("bla" - t)\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f() -> Any: ...\n    ')

    def test_inherit_from_instance(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo(3):\n        pass\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo(Any):\n        pass\n    ')

    def test_name_error(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      x = foobar\n      class A(x):\n        pass\n      pow(A(), 2)\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      x = ...  # type: Any\n      class A(Any):\n        pass\n    ')

    def test_object_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNoCrash(self.Check, '\n      object.bla(int)\n    ')

    def test_attr_error(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class A:\n        pass\n      x = A.x\n      class B:\n        pass\n      y = "foo".foo()\n      object.bar(int)\n      class C:\n        pass\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        pass\n      x = ...  # type: Any\n      class B:\n        pass\n      y = ...  # type: Any\n      class C:\n        pass\n    ')

    def test_wrong_call(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        pass\n      f("foo")\n      x = 3\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      def f() -> None: ...\n      x = ...  # type: int\n    ')

    def test_duplicate_identifier(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.foo = 3\n        def foo(self):\n          pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        foo = ...  # type: Any\n        def __init__(self) -> None: ...\n    ')

    def test_method_with_unknown_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      from nowhere import decorator  # import-error\n      class Foo:\n        @decorator\n        def f():\n          name_error  # name-error\n    ', deep=True)

    def test_assert_in_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo:\n        def __init__(self):\n          self._bar = "foo"\n          assert False\n        def __str__(self):\n          return self._bar\n    ')

    @test_base.skip("Line 7, in __str__: No attribute '_bar' on Foo'")
    def test_constructor_infinite_loop(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __init__(self):\n          self._bar = "foo"\n          while True: pass\n        def __str__(self):\n          return self._bar\n    ')

    def test_attribute_access_in_impossible_path(self):
        if False:
            print('Hello World!')
        self.InferWithErrors('\n      x = 3.14 if __random__ else 42\n      if isinstance(x, int):\n        if isinstance(x, float):\n          x.upper  # not reported\n          3 in x  # unsupported-operands\n    ')

    def test_binary_operator_on_impossible_path(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      x = "" if __random__ else []\n      if isinstance(x, list):\n        if isinstance(x, str):\n          x / x  # unsupported-operands\n    ')
if __name__ == '__main__':
    test_base.main()