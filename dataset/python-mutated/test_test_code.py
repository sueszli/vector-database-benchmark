"""Tests for type checking test code."""
from pytype.tests import test_base

class AssertionTest(test_base.BaseTest):
    """Tests for test assertions."""

    def test_assert_not_none(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import unittest\n      from typing import Optional\n      def foo():\n        return '10' if __random__ else None\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = foo()\n          assert_type(x, Optional[str])\n          self.assertIsNotNone(x)\n          assert_type(x, str)\n    ")

    def test_assert_not_none_with_message(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import unittest\n      from typing import Optional\n      def foo():\n        return \'10\' if __random__ else None\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = foo()\n          assert_type(x, Optional[str])\n          self.assertIsNotNone(x, "assertion message")\n          assert_type(x, str)\n    ')

    def test_assert_isinstance(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import unittest\n      from typing import Union\n      def foo():\n        return '10' if __random__ else 10\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = foo()\n          assert_type(x, Union[int, str])\n          self.assertIsInstance(x, str)\n          assert_type(x, str)\n    ")

    def test_assert_isinstance_with_message(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import unittest\n      from typing import Union\n      def foo():\n        return \'10\' if __random__ else 10\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = foo()\n          assert_type(x, Union[int, str])\n          self.assertIsInstance(x, str, "assertion message")\n          assert_type(x, str)\n    ')

    def test_narrowed_type_from_assert_isinstance(self):
        if False:
            print('Hello World!')
        self.Check('\n      import unittest\n      from typing import Union\n      class A:\n        pass\n      class B(A):\n        pass\n      class FooTest(unittest.TestCase):\n        def test_foo(self, x: Union[A, B, int]):\n          self.assertIsInstance(x, A)\n          assert_type(x, Union[A, B])\n    ')

    def test_new_type_from_assert_isinstance(self):
        if False:
            print('Hello World!')
        self.Check('\n      import unittest\n      class A:\n        pass\n      class B(A):\n        pass\n      def foo() -> A:\n        return B()\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = foo()\n          assert_type(x, A)\n          self.assertIsInstance(x, B)\n          assert_type(x, B)\n    ')

    def test_assert_isinstance_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import unittest\n      from typing import Union\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          x = None\n          self.assertIsInstance(x, (int, str))\n          assert_type(x, Union[int, str])\n          self.assertIsInstance(x, (int,))\n          assert_type(x, int)\n    ')

    def test_instance_attribute(self):
        if False:
            return 10
        self.Check('\n      import unittest\n      class Foo:\n        def __init__(self, x):\n          self.x = x\n      class FooTest(unittest.TestCase):\n        def test_foo(self):\n          foo = __any_object__\n          self.assertIsInstance(foo, Foo)\n          print(foo.x)\n    ')

class MockTest(test_base.BaseTest):
    """Tests for unittest.mock."""

    def test_patch(self):
        if False:
            return 10
        self.Check("\n      import unittest\n      from unittest import mock\n      foo = __any_object__\n      bar = __any_object__\n      class Foo(unittest.TestCase):\n        def setUp(self):\n          super().setUp()\n          self.some_mock = mock.patch.object(foo, 'foo').start()\n          self.some_mock.return_value = True\n        def test_bar(self):\n          other_mock = mock.patch.object(bar, 'bar').start()\n          other_mock.return_value.__enter__ = lambda x: x\n    ")
if __name__ == '__main__':
    test_base.main()