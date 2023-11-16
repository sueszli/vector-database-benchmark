"""Tests for bad-return-type errors."""
from pytype.tests import test_base

class TestReturns(test_base.BaseTest):
    """Tests for bad-return-type."""

    def test_implicit_none(self):
        if False:
            return 10
        self.CheckWithErrors('\n      def f(x) -> int:\n        pass  # bad-return-type\n    ')

    def test_implicit_none_with_decorator(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      def decorator(f):\n        return f\n      @decorator\n      def f(x) -> int:\n        '''docstring'''  # bad-return-type\n    ")

    def test_if(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      def f(x) -> int:\n        if x:\n          pass\n        else:\n          return 10  # bad-return-type\n    ')

    def test_nested_if(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      def f(x) -> int:\n        if x:\n          if __random__:\n            pass\n          else:\n            return 'a'  # bad-return-type\n        else:\n          return 10\n        pass  # bad-return-type\n    ")

    def test_with(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      def f(x) -> int:\n        with open('foo'):\n          if __random__:\n            pass\n          else:\n            return 'a'  # bad-return-type  # bad-return-type\n    ")

    def test_nested_with(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      def f(x) -> int:\n        with open('foo'):\n          if __random__:\n            with open('bar'):\n              if __random__:\n                pass\n              else:\n                return 'a'  # bad-return-type  # bad-return-type\n    ")

    def test_no_return_any(self):
        if False:
            while True:
                i = 10
        self.options.set_feature_flags({'no-return-any'})
        self.CheckWithErrors('\n      from typing import Any\n\n      def f(x: Any):\n        return x  # bad-return-type\n    ')
if __name__ == '__main__':
    test_base.main()