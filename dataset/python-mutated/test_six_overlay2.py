"""Tests for methods in six_overlay.py."""
from pytype.tests import test_base

class SixTests(test_base.BaseTest):
    """Tests for six and six_overlay."""

    def test_version_check(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import six\n      if six.PY2:\n        v = 42\n      elif six.PY3:\n        v = "hello world"\n      else:\n        v = None\n    ')
        self.assertTypesMatchPytd(ty, '\n      import six\n      v = ...  # type: str\n    ')

    def test_string_types(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import List, Union\n      import six\n      a = ''  # type: Union[str, List[str]]\n      if isinstance(a, six.string_types):\n        a = [a]\n      b = ''  # type: str\n      if isinstance(b, six.string_types):\n        c = len(b)\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      import six\n      a: List[str]\n      b: str\n      c: int\n    ')

    def test_integer_types(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import six\n      from typing import List, Union\n      def foo(x: Union[List[int], int]) -> List[int]:\n        if isinstance(x, six.integer_types):\n          return [x]\n        else:\n          return x\n    ')
if __name__ == '__main__':
    test_base.main()