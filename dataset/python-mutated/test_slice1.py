"""Tests for slices."""
from pytype.tests import test_base

class SliceTest(test_base.BaseTest):
    """Tests for the SLICE_<n> opcodes, as well as for __getitem__(slice)."""

    def test_getslice(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      x = [1,2,3]\n      a = x[:]\n      b = x[1:]\n      c = x[1:2]\n      d = x[1:2:3]\n      e = x[:2:3]\n      f = x[1::3]\n      g = x[1:2:]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      x = ...  # type: List[int]\n      a = ...  # type: List[int]\n      b = ...  # type: List[int]\n      c = ...  # type: List[int]\n      d = ...  # type: List[int]\n      e = ...  # type: List[int]\n      f = ...  # type: List[int]\n      g = ...  # type: List[int]\n    ')

    def test_slice_getitem(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __getitem__(self, s):\n          return s\n      Foo()[:]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      class Foo:\n        def __getitem__(self, s: slice) -> slice: ...\n    ')
if __name__ == '__main__':
    test_base.main()