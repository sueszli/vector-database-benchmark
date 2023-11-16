"""Tests for slices."""
from pytype.tests import test_base

class SliceTest(test_base.BaseTest):
    """Tests for the SLICE_<n> opcodes, as well as for __getitem__(slice)."""

    def test_custom_getslice(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        def __getitem__(self, index):\n          return index\n      x = Foo()\n      a = x[:]\n      b = x[1:]\n      c = x[1:2]\n      d = x[1:2:3]\n      e = x[:2:3]\n      f = x[1::3]\n      g = x[1:2:]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __getitem__(self, index: slice) -> slice: ...\n      x = ...  # type: Foo\n      a = ...  # type: slice\n      b = ...  # type: slice\n      c = ...  # type: slice\n      d = ...  # type: slice\n      e = ...  # type: slice\n      f = ...  # type: slice\n      g = ...  # type: slice\n    ')
if __name__ == '__main__':
    test_base.main()