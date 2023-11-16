"""Tests for builtins.list."""
from pytype.tests import test_base

class ListTest(test_base.BaseTest):
    """Tests for builtins.list."""

    def test_add(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      a = []\n      a = a + [42]\n      b = []\n      b = b + [42]\n      b = b + ["foo"]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      a = ...  # type: List[int]\n      b = ...  # type: List[Union[int, str]]\n    ')

    def test_inplace_add(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      a = []\n      a += [42]\n      b = []\n      b += [42]\n      b += ["foo"]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      a = ...  # type: List[int]\n      b = ...  # type: List[Union[int, str]]\n    ')

    def test_inplace_mutates(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = []\n      b = a\n      a += [42]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      a = ...  # type: List[int]\n      b = ...  # type: List[int]\n    ')

    def test_extend_with_empty(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import List\n      v = []  # type: List[str]\n      for x in []:\n        v.extend(x)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List\n      v = ...  # type: List[str]\n      x = ...  # type: Any\n    ')

    def test_getitem_slot(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      a = [1, \'2\', 3, 4]\n      b = a[1]\n      c = 1 if __random__ else 2\n      d = a[c]\n      e = a["s"]  # unsupported-operands[e]\n      f = a[-1]\n      g = a[slice(1,2)]  # should be List[str]\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Union\n      a = ...  # type: List[Union[int, str]]\n      b = ...  # type: str\n      c = ...  # type: int\n      d = ...  # type: Union[int, str]\n      e = ...  # type: Any\n      f = ...  # type: int\n      g = ...  # type: List[Union[int, str]]\n      ')
        self.assertErrorRegexes(errors, {'e': '__getitem__ on List'})

    def test_index_out_of_range(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      a = [0] if __random__ else []\n      b = 0\n      if b < len(a):\n        c = a[b]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      a = ...  # type: List[int]\n      b = ...  # type: int\n      c = ...  # type: int\n    ')
if __name__ == '__main__':
    test_base.main()