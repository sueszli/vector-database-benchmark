"""Tests for builtins.list."""
from pytype.tests import test_base

class ListTestBasic(test_base.BaseTest):
    """Basic tests for builtins.list in Python 3."""

    def test_repeated_add(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing import List, Text, Tuple\n      def f() -> Tuple[List[Text]]:\n        x = (\n            ['' % __any_object__, ''] + [''] + [''] + [''.format()] + [''] +\n            [['' % __any_object__, '', '']]\n        )\n        return ([__any_object__] + [''] + x,)  # bad-return-type\n    ")

    def test_bad_comprehension(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      x = None\n      l = [y for y in x]  # attribute-error\n    ')

class ListTest(test_base.BaseTest):
    """Tests for builtins.list in Python 3."""

    def test_byte_unpack_ex(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import List\n      a, *b, c, d = 1, 2, 3, 4, 5, 6, 7\n      i, *j = 1, 2, 3, "4"\n      *k, l = 4, 5, 6\n      m, *n, o = [4, 5, "6", None, 7, 8]\n      p, *q, r = 4, 5, "6", None, 7, 8\n      vars = None # type : List[int]\n      s, *t, u = vars\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Optional, Union\n      a = ... # type: int\n      b = ... # type: List[int]\n      c = ... # type: int\n      d = ... # type: int\n      i = ... # type: int\n      j = ... # type: List[Union[int, str]]\n      k = ... # type: List[int]\n      l = ... # type: int\n      m = ... # type: int\n      n = ... # type: List[Optional[Union[int, str]]]\n      o = ... # type: int\n      p = ... # type: int\n      q = ... # type: List[Optional[Union[int, str]]]\n      r = ... # type: int\n      s = ...  # type: int\n      t = ...  # type: List[int]\n      u = ...  # type: int\n      vars = ...  # type: List[int]\n    ')

    def test_getitem_slot(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, _) = self.InferWithErrors('\n      a = [1, \'2\', 3, 4]\n      p = a[1]\n      q = 1 if __random__ else 2\n      r = a[q]\n      s = a["s"]  # unsupported-operands\n      t = a[-1]\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Union\n      a = ...  # type: List[Union[int, str]]\n      p = ...  # type: str\n      q = ...  # type: int\n      r = ...  # type: Union[int, str]\n      s = ...  # type: Any\n      t = ...  # type: int\n      ')

    def test_slice_returntype(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, _) = self.InferWithErrors('\n      from typing import Sequence, MutableSequence\n      a: Sequence[int] = [1]\n      b = a[0:1]\n      c: MutableSequence[int] = [1]\n      d = c[0:1]\n      e = [2]\n      f = e[0:1]\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, MutableSequence, Sequence\n      a = ...  # type: Sequence[int]\n      b = ...  # type: Sequence[int]\n      c = ...  # type: MutableSequence[int]\n      d = ...  # type: MutableSequence[int]\n      e = ...  # type: List[int]\n      f = ...  # type: List[int]\n      ')

    @test_base.skip('Requires more precise slice objects')
    def test_getitem_slice(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors('\n      a = [1, \'2\', 3, 4]\n      b = a[:]\n      c = 1 if __random__ else 2\n      d = a[c:2]\n      e = a[c:]\n      f = a[2:]\n      g = a[2:None]\n      h = a[None:2]\n      i = a[None:None]\n      j = a[int:str]  # wrong-arg-types\n      k = a["s":]  # wrong-arg-types\n      m = a[1:-1]\n      n = a[0:0]\n      o = a[1:1]\n      p = a[1:2]\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Union\n      a = ...  # type: List[Union[int, str]]\n      b = ...  # type: List[Union[int, str]]\n      c = ...  # type: int\n      d = ...  # type: List[str]\n      e = ...  # type: List[Union[int, str]]\n      f = ...  # type: List[int]\n      g = ...  # type: List[int]\n      h = ...  # type: List[Union[int, str]]\n      i = ...  # type: List[Union[int, str]]\n      j = ...  # type: Any\n      k = ...  # type: Any\n      m = ...  # type: List[Union[int, str]]\n      n = ...  # type: List[nothing]\n      o = ...  # type: List[nothing]\n      p = ...  # type: List[str]\n      ')

    def test_appends(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n    from typing import List\n    def f():\n      lst1: List[List[str]] = []\n      lst2: List[List[str]] = []\n      if __random__:\n        x, lst1 = __any_object__\n      else:\n        x = lst2[-1]\n      lst1.append(x)\n      lst2.append(lst1[-1])\n    ')

    def test_clear(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      a = [0]\n      a.clear()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      a = ...  # type: List[int]\n    ')
if __name__ == '__main__':
    test_base.main()