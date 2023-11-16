"""Tests of builtins.tuple."""
from pytype.tests import test_base

class TupleTest(test_base.BaseTest):
    """Tests for builtins.tuple."""

    def test_getitem_int(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      t = ("", 42)\n      v1 = t[0]\n      v2 = t[1]\n      v3 = t[2]\n      v4 = t[-1]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple, Union\n      t = ...   # type: Tuple[str, int]\n      v1 = ...  # type: str\n      v2 = ...  # type: int\n      v3 = ...  # type: Union[str, int]\n      v4 = ...  # type: int\n    ')

    @test_base.skip('Needs better slice support in abstract.Tuple, convert.py.')
    def test_getitem_slice(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      t = ("", 42)\n      v1 = t[:]\n      v2 = t[:1]\n      v3 = t[1:]\n      v4 = t[0:1]\n      v5 = t[0:2:2]\n      v6 = t[:][0]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      t = ...  # type: Tuple[str, int]\n      v1 = ...  # type: Tuple[str, int]\n      v2 = ...  # type: Tuple[str]\n      v3 = ...  # type: Tuple[int]\n      v4 = ...  # type: Tuple[str]\n      v5 = ...  # type: Tuple[str]\n      v6 = ...  # type: str\n    ')

    def test_unpack_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      v1, v2 = ("", 42)\n      _, w = ("", 42)\n      x, (y, z) = ("", (3.14, True))\n    ')
        self.assertTypesMatchPytd(ty, '\n      v1 = ...  # type: str\n      v2 = ...  # type: int\n      _ = ...  # type: str\n      w = ...  # type: int\n      x = ...  # type: str\n      y = ...  # type: float\n      z = ...  # type: bool\n    ')

    def test_bad_unpacking(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      tup = (1, "")\n      a, = tup  # bad-unpacking[e1]\n      b, c, d = tup  # bad-unpacking[e2]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple, Union\n      tup = ...  # type: Tuple[int, str]\n      a = ...  # type: Union[int, str]\n      b = ...  # type: Union[int, str]\n      c = ...  # type: Union[int, str]\n      d = ...  # type: Union[int, str]\n    ')
        self.assertErrorRegexes(errors, {'e1': '2 values.*1 variable', 'e2': '2 values.*3 variables'})

    def test_mutable_item(self):
        if False:
            return 10
        ty = self.Infer('\n      v = {}\n      w = v.setdefault("", ([], []))\n      w[1].append(42)\n      u = w[2]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      v = ...  # type: dict[str, tuple[list[nothing], list[int]]]\n      w = ...  # type: tuple[list[nothing], list[int]]\n      u = ...  # type: list[int]\n    ')

    def test_bad_tuple_class_getitem(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      v = type((3, ""))\n      w = v[0]  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'expected 0 parameters, got 1'})

    def test_tuple_isinstance(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      x = ()\n      if isinstance(x, tuple):\n        y = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      x = ...  # type: Tuple[()]\n      y = ...  # type: int\n    ')

    def test_add_twice(self):
        if False:
            while True:
                i = 10
        self.Check('() + () + ()')

    def test_inplace_add(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      a = ()\n      a += (42,)\n      b = ()\n      b += (42,)\n      b += ("foo",)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple, Union\n      a = ...  # type: Tuple[int]\n      b = ...  # type: Tuple[int, str]\n    ')

    def test_add(self):
        if False:
            return 10
        self.Check("\n      from typing import Tuple\n      a = (1, 2)\n      b = ('3', '4')\n      c = a + b\n      assert_type(c, Tuple[int, int, str, str])\n    ")

    def test_tuple_of_tuple(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Infer, '\n      def f(x=()):\n        x = (x,)\n        enumerate(x)\n        lambda: x\n        return x\n    ')

    def test_tuple_container_matching(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Dict, Tuple\n\n      class Foo:\n        pass\n\n      class _SupplyPoolAsset:\n        def __init__(self):\n          self._resources_available = {}\n          self._resources_used = {}  # type: Dict[str, Tuple[Foo, Foo]]\n          self._PopulateResources()\n\n        def _PopulateResources(self):\n          for x, y, z in __any_object__:\n            self._resources_available[x] = (y, z)\n          for x, y, z in __any_object__:\n            self._resources_available[x] = (y, z)\n\n        def RequestResource(self, resource):\n          self._resources_used[\n              resource.Name()] = self._resources_available[resource.Name()]\n    ')

    def test_bad_extra_parameterization(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import Tuple\n      X = Tuple[int][str]  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'expected 0 parameters, got 1'})

    def test_legal_extra_parameterization(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[T][T][str]\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[str]\n    ")
if __name__ == '__main__':
    test_base.main()