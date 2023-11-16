"""Tests for dictionaries."""
from pytype.tests import test_base
from pytype.tests import test_utils

class DictTest(test_base.BaseTest):
    """Tests for dictionaries."""

    def test_filtered_getitem(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Union\n      MAP = {0: "foo"}\n      def foo(x: Union[int, None]):\n        if x is not None:\n          return MAP[x]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Optional, Union\n      MAP = ...  # type: Dict[int, str]\n      def foo(x: Union[int, None]) -> Optional[str]: ...\n    ')

    def test_object_in_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import Any, Dict\n      def objectIsStr() -> Dict[str, Any]:\n        return {object(): ""}  # bad-return-type\n    ')

    def test_big_concrete_dict(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Dict, Tuple, Union\n      # A concrete dictionary with lots of concrete keys and a complicated\n      # value type.\n      d = {}\n      ValueType = Dict[Union[str, int], Union[str, int]]\n      v = ...  # type: ValueType\n      d['a'] = v\n      d['b'] = v\n      d['c'] = v\n      d['d'] = v\n      d['e'] = v\n      d[('a', None)] = v\n      d[('b', None)] = v\n      d[('c', None)] = v\n      d[('d', None)] = v\n      d[('e', None)] = v\n      def f() -> Dict[Union[str, Tuple[str, None]], ValueType]:\n        return d\n      def g() -> Dict[int, int]:\n        return d  # bad-return-type\n    ")

    def test_dict_of_tuple(self):
        if False:
            return 10
        self.Check('\n      from typing import Dict, Tuple\n      def iter_equality_constraints(op):\n        yield (op, 0 if __random__ else __any_object__)\n      def get_equality_groups(ops) -> Dict[Tuple, Tuple]:\n        group_dict = {}\n        for op in ops:\n          for a0 in iter_equality_constraints(op):\n            group_dict[a0] = a0\n            group_dict[__any_object__] = a0\n        return group_dict\n    ')

    def test_recursion(self):
        if False:
            return 10
        self.Check("\n      from typing import Any, Dict\n      def convert(d: Dict[Any, Any]):\n        keys = ['foo', 'bar']\n        for key in keys:\n          if key not in d:\n            d[key + '_suffix1'] = {}\n          if key + '_suffix2' in d:\n            d[key + '_suffix1']['suffix2'] = d[key + '_suffix2']\n          if key + '_suffix3' in d:\n            d[key + '_suffix1']['suffix3'] = d[key + '_suffix3']\n    ")

    @test_utils.skipBeforePy((3, 9), 'Dict | was added in 3.9.')
    def test_union(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors("\n      from typing import Dict\n      a = {'a': 1} | {'b': 2}\n      b = {'a': 1}\n      b |= {1: 'a'}\n      c: Dict[str, int] = {'a': 1} | {1: 'a'}  # annotation-type-mismatch\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      a: Dict[str, int]\n      b: Dict[Union[str, int], Union[str, int]]\n      c: Dict[str, int]\n    ')

    def test_reverse_views(self):
        if False:
            print('Hello World!')
        self.Check("\n      x = {'a': 'b'}\n      print(reversed(x.keys()))\n      print(reversed(x.values()))\n      print(reversed(x.items()))\n    ")

    def test_does_not_match_sequence(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Sequence\n      x: Sequence[str] = {1: 'a'}  # annotation-type-mismatch\n      y: Sequence[str] = {'a': 1}  # annotation-type-mismatch\n    ")

    def test_bad_update(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      d = {}\n      d.update(1)  # wrong-arg-types\n    ')

    @test_utils.skipBeforePy((3, 9), 'Requires new unpacking logic in 3.9.')
    def test_bad_unpack(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      lst = [3, 4]\n      def f(**kwargs):\n        pass\n      f(**lst)  # wrong-arg-types\n    ')

    def test_update_multiple_types(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      def f(**kwargs):\n        kwargs.update(a=0, b='1')\n    ")
if __name__ == '__main__':
    test_base.main()