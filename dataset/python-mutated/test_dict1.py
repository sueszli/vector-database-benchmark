"""Tests for dictionaries."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class DictTest(test_base.BaseTest):
    """Tests for dictionaries."""

    def test_pop(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      d = {"a": 42}\n      v1 = d.pop("a")\n      v2 = d.pop("b", None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      d = ...  # type: Dict[str, int]\n      v1 = ...  # type: int\n      v2 = ...  # type: None\n    ')

    def test_bad_pop(self):
        if False:
            return 10
        ty = self.Infer('\n      d = {"a": 42}\n      v = d.pop("b")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      d = ...  # type: Dict[str, int]\n      v = ...  # type: Any\n    ')

    def test_ambiguous_pop(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      d = {"a": 42}\n      k = None  # type: str\n      v1 = d.pop(k)\n      v2 = d.pop(k, None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Optional\n      d = ...  # type: Dict[str, int]\n      k = ...  # type: str\n      v1 = ...  # type: int\n      v2 = ...  # type: Optional[int]\n    ')

    def test_pop_from_ambiguous_dict(self):
        if False:
            return 10
        ty = self.Infer('\n      d = {}\n      k = None  # type: str\n      v = None  # type: int\n      d[k] = v\n      v1 = d.pop("a")\n      v2 = d.pop("a", None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Optional\n      d = ...  # type: Dict[str, int]\n      k = ...  # type: str\n      v = ...  # type: int\n      v1 = ...  # type: int\n      v2 = ...  # type: Optional[int]\n    ')

    def test_update_empty(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Dict\n      d1 = {}\n      d2 = None  # type: Dict[str, int]\n      d1.update(d2)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      d1 = ...  # type: Dict[str, int]\n      d2 = ...  # type: Dict[str, int]\n    ')

    def test_update_any_subclass(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: T = ...) -> T: ...\n      ')
            self.Check('\n        from typing import Any\n        import foo\n        class Foo(Any):\n          def f(self):\n            kwargs = {}\n            kwargs.update(foo.f(self))\n      ', pythonpath=[d.path])

    def test_update_noargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Dict\n      d = {}  # type: Dict\n      d.update()\n    ')

    def test_determinism(self):
        if False:
            print('Hello World!')
        canonical = None
        for _ in range(10):
            ty = self.Infer('\n        class Foo:\n          def __init__(self, filenames):\n            self._dict = {}\n            for filename in filenames:\n              d = self._dict\n              if __random__:\n                d[__any_object__] = {}\n                d = d[__any_object__]\n              if __random__:\n                d[__any_object__] = None\n      ')
            out = pytd_utils.Print(ty)
            if canonical is None:
                canonical = out
            else:
                self.assertMultiLineEqual(canonical, out)

    def test_unpack_ordered_dict_value(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n      def f():\n        d = collections.OrderedDict()\n        for k, (v1, v2) in d.items():\n          pass\n      f()\n    ')
if __name__ == '__main__':
    test_base.main()