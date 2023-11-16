"""Tests for the methods in the typing module."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TypingMethodsTest(test_base.BaseTest):
    """Tests for typing.py."""

    def test_mapping(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Mapping\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class MyDict(Mapping[K, V]): ...\n        def f() -> MyDict[str, int]: ...\n      ')
            ty = self.Infer('\n        import foo\n        m = foo.f()\n        a = m.copy()\n        b = "foo" in m\n        c = m["foo"]\n        d = m.get("foo", 3)\n        e = [x for x in m.items()]\n        f = [x for x in m.keys()]\n        g = [x for x in m.values()]\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List, Tuple, Union\n        import foo\n        m = ...  # type: foo.MyDict[str, int]\n        a = ...  # type: typing.Mapping[str, int]\n        b = ...  # type: bool\n        c = ...  # type: int\n        d = ...  # type: int\n        e = ...  # type: List[Tuple[str, int]]\n        f = ...  # type: List[str]\n        g = ...  # type: List[int]\n      ')

    def test_supportsbytes(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import SupportsBytes\n        def f() -> SupportsBytes: ...\n      ')
            self.Check('\n        import foo\n        x = foo.f()\n        bytes(x)\n      ', pythonpath=[d.path])

    def test_assert_never(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union\n      from typing_extensions import assert_never\n      def int_or_str(arg: Union[int, str]) -> None:\n        if isinstance(arg, int):\n          pass\n        elif isinstance(arg, str):\n          pass\n        else:\n          assert_never("oops!")\n    ')

    def test_assert_never_failure(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Union\n      from typing_extensions import assert_never\n      def int_or_str(arg: Union[int, str]) -> None:\n        if isinstance(arg, int):\n          pass\n        else:\n          assert_never("oops!")  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Expected', 'empty', 'Actual', 'str']})
if __name__ == '__main__':
    test_base.main()