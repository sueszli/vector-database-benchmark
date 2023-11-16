"""Tests for typing.AnyStr."""
from pytype.tests import test_base
from pytype.tests import test_utils

class AnyStrTest(test_base.BaseTest):
    """Tests for issues related to AnyStr."""

    def test_callable(self):
        if False:
            i = 10
            return i + 15
        'Tests Callable + AnyStr.'
        self.Check('\n      from typing import AnyStr, Callable\n\n      def f1(f: Callable[[AnyStr], AnyStr]):\n        f2(f)\n      def f2(f: Callable[[AnyStr], AnyStr]):\n        pass\n      ')

    def test_unknown_against_multiple_anystr(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Any, Dict, Tuple, AnyStr\n\n      def foo(x: Dict[Tuple[AnyStr], AnyStr]): ...\n      foo(__any_object__)\n    ')

    def test_multiple_unknown_against_multiple_anystr(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import AnyStr, List\n      def foo(x: List[AnyStr], y: List[AnyStr]): ...\n      foo(__any_object__, [__any_object__])\n    ')

    def test_anystr_in_closure(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      from typing import AnyStr, Dict, Optional\n      def foo(d: Dict[unicode, Optional[AnyStr]] = None):\n        def bar() -> Optional[AnyStr]:\n          return __any_object__\n        d[__any_object__] = bar()\n    ')

    def test_missing_import(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      def f(x: AnyStr) -> AnyStr:  # name-error\n        return x\n    ')

    def test_generic_inheritance(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import AnyStr, Generic\n      class Foo(Generic[AnyStr]):\n        @property\n        def name(self) -> AnyStr | None: ...\n      def dofoo() -> Foo[str]: ...\n    ')]):
            self.Check("\n        import foo\n        assert_type(foo.dofoo().name, 'Optional[str]')\n      ")

class AnyStrTestPy3(test_base.BaseTest):
    """Tests for issues related to AnyStr in Python 3."""

    def test_anystr(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import AnyStr\n      def f(x: AnyStr) -> AnyStr:\n        return __any_object__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar\n      AnyStr = TypeVar("AnyStr", str, bytes)\n      def f(x: AnyStr) -> AnyStr: ...\n    ')
        self.assertTrue(ty.Lookup('f').signatures[0].template)

    def test_anystr_function_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import AnyStr\n        def f(x: AnyStr) -> AnyStr: ...\n      ')
            ty = self.Infer('\n        from a import f\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import TypesVar\n        AnyStr = TypeVar("AnyStr", str, bytes)\n        def f(x: AnyStr) -> AnyStr: ...\n      ')

    def test_use_anystr_constraints(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      from typing import AnyStr, TypeVar\n      def f(x: AnyStr, y: AnyStr) -> AnyStr:\n        return __any_object__\n      v1 = f(__any_object__, u"")  # ok\n      v2 = f(__any_object__, 42)  # wrong-arg-types[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, TypeVar\n      AnyStr = TypeVar("AnyStr", str, bytes)\n      def f(x: AnyStr, y: AnyStr) -> AnyStr: ...\n      v1 = ...  # type: str\n      v2 = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e': 'Union\\[bytes, str\\].*int'})

    def test_constraint_mismatch(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      from typing import AnyStr\n      def f(x: AnyStr, y: AnyStr): ...\n      f("", "")  # ok\n      f("", b"")  # wrong-arg-types[e]\n      f(b"", b"")  # ok\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*y: str.*Actual.*y: bytes'})

    def test_custom_generic(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import AnyStr, Generic\n      class Foo(Generic[AnyStr]):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      AnyStr = TypeVar('AnyStr', str, bytes)\n      class Foo(Generic[AnyStr]): ...\n    ")
if __name__ == '__main__':
    test_base.main()