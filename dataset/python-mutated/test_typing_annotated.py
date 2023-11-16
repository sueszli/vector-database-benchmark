"""Tests for PEP-593 typing.Annotated types."""
from pytype.tests import test_base
from pytype.tests import test_utils

class AnnotatedTest(test_base.BaseTest):
    """Tests for typing.Annotated types."""

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing_extensions import Annotated\n      i = ... # type: Annotated[int, "foo"]\n      s: Annotated[str, "foo", "bar"] = "baz"\n    ')
        self.assertTypesMatchPytd(ty, '\n      i: int\n      s: str\n    ')

    def test_nested(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import List\n      from typing_extensions import Annotated\n      i = ... # type: Annotated[Annotated[int, "foo"], "bar"]\n      strings = ... # type: Annotated[List[str], "bar"]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      i: int\n      strings: List[str]\n    ')

    def test_func(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing_extensions import Annotated\n      def id(x:  Annotated[int, "foo"]):\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def id(x: int) -> int: ...\n    ')

    def test_invalid_type(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing_extensions import Annotated\n      x: Annotated[0, int] = 0  # invalid-annotation[err]\n    ')
        self.assertErrorRegexes(errors, {'err': 'Not a type'})

    def test_missing_type(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing_extensions import Annotated\n      x: Annotated = 0  # invalid-annotation[err]\n    ')
        self.assertErrorRegexes(errors, {'err': 'Not a type'})

    def test_missing_annotation(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing_extensions import Annotated\n      x: Annotated[int] # invalid-annotation[err]\n    ')
        self.assertErrorRegexes(errors, {'err': 'must have at least 1 annotation'})

    def test_annotated_in_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Annotated\n        class A:\n          x: Annotated[int, 'tag'] = ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.A().x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x: int\n      ')

    def test_annotated_type_in_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Annotated\n        class Foo:\n          w: int\n        class A:\n          x: Annotated[Foo, 'tag'] = ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.A().x.w\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x: int\n      ')

    def test_subclass_annotated_in_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Annotated\n        class A:\n          x: Annotated[int, 'tag1', 'tag2'] = ...\n      ")
            ty = self.Infer('\n        import a\n        class B(a.A):\n          pass\n        x = B().x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        class B(a.A): ...\n        x: int\n      ')
if __name__ == '__main__':
    test_base.main()