"""Tests for type comments."""
from pytype.tests import test_base

class FunctionCommentWithAnnotationsTest(test_base.BaseTest):
    """Tests for type comments that require annotations."""

    def test_function_type_comment_plus_annotations(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      def foo(x: int) -> float:\n        # type: (int) -> float  # redundant-function-type-comment\n        return x\n    ')

    def test_list_comprehension_comments(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import List\n      def f(x: str):\n        pass\n      def g(xs: List[str]) -> List[str]:\n        ys = [f(x) for x in xs]  # type: List[str]  # annotation-type-mismatch[e]\n        return ys\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      def f(x: str) -> None: ...\n      def g(xs: List[str]) -> List[str]: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: List\\[str\\].*Assignment: List\\[None\\]'})

class Py3TypeCommentTest(test_base.BaseTest):
    """Type comment tests that use Python 3-only features."""

    def test_ignored_comment(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      def f():\n        v: int = None  # type: str  # ignored-type-comment\n    ')

    def test_first_line_of_code(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Dict\n      def f() -> Dict[str, int]:\n        # some_var = ''\n        # something more\n        cast_type: Dict[str, int] = {\n          'one': 1,\n          'two': 2,\n          'three': 3,\n        }\n        return cast_type\n    ")

    def test_multiline_comment(self):
        if False:
            return 10
        self.Check('\n      x = [\n        k for k in range(5)\n      ]  # type: list[int]\n    ')

    def test_multiline_comment_on_function_close_line(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(\n        x=None\n      ): y = [\n          k for k in range(5)\n      ]  # type: list[int]\n    ')

    def test_type_comment_and_type_ignore(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      x = ''  # type: int  # type: ignore\n    ")

    def test_adjust_type_ignore(self):
        if False:
            print('Hello World!')
        self.Check("\n      def f(x: int):\n        pass\n      f(\n          'oops')  # type: ignore\n    ")
if __name__ == '__main__':
    test_base.main()