"""Tests of special builtins (special_builtins.py)."""
from pytype.tests import test_base

class SpecialBuiltinsTest(test_base.BaseTest):
    """Tests for special_builtins.py."""

    def test_property_with_type_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Union\n      class Foo:\n        @property\n        def foo(self) -> Union[str, int]:\n          return __any_object__\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated, Union\n      class Foo:\n        foo = ...  # type: Annotated[Union[int, str], 'property']\n    ")

    def test_property_with_contained_type_parameter(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import List, Union\n      class Foo:\n        @property\n        def foo(self) -> List[Union[str, int]]:\n          return __any_object__\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated, List, Union\n      class Foo:\n        foo = ...  # type: Annotated[List[Union[int, str]], 'property']\n    ")

    def test_callable_matching(self):
        if False:
            return 10
        self.Check('\n      from typing import Any, Callable\n      def f(x: Callable[[Any], bool]):\n        pass\n      f(callable)\n    ')

    def test_filter_starargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f(*args, **kwargs):\n        filter(*args, **kwargs)\n    ')
if __name__ == '__main__':
    test_base.main()