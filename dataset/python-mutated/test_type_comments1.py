"""Tests for type comments."""
import sys
from pytype.tests import test_base
from pytype.tests import test_utils

class FunctionCommentTest(test_base.BaseTest):
    """Tests for type comments."""

    def test_function_unspecified_args(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo(x):\n        # type: (...) -> int\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(x) -> int: ...\n    ')

    def test_function_return_space(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import Dict\n      def foo(x):\n        # type: (...) -> Dict[int, int]\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      def foo(x) -> Dict[int, int]: ...\n    ')

    def test_function_zero_args(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def foo():\n        # type: (  ) -> int\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo() -> int: ...\n    ')

    def test_function_one_arg(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo(x):\n        # type: ( int ) -> int\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(x: int) -> int: ...\n    ')

    def test_function_several_args(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def foo(x, y, z):\n        # type: (int, str, float) -> None\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(x: int, y: str, z: float) -> None: ...\n    ')

    def test_function_several_lines(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo(x,\n              y,\n              z):\n        # type: (int, str, float) -> None\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(x: int, y: str, z: float) -> None: ...\n    ')

    def test_function_comment_on_colon(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      def f(x) \\\n        : # type: (None) -> None\n        return True  # bad-return-type\n    ')

    def test_function_comment_on_def_line(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):  # type: (int) -> int\n        return x\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_multiple_function_comments(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        # type: (None) -> bool\n        # type: (str) -> str  # ignored-type-comment[e]\n        return True\n    ')
        self.assertErrorRegexes(errors, {'e': 'Stray type comment:.*str'})

    def test_function_none_in_args(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def foo(x, y, z):\n        # type: (int, str, None) -> None\n        return x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(x: int, y: str, z: None) -> None: ...\n    ')

    def test_self_is_optional(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def f(self, x):\n          # type: (int) -> None\n          pass\n\n        def g(self, x):\n          # type: (Foo, int) -> None\n          pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def f(self, x: int) -> None: ...\n        def g(self, x: int) -> None: ...\n    ')

    def test_cls_is_optional(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        @classmethod\n        def f(cls, x):\n          # type: (int) -> None\n          pass\n\n        @classmethod\n        def g(cls, x):\n          # type: (Foo, int) -> None\n          pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        @classmethod\n        def f(cls, x: int) -> None: ...\n        @classmethod\n        def g(cls: Foo, x: int) -> None: ...\n    ')

    def test_function_stararg(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __init__(self, *args):\n          # type: (int) -> None\n          self.value = args[0]\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        value = ...  # type: int\n        def __init__(self, *args: int) -> None: ...\n    ')

    def test_function_starstararg(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      class Foo:\n        def __init__(self, **kwargs):\n          # type: (int) -> None\n          self.value = kwargs['x']\n    ")
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        value = ...  # type: int\n        def __init__(self, **kwargs: int) -> None: ...\n    ')

    @test_utils.skipFromPy((3, 10), "In 3.10+, we can't associate the function type comment to the function because the function body opcodes have line number 1 instead of 3. Since function type comments are long-deprecated, we don't bother trying to make this work.")
    def test_function_without_body(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def foo(x, y):\n        # type: (int, str) -> None\n        '''Docstring but no body.'''\n    ")
        self.assertTypesMatchPytd(ty, '\n      def foo(x: int, y: str) -> None: ...\n    ')

    def test_filter_out_class_constructor(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        x = 0 # type: int\n    ')

    def test_type_comment_after_docstring(self):
        if False:
            i = 10
            return i + 15
        'Type comments after the docstring should not be picked up.'
        self.InferWithErrors("\n      def foo(x, y):\n        '''Ceci n'est pas une type.'''\n        # type: (int, str) -> None  # ignored-type-comment\n    ")

    def test_function_no_return(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      def foo():\n        # type: () ->  # invalid-function-type-comment\n        pass\n    ')

    def test_function_too_many_args(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      def foo(x):\n        # type: (int, str) -> None  # invalid-function-type-comment[e]\n        y = x\n        return x\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected 1 args, 2 given'})

    def test_function_too_few_args(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def foo(x, y, z):\n        # type: (int, str) -> None  # invalid-function-type-comment[e]\n        y = x\n        return x\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected 3 args, 2 given'})

    def test_function_too_few_args_do_not_count_self(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      def foo(self, x, y, z):\n        # type: (int, str) -> None  # invalid-function-type-comment[e]\n        y = x\n        return x\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected 3 args, 2 given'})

    def test_function_missing_args(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      def foo(x):\n        # type: () -> int  # invalid-function-type-comment\n        return x\n    ')

    def test_invalid_function_type_comment(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      def foo(x):\n        # type: blah blah blah  # invalid-function-type-comment\n        return x\n    ')

    def test_invalid_function_args(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      def foo(x):\n        # type: (abc def) -> int  # invalid-function-type-comment[e]\n        return x\n    ')
        if self.python_version >= (3, 10):
            error_reason = 'invalid syntax'
        else:
            error_reason = 'unexpected EOF'
        self.assertErrorRegexes(errors, {'e': f'abc def.*{error_reason}'})

    def test_ambiguous_annotation(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      def foo(x):\n        # type: (int if __random__ else str) -> None  # invalid-function-type-comment[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*str.*constant'})

    @test_utils.skipFromPy((3, 10), "In 3.10+, we can't associate the function type comment to function g because the opcodes in the body of g have line number 2 instead of 4. Since function type comments are long-deprecated, we don't bother trying to make this work.")
    def test_one_line_function(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      def f(): return 0\n      def g():\n        # type: () -> None\n        '''Docstring.'''\n    ")
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n      def g() -> None: ...\n    ')

    def test_comment_after_type_comment(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        # type: (...) -> type\n        # comment comment comment\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(x) -> type: ...\n    ')

class AssignmentCommentTest(test_base.BaseTest):
    """Tests for type comments applied to assignments."""

    def test_class_attribute_comment(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        s = None  # type: str\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        s = ...  # type: str\n    ')

    def test_instance_attribute_comment(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __init__(self):\n          self.s = None  # type: str\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        s = ...  # type: str\n        def __init__(self) -> None: ...\n    ')

    def test_global_comment(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      X = None  # type: str\n    ')
        self.assertTypesMatchPytd(ty, '\n      X = ...  # type: str\n    ')

    def test_global_comment2(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      X = None  # type: str\n      def f(): global X\n    ')
        self.assertTypesMatchPytd(ty, '\n      X = ...  # type: str\n      def f() -> None: ...\n    ')

    def test_local_comment(self):
        if False:
            return 10
        ty = self.Infer('\n      X = None\n\n      def foo():\n        x = X  # type: str\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      X = ...  # type: None\n      def foo() -> str: ...\n    ')

    def test_cellvar_comment(self):
        if False:
            print('Hello World!')
        'Type comment on an assignment generating the STORE_DEREF opcode.'
        ty = self.Infer('\n      from typing import Mapping\n      def f():\n        map = dict()  # type: Mapping\n        return (map, {x: map.get(y) for x, y in __any_object__})\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Mapping, Tuple\n      def f() -> Tuple[Mapping, dict]: ...\n    ')

    def test_bad_comment(self):
        if False:
            while True:
                i = 10
        (ty, errors) = self.InferWithErrors('\n      X = None  # type: abc def  # invalid-annotation[e]\n    ', deep=True)
        if self.python_version >= (3, 10):
            error_reason = 'invalid syntax'
        else:
            error_reason = 'unexpected EOF'
        self.assertErrorRegexes(errors, {'e': f'abc def.*{error_reason}'})
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      X = ...  # type: Any\n    ')

    def test_conversion_error(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      X = None  # type: 1 if __random__ else 2  # invalid-annotation[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'X.*Must be constant'})
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      X = ...  # type: Any\n    ')

    def test_name_error_inside_comment(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      X = None  # type: Foo  # name-error[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'Foo'})

    def test_warn_on_ignored_type_comment(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      X = []\n      X[0] = None  # type: str  # ignored-type-comment[e1]\n      # type: int  # ignored-type-comment[e2]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e1': 'str', 'e2': 'int'})

    def test_attribute_initialization(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.x = 42\n      a = None  # type: A\n      x = a.x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class A:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n      a = ...  # type: A\n      x = ...  # type: int\n    ')

    def test_none_to_none_type(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      x = None  # type: None\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: None\n    ')

    def test_module_instance_as_bad_type_comment(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      import sys\n      x = None  # type: sys  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'instance of module.*x'})

    def test_forward_reference(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors('\n      a = None  # type: "A"\n      b = None  # type: "Nonexistent"  # name-error[e]\n      class A:\n        def __init__(self):\n          self.x = 42\n        def f(self):\n          return a.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n        def f(self) -> int: ...\n      a = ...  # type: A\n      b = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e': 'Nonexistent'})

    def test_class_variable_forward_reference(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      class A:\n        a = None  # type: 'A'\n        def __init__(self):\n          self.x = 42\n    ")
        self.assertTypesMatchPytd(ty, '\n      class A:\n        a: A\n        x: int\n        def __init__(self) -> None: ...\n    ')

    def test_use_forward_reference(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = None  # type: "A"\n      x = a.x\n      class A:\n        def __init__(self):\n          self.x = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n      a = ...  # type: A\n      x = ...  # type: Any\n    ')

    def test_use_class_variable_forward_reference(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      class A:\n        a = None  # type: 'A'\n        def f(self):\n          return self.a\n      x = A().a\n      def g():\n        return A().a\n      y = g()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, TypeVar\n      _TA = TypeVar('_TA', bound=A)\n      class A:\n        a: A\n        def f(self: _TA) -> _TA: ...\n      x: A\n      y: A\n      def g() -> A: ...\n    ")

    def test_class_variable_forward_reference_error(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors("\n      class A:\n        a = None  # type: 'A'\n      g = A().a.foo()  # attribute-error\n    ")

    def test_multiline_value(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      v = [\n        {\n        "a": 1  # type: complex  # ignored-type-comment[e1]\n\n        }  # type: dict[str, int]  # ignored-type-comment[e2]\n      ]  # type: list[dict[str, float]]\n    ')
        self.assertTypesMatchPytd(ty, '\n      v = ...  # type: list[dict[str, float]]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Stray type comment: complex', 'e2': 'Stray type comment: dict\\[str, int\\]'})

    def test_multiline_value_with_blank_lines(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      a = [[\n\n      ]\n\n      ]  # type: list[list[int]]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      a = ...  # type: list[list[int]]\n    ')

    def test_type_comment_name_error(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f():\n        x = None  # type: Any  # invalid-annotation[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'not defined$'})

    def test_type_comment_invalid_syntax(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f():\n        x = None  # type: y = 1  # invalid-annotation[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'invalid syntax$'})

    def test_discarded_type_comment(self):
        if False:
            while True:
                i = 10
        'Discard the first whole-line comment, keep the second.'
        ty = self.Infer("\n        # We want either # type: ignore or # type: int\n        def hello_world():\n          # type: () -> str\n          return 'hello world'\n    ", deep=True)
        self.assertTypesMatchPytd(ty, '\n      def hello_world() -> str: ...\n    ')

    def test_multiple_type_comments(self):
        if False:
            print('Hello World!')
        'We should not allow multiple type comments on one line.'
        (_, errors) = self.InferWithErrors('\n      a = 42  # type: int  # type: float  # invalid-directive[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Multiple'})

    def test_nested_comment_alias(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A: pass\n      class B:\n        C = A\n        x = None  # type: C\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class A: pass\n      class B:\n        C: Type[A]\n        x: A\n      ')

    def test_nested_classes_comments(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        class B: pass\n        x = None  # type: B\n      ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        class B: ...\n        x: A.B\n      ')

    def test_list_comprehension_comments(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import List\n      def f(x):\n        # type: (str) -> None\n        pass\n      def g(xs):\n        # type: (List[str]) -> List[str]\n        ys = [f(x) for x in xs]  # type: List[str]  # annotation-type-mismatch[e]\n        return ys\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      def f(x: str) -> None: ...\n      def g(xs: List[str]) -> List[str]: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: List\\[str\\].*Assignment: List\\[None\\]'})

    def test_multiple_assignments(self):
        if False:
            return 10
        ty = self.Infer('\n      a = 1; b = 2; c = 4  # type: float\n    ')
        self.assertTypesMatchPytd(ty, '\n      a = ...  # type: int\n      b = ...  # type: int\n      c = ...  # type: float\n    ')

    def test_instantiate_fully_quoted_type(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import Optional\n      x = None  # type: "Optional[A]"\n      class A:\n        a = 0\n      y = x.a  # attribute-error[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional\n      x: Optional[A]\n      class A:\n        a: int\n      y: int\n    ')
        self.assertErrorRegexes(errors, {'e': 'a.*None'})

    def test_do_not_resolve_late_type_to_function(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      v = None  # type: "A"\n      class A:\n        def A(self):\n          pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      v: A\n      class A:\n        def A(self) -> None: ...\n    ')

    def test_illegal_function_late_type(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      v = None  # type: "F"  # invalid-annotation\n      def F(): pass\n    ')

    def test_bad_type_comment_in_constructor(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class Foo:\n        def __init__(self):\n          self.x = None  # type: "Bar"  # name-error\n    ')

    def test_dict_type_comment(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Any, Callable, Dict, Tuple\n      d = {\n          'a': 'long'\n               'string'\n               'value'\n      }  # type: Dict[str, str]\n    ")

    def test_break_on_period(self):
        if False:
            print('Hello World!')
        self.Check("\n      really_really_really_long_module_name = None  # type: module\n      d = {}\n      v = d.get('key', (really_really_really_long_module_name.\n                        also_long_attribute_name))  # type: int\n    ")

    def test_assignment_between_functions(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      def f(): pass\n      x = 0  # type: int\n      def g():\n        '''Docstring.'''\n    ")
        self.assertTypesMatchPytd(ty, '\n      def f() -> None: ...\n      x: int\n      def g() -> None: ...\n    ')

    def test_type_comment_on_class(self):
        if False:
            while True:
                i = 10
        if sys.version_info[:2] >= (3, 9):
            line1_error = ''
            line2_error = '  # ignored-type-comment'
        else:
            line1_error = '  # annotation-type-mismatch'
            line2_error = ''
        self.CheckWithErrors(f'\n      class Foo({line1_error}\n          int):  # type: str{line2_error}\n        pass\n    ')
if __name__ == '__main__':
    test_base.main()