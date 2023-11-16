"""Tests for PEP526 variable annotations."""
from pytype.tests import test_base
from pytype.tests import test_utils

class VariableAnnotationsBasicTest(test_base.BaseTest):
    """Tests for PEP526 variable annotations."""

    def test_pyi_annotations(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List\n        x: int\n        y: List[int]\n        class A:\n          a: int\n          b: str\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        def f(x: int) -> None:\n          pass\n        obj = foo.A()\n        f(foo.x)\n        f(foo.y)  # wrong-arg-types[e1]\n        f(obj.a)\n        f(obj.b)  # wrong-arg-types[e2]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'int.*List', 'e2': 'int.*str'})

    def test_typevar_annot_with_subclass(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def f(self):\n          x: T = None\n          return x\n      class Bar(Foo[str]):\n        pass\n      assert_type(Bar().f(), str)\n    ")

class VariableAnnotationsFeatureTest(test_base.BaseTest):
    """Tests for PEP526 variable annotations."""

    def test_infer_types(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import List\n\n      lst: List[int] = []\n\n      x: int = 1\n      y = 2\n\n      class A:\n        a: int = 1\n        b = 2\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n\n      lst: List[int]\n      x: int\n      y: int\n\n      class A:\n          a: int\n          b: int\n    ')

    def test_illegal_annotations(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import List, TypeVar, NoReturn\n\n      T = TypeVar(\'T\')\n\n      a: "abc" = "1"  # name-error[e1]\n      b: 123 = "2"  # invalid-annotation[e2]\n      c: List[int] = []\n      d: List[T] = []  # invalid-annotation[e3]\n      e: int if __random__ else str = 123  # invalid-annotation[e4]\n    ')
        self.assertErrorRegexes(errors, {'e1': "Name \\'abc\\' is not defined", 'e2': 'Not a type', 'e3': "'T' not in scope", 'e4': 'Must be constant'})

    def test_never(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import NoReturn\n      x: NoReturn = 0  # annotation-type-mismatch[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Annotation: Never', 'Assignment: int']})

    def test_uninitialized_class_annotation(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        bar: int\n        def baz(self):\n          return self.bar\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        bar: int\n        def baz(self) -> int: ...\n    ')

    def test_uninitialized_module_annotation(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      foo: int\n      bar = foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      foo: int\n      bar: int\n    ')

    def test_overwrite_annotations_dict(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      __annotations__ = None\n      foo: int  # unsupported-operands[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'None.*__setitem__'})

    def test_shadow_none(self):
        if False:
            return 10
        ty = self.Infer('\n      v: int = None\n    ')
        self.assertTypesMatchPytd(ty, '\n      v: int\n    ')

    def test_overwrite_annotation(self):
        if False:
            while True:
                i = 10
        (ty, errors) = self.InferWithErrors('\n      x: int\n      x = ""  # annotation-type-mismatch[e]\n    ')
        self.assertTypesMatchPytd(ty, 'x: int')
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_overwrite_annotation_in_class(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      class Foo:\n        x: int\n        x = ""  # annotation-type-mismatch[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x: int\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_class_variable_forward_reference(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      class A:\n        a: 'A' = ...\n        x = 42\n    ")
        self.assertTypesMatchPytd(ty, '\n      class A:\n        a: A\n        x: int\n    ')

    def test_callable_forward_reference(self):
        if False:
            return 10
        ty = self.Infer("\n      from typing import Callable\n      class A:\n        def __init__(self, fn: Callable[['A'], bool]):\n          self.fn = fn\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable\n      class A:\n        fn: Callable[[A], bool]\n        def __init__(self, fn: Callable[[A], bool]) -> None: ...\n    ')

    def test_multiple_forward_reference(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Dict\n      class A:\n        x: Dict['A', 'B']\n      class B:\n        pass\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      class A:\n        x: Dict[A, B]\n      class B: ...\n    ')

    def test_non_annotations_dict(self):
        if False:
            return 10
        self.Check("\n      class K(dict):\n        pass\n      x = K()\n      y: int = 9\n      x['z'] = 5\n    ")

    def test_function_local_annotation(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        x: int = None\n        return x\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    @test_base.skip('b/167613685')
    def test_function_local_annotation_no_assignment(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        x: int\n        return x\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_multi_statement_line(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        if __random__: v: int = None\n        else: v = __any_object__\n        return v\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f() -> Any: ...\n    ')

    def test_multi_line_assignment(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      v: int = (\n          None)\n    ')
        self.assertTypesMatchPytd(ty, 'v: int')

    def test_complex_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import Dict\n      def f():\n        column_map: Dict[str, Dict[str, bool]] = {\n            column: {\n                'visible': True\n            } for column in __any_object__.intersection(\n                __any_object__)\n        }\n        return column_map\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      def f() -> Dict[str, Dict[str, bool]]: ...\n    ')

    def test_none_or_ellipsis_assignment(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      v1: int = None\n      v2: str = ...\n    ')

    def test_any(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Any\n      def f():\n        x: Any = None\n        print(x.upper())\n        x = None\n        print(x.upper())\n    ')

    def test_uninitialized_variable_container_check(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import List\n      x: List[str]\n      x.append(0)  # container-type-mismatch\n    ')

    def test_uninitialized_attribute_container_check(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing import List\n      class Foo:\n        x: List[str]\n        def __init__(self):\n          self.x.append(0)  # container-type-mismatch\n    ')

    def test_any_container(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Any, Dict\n      def f():\n        x: Dict[str, Any] = {}\n        x['a'] = 'b'\n        for v in x.values():\n          print(v.whatever)\n        return x\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      def f() -> Dict[str, Any]: ...\n    ')

    def test_function_parameter(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import TypeVar\n      T = TypeVar('T')\n      def f(x: T, y: T):\n        z: T = x\n        return z\n      assert_type(f(0, 1), int)\n    ")

    def test_illegal_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import TypeVar\n      T = TypeVar('T')\n      S = TypeVar('S')\n      def f(x: T, y: T):\n        z: S = x  # invalid-annotation[e]\n        return z\n    ")
        self.assertErrorRegexes(errors, {'e': "'S' not in scope for method 'f'"})

    def test_callable_parameters(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      f: Callable[[T, T], T]\n      assert_type(f(0, 1), int)\n      f(0, '1')  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*int.*Actual.*str'})

    def test_nested_callable_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Callable, List, TypeVar\n      T = TypeVar('T')\n      fs: List[Callable[[T], T]]\n      assert_type(fs[0]('hello world'), str)\n    ")

    def test_callable_parameters_in_method(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      def f():\n        g: Callable[[T], T] = None\n        assert_type(g(0), int)\n    ")

    def test_class_and_callable_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import Callable, Generic, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      class Foo(Generic[T1]):\n        x: Callable[[T1, T2], T2]\n        def f(self):\n          x: Callable[[T1, T2], T2] = None\n          return x\n      foo = Foo[int]()\n      assert_type(foo.x(0, 'hello world'), str)\n      assert_type(foo.f()(0, 4.2), float)\n      foo.x(None, 'hello world')  # wrong-arg-types[e1]\n      foo.f()('oops', 4.2)  # wrong-arg-types[e2]\n    ")
        self.assertErrorRegexes(errors, {'e1': 'Expected.*int.*Actual.*None', 'e2': 'Expected.*int.*Actual.*str'})

    def test_invalid_callable_parameter(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      f: Callable[[T], int]  # invalid-annotation\n      def g(x: T, y: T):\n        f2: Callable[[T], int]  # ok, since T is from the signature of g\n    ")

    def test_typevar_annot_and_list_comprehension(self):
        if False:
            print('Hello World!')
        self.Check("\n      from collections import defaultdict\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n\n      class Min(Generic[T]):\n        def __init__(self, items: list[T]):\n          self.min = 2\n          self.items = items\n        def __call__(self) -> list[T]:\n          counts: defaultdict[T, int] = defaultdict(int)\n          return [b for b in self.items if counts[b] >= self.min]\n    ")
if __name__ == '__main__':
    test_base.main()