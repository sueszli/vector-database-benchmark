"""Tests for TypeVar."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TypeVarTest(test_base.BaseTest):
    """Tests for TypeVar."""

    def test_unused_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import TypeVar\n      T = TypeVar("T")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar\n      T = TypeVar("T")\n    ')

    def test_import_typevar(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'T = TypeVar("T")')
            ty = self.Infer('\n        from a import T\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import TypeVar\n        T = TypeVar("T")\n      ')

    def test_invalid_typevar(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      typevar = TypeVar\n      T = typevar()  # invalid-typevar[e1]\n      T = typevar("T")  # ok\n      T = typevar(42)  # invalid-typevar[e2]\n      T = typevar(str())  # invalid-typevar[e3]\n      T = typevar("T", str, int if __random__ else float)  # invalid-typevar[e4]\n      T = typevar("T", 0, float)  # invalid-typevar[e5]\n      T = typevar("T", str)  # invalid-typevar[e6]\n      # pytype: disable=not-supported-yet\n      S = typevar("S", covariant=False)  # ok\n      T = typevar("T", covariant=False)  # duplicate ok\n      # pytype: enable=not-supported-yet\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar\n      typevar = ...  # type: type\n      S = TypeVar("S")\n      T = TypeVar("T")\n    ')
        self.assertErrorRegexes(errors, {'e1': 'wrong arguments', 'e2': 'Expected.*str.*Actual.*int', 'e3': 'constant str', 'e4': 'constraint.*Must be constant', 'e5': 'Expected.*_1:.*type.*Actual.*_1: int', 'e6': '0 or more than 1'})

    def test_print_constraints(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import List, TypeVar\n      S = TypeVar("S", int, float, covariant=True)  # pytype: disable=not-supported-yet\n      T = TypeVar("T", int, float)\n      U = TypeVar("U", List[int], List[float])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, TypeVar\n      S = TypeVar("S", int, float)\n      T = TypeVar("T", int, float)\n      U = TypeVar("U", List[int], List[float])\n    ')

    def test_infer_typevars(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def id(x):\n        return x\n      def wrap_tuple(x, y):\n        return (x, y)\n      def wrap_list(x, y):\n        return [x, y]\n      def wrap_dict(x, y):\n        return {x: y}\n      def return_second(x, y):\n        return y\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Tuple, Union\n      _T0 = TypeVar("_T0")\n      _T1 = TypeVar("_T1")\n      def id(x: _T0) -> _T0: ...\n      def wrap_tuple(x: _T0, y: _T1) -> Tuple[_T0, _T1]: ...\n      def wrap_list(x: _T0, y: _T1) -> List[Union[_T0, _T1]]: ...\n      def wrap_dict(x: _T0, y: _T1) -> Dict[_T0, _T1]: ...\n      def return_second(x, y: _T1) -> _T1: ...\n    ')

    def test_infer_union(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def return_either(x, y):\n        return x or y\n      def return_arg_or_42(x):\n        return x or 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      _T0 = TypeVar("_T0")\n      _T1 = TypeVar("_T1")\n      def return_either(x: _T0, y: _T1) -> Union[_T0, _T1]: ...\n      def return_arg_or_42(x: _T0) -> Union[_T0, int]: ...\n    ')

    def test_typevar_in_type_comment(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      x = None  # type: T  # invalid-annotation\n      y = None  # type: List[T]  # invalid-annotation\n    ')

    def test_base_class_with_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      class A(List[T]): pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      class A(List[T]): ...\n    ')

    def test_overwrite_base_class_with_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      l = List[T]\n      l = list\n      class X(l): pass\n    ')

    def test_bound(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", int, float, bound=str)  # invalid-typevar\n      S = TypeVar("S", bound="")  # invalid-typevar\n      U = TypeVar("U", bound=str)  # ok\n      V = TypeVar("V", bound=int if __random__ else float)  # invalid-typevar\n    ')

    def test_covariant(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", covariant=True)  # not-supported-yet\n      S = TypeVar("S", covariant=42)  # invalid-typevar[e1]\n      U = TypeVar("U", covariant=True if __random__ else False)  # invalid-typevar[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Expected.*bool.*Actual.*int', 'e2': 'constant'})

    def test_contravariant(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", contravariant=True)  # not-supported-yet\n      S = TypeVar("S", contravariant=42)  # invalid-typevar[e1]\n      U = TypeVar("U", contravariant=True if __random__ else False)  # invalid-typevar[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Expected.*bool.*Actual.*int', 'e2': 'constant'})

    def test_dont_propagate_pyval(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import TypeVar\n        AnyInt = TypeVar('AnyInt', int)\n        def f(x: AnyInt) -> AnyInt: ...\n      ")
            ty = self.Infer('\n        import a\n        if a.f(0):\n          x = 3\n        if a.f(1):\n          y = 3\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n        y = ...  # type: int\n      ')

    def test_property_type_param(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, List\n      T = TypeVar('T')\n      class A:\n          @property\n          def foo(self: T) -> List[T]: ...\n      class B(A): ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.A().foo\n        y = a.B().foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import List\n        x = ...  # type: List[a.A]\n        y = ...  # type: List[a.B]\n      ')

    def test_property_type_param2(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, List, Generic\n      T = TypeVar('T')\n      U = TypeVar('U')\n      class A(Generic[U]):\n          @property\n          def foo(self: T) -> List[T]: ...\n      class B(A, Generic[U]): ...\n      def make_A() -> A[int]: ...\n      def make_B() -> B[int]: ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.make_A().foo\n        y = a.make_B().foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import List\n        x = ...  # type: List[a.A[int]]\n        y = ...  # type: List[a.B[int]]\n      ')

    @test_base.skip('Type parameter bug')
    def test_property_type_param3(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, List, Generic\n      T = TypeVar('T')\n      U = TypeVar('U')\n      class A(Generic[U]):\n          @property\n          def foo(self: T) -> List[U]: ...\n      def make_A() -> A[int]: ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.make_A().foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: List[int]\n      ')

    def test_property_type_param_with_constraints(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, List, Generic\n      T = TypeVar('T')\n      U = TypeVar('U', int, str)\n      X = TypeVar('X', int)\n      class A(Generic[U]):\n          @property\n          def foo(self: A[X]) -> List[X]: ...\n      def make_A() -> A[int]: ...\n      ")
            ty = self.Infer('\n        import a\n        x = a.make_A().foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import List\n        x = ...  # type: List[int]\n      ')

    def test_classmethod_type_param(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, List, Type\n      T = TypeVar('T')\n      class A:\n          @classmethod\n          def foo(self: Type[T]) -> List[T]: ...\n      class B(A): ...\n      ")
            ty = self.Infer('\n        import a\n        v = a.A.foo()\n        w = a.B.foo()\n        x = a.A().foo()\n        y = a.B().foo()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import List\n        v = ...  # type: List[a.A]\n        w = ...  # type: List[a.B]\n        x = ...  # type: List[a.A]\n        y = ...  # type: List[a.B]\n      ')

    def test_metaclass_property_type_param(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n      from typing import TypeVar, Type, List\n      T = TypeVar('T')\n      class Meta():\n        @property\n        def foo(self: Type[T]) -> List[T]: ...\n\n      class A(metaclass=Meta):\n        pass\n      ")
            ty = self.Infer('\n        import a\n        x = a.A.foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import List\n        x = ...  # type: List[a.A]\n      ')

    def test_top_level_union(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import TypeVar\n      if __random__:\n        T = TypeVar("T")\n      else:\n        T = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      T = ...  # type: Any\n    ')

    def test_store_typevar_in_dict(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import TypeVar\n      T = TypeVar("T")\n      a = {\'key\': T}\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Dict, TypeVar\n      a = ...  # type: Dict[str, nothing]\n      T = TypeVar('T')\n    ")

    def test_late_bound(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar, Union\n      T = TypeVar("T", int, float, bound="str")  # invalid-typevar[e1]\n      S = TypeVar("S", bound="")  # invalid-typevar[e2]\n      U = TypeVar("U", bound="str")  # ok\n      V = TypeVar("V", bound="int if __random__ else float")  # invalid-typevar[e3]\n      W = TypeVar("W", bound="Foo") # ok, forward reference\n      X = TypeVar("X", bound="Bar")  # name-error[e4]\n      class Foo:\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e1': 'mutually exclusive', 'e2': 'empty string', 'e3': 'Must be constant', 'e4': 'Name.*Bar'})

    def test_late_constraints(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import List, TypeVar\n      S = TypeVar("S", int, float)\n      T = TypeVar("T", "int", "float")\n      U = TypeVar("U", "List[int]", List[float])\n      V = TypeVar("V", "Foo", "List[Foo]")\n      class Foo:\n        pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, TypeVar\n      S = TypeVar("S", int, float)\n      T = TypeVar("T", int, float)\n      U = TypeVar("U", List[int], List[float])\n      V = TypeVar("V", Foo, List[Foo])\n      class Foo:\n        pass\n    ')

    def test_typevar_in_alias(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import TypeVar, Union\n      T = TypeVar("T", int, float)\n      Num = Union[T, complex]\n      x = 10  # type: Num[int]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar, Union\n      T = TypeVar("T", int, float)\n      Num = Union[T, complex]\n      x: Union[int, complex]\n    ')

    def test_type_of_typevar(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Sequence, TypeVar\n      T = TypeVar('T')\n      def f(x):  # type: (Sequence[T]) -> Sequence[T]\n        print(type(x))\n        return x\n    ")

    def test_type_of_typevar_error(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Sequence, Type, TypeVar\n      T = TypeVar('T')\n      def f(x):  # type: (int) -> int\n        return x\n      def g(x):  # type: (Sequence[T]) -> Type[Sequence[T]]\n        return f(type(x))  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*int.*Actual.*Sequence'})

    def test_typevar_in_constant(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import TypeVar\n      T = TypeVar('T')\n      class Foo:\n        def __init__(self):\n          self.f1 = self.f2\n        def f2(self, x):\n          # type: (T) -> T\n          return x\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      class Foo:\n        f1: Callable[[T], T]\n        def __init__(self) -> None: ...\n        def f2(self, x: T) -> T: ...\n    ")

    def test_extra_arguments(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", extra_arg=42)  # invalid-typevar[e1]\n      S = TypeVar("S", *__any_object__)  # invalid-typevar[e2]\n      U = TypeVar("U", **__any_object__)  # invalid-typevar[e3]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'extra_arg', 'e2': '\\*args', 'e3': '\\*\\*kwargs'})

    def test_simplify_args_and_kwargs(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import TypeVar\n      constraints = (int, str)\n      kwargs = {"covariant": True}\n      T = TypeVar("T", *constraints, **kwargs)  # pytype: disable=not-supported-yet\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Tuple, Type, TypeVar\n      T = TypeVar("T", int, str)\n      constraints = ...  # type: Tuple[Type[int], Type[str]]\n      kwargs = ...  # type: Dict[str, bool]\n    ')

    def test_typevar_starargs(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Generic, TypeVar, Union\n        T = TypeVar('T')\n        S = TypeVar('S')\n        SS = TypeVar('SS')\n        class A(Generic[T]):\n          def __init__(self, x: T, *args: S, **kwargs: SS):\n            self = A[Union[T, S, SS]]\n      ")
            self.Check('\n        import a\n        a.A(1)\n        a.A(1, 2, 3)\n        a.A(1, 2, 3, a=1, b=2)\n      ', pythonpath=[d.path])

    def test_cast_generic_callable(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Callable, TypeVar, cast\n      T = TypeVar('T')\n      def f(x):\n        return cast(Callable[[T, T], T], x)\n      assert_type(f(None)(0, 1), int)\n      f(None)(0, '1')  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*int.*Actual.*str'})
if __name__ == '__main__':
    test_base.main()