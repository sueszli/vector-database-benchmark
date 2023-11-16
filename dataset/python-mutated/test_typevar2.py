"""Tests for TypeVar."""
import sys
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TypeVarTest(test_base.BaseTest):
    """Tests for TypeVar."""

    def test_id(self):
        if False:
            return 10
        ty = self.Infer('\n      import typing\n      T = typing.TypeVar("T")\n      def f(x: T) -> T:\n        return __any_object__\n      v = f(42)\n      w = f("")\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      from typing import Any\n      T = TypeVar("T")\n      def f(x: T) -> T: ...\n      v = ...  # type: int\n      w = ...  # type: str\n    ')
        self.assertTrue(ty.Lookup('f').signatures[0].template)

    def test_extract_item(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import List, TypeVar\n      S = TypeVar("S")  # unused\n      T = TypeVar("T")\n      def f(x: List[T]) -> T:\n        return __any_object__\n      v = f(["hello world"])\n      w = f([True])\n    ')
        self.assertTypesMatchPytd(ty, '\n      S = TypeVar("S")\n      T = TypeVar("T")\n      def f(x: typing.List[T]) -> T: ...\n      v = ...  # type: str\n      w = ...  # type: bool\n    ')
        self.assertTrue(ty.Lookup('f').signatures[0].template)

    def test_wrap_item(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      def f(x: T) -> List[T]:\n        return __any_object__\n      v = f(True)\n      w = f(3.14)\n    ')
        self.assertTypesMatchPytd(ty, '\n      T = TypeVar("T")\n      def f(x: T) -> typing.List[T]: ...\n      v = ...  # type: typing.List[bool]\n      w = ...  # type: typing.List[float]\n    ')

    def test_import_typevar_name_change(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        X = TypeVar("X")\n      ')
            (_, errors) = self.InferWithErrors('\n        # This is illegal: A TypeVar("T") needs to be stored under the name "T".\n        from a import T as T2  # invalid-typevar[e1]\n        from a import X\n        Y = X  # invalid-typevar[e2]\n        def f(x: T2) -> T2: ...\n      ', pythonpath=[d.path])
        self.assertErrorRegexes(errors, {'e1': 'T.*T2', 'e2': 'X.*Y'})

    def test_typevar_in_typevar(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      S1 = TypeVar('S1', bound=T1)  # invalid-typevar\n      S2 = TypeVar('S2', T1, T2)  # invalid-typevar\n      # Using the invalid TypeVar should not produce an error.\n      class Foo(Generic[S1]):\n        pass\n    ")

    def test_multiple_substitution(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Dict, Tuple, TypeVar\n      K = TypeVar("K")\n      V = TypeVar("V")\n      def f(x: Dict[K, V]) -> Tuple[V, K]:\n        return __any_object__\n      v = f({})\n      w = f({"test": 42})\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict, Tuple, TypeVar\n      K = TypeVar("K")\n      V = TypeVar("V")\n      def f(x: Dict[K, V]) -> Tuple[V, K]: ...\n      v = ...  # type: Tuple[Any, Any]\n      w = ...  # type: Tuple[int, str]\n    ')

    def test_union(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import TypeVar, Union\n      S = TypeVar("S")\n      T = TypeVar("T")\n      def f(x: S, y: T) -> Union[S, T]:\n        return __any_object__\n      v = f("", 42)\n      w = f(3.14, False)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar, Union\n      S = TypeVar("S")\n      T = TypeVar("T")\n      def f(x: S, y: T) -> Union[S, T]: ...\n      v = ...  # type: Union[str, int]\n      w = ...  # type: Union[float, bool]\n    ')

    def test_bad_substitution(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import List, TypeVar\n      S = TypeVar("S")\n      T = TypeVar("T")\n      def f1(x: S) -> List[S]:\n        return {x}  # bad-return-type[e1]\n      def f2(x: S) -> S:\n        return 42  # no error because never called\n      def f3(x: S) -> S:\n        return 42  # bad-return-type[e2]  # bad-return-type[e3]\n      def f4(x: S, y: T, z: T) -> List[S]:\n        return [y]  # bad-return-type[e4]\n      f3("")\n      f3(16)  # ok\n      f3(False)\n      f4(True, 3.14, 0)\n      f4("hello", "world", "domination")  # ok\n    ')
        self.assertErrorRegexes(errors, {'e1': 'list.*set', 'e2': 'str.*int', 'e3': 'bool.*int', 'e4': 'List\\[bool\\].*List\\[Union\\[float, int\\]\\]'})

    def test_use_constraints(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", int, float)\n      def f(x: T) -> T:\n        return __any_object__\n      v = f("")  # wrong-arg-types[e]\n      w = f(True)  # ok\n      u = f(__any_object__)  # ok\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, TypeVar, Union\n      T = TypeVar("T", int, float)\n      def f(x: T) -> T: ...\n      v = ...  # type: Any\n      w = ...  # type: bool\n      u = ...  # type: Union[int, float]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Union\\[float, int\\].*str'})

    def test_type_parameter_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import Type, TypeVar\n      T = TypeVar("T")\n      def f(x: Type[T]) -> T:\n        return __any_object__\n      v = f(int)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type, TypeVar\n      T = TypeVar("T")\n      def f(x: Type[T]) -> T: ...\n      v = ...  # type: int\n    ')

    def test_type_parameter_type_error(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Sequence, Type, TypeVar\n      T = TypeVar('T')\n      def f(x: int):\n        pass\n      def g(x: Type[Sequence[T]]) -> T:\n        print(f(x))  # wrong-arg-types[e]\n        return x()[0]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*int.*Actual.*Type\\[Sequence\\]'})

    def test_print_nested_type_parameter(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import List, TypeVar\n      T = TypeVar("T", int, float)\n      def f(x: List[T]): ...\n      f([""])  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'List\\[Union\\[float, int\\]\\].*List\\[str\\]'})

    def test_constraint_subtyping(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", int, float)\n      def f(x: T, y: T): ...\n      f(True, False)  # ok\n      f(True, 42)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*y: bool.*Actual.*y: int'})

    def test_filter_value(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", str, float)\n      def f(x: T, y: T): ...\n      x = \'\'\n      x = 42.0\n      f(x, \'\')  # wrong-arg-types[e]\n      f(x, 42.0)  # ok\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*y: float.*Actual.*y: str'})

    def test_filter_class(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import TypeVar\n      class A: pass\n      class B: pass\n      T = TypeVar("T", A, B)\n      def f(x: T, y: T): ...\n      x = A()\n      x.__class__ = B\n      # Setting __class__ makes the type ambiguous to pytype.\n      f(x, A())\n      f(x, B())\n    ')

    def test_split(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import TypeVar\n      T = TypeVar("T", int, type(None))\n      def f(x: T) -> T:\n        return __any_object__\n      if __random__:\n        x = None\n      else:\n        x = 3\n      v = id(x) if x else 42\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional, TypeVar\n      v = ...  # type: int\n      x = ...  # type: Optional[int]\n      T = TypeVar("T", int, None)\n      def f(x: T) -> T: ...\n    ')

    def test_enforce_non_constrained_typevar(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T")\n      def f(x: T, y: T): ...\n      f(42, True)  # ok\n      f(42, "")  # wrong-arg-types[e1]\n      f(42, 16j)  # ok\n      f(object(), 42)  # ok\n      f(42, object())  # ok\n      f(42.0, "")  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Expected.*y: int.*Actual.*y: str', 'e2': 'Expected.*y: float.*Actual.*y: str'})

    def test_useless_typevar(self):
        if False:
            return 10
        self.InferWithErrors('\n      from typing import Tuple, TypeVar\n      T = TypeVar("T")\n      S = TypeVar("S", int, float)\n      def f1(x: T): ...  # invalid-annotation\n      def f2() -> T: ...  # invalid-annotation\n      def f3(x: Tuple[T]): ...  # invalid-annotation\n      def f4(x: Tuple[T, T]): ...  # ok\n      def f5(x: S): ...  # ok\n      def f6(x: "U"): ...  # invalid-annotation\n      def f7(x: T, y: "T"): ...  # ok\n      def f8(x: "U") -> "U": ...  # ok\n      U = TypeVar("U")\n    ')

    def test_use_bound(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import TypeVar\n      T = TypeVar("T", bound=float)\n      def f(x: T) -> T:\n        return x\n      v1 = f(__any_object__)  # ok\n      v2 = f(True)  # ok\n      v3 = f(42)  # ok\n      v4 = f(3.14)  # ok\n      v5 = f("")  # wrong-arg-types[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, TypeVar\n      T = TypeVar("T", bound=float)\n      def f(x: T) -> T: ...\n      v1 = ...  # type: float\n      v2 = ...  # type: bool\n      v3 = ...  # type: int\n      v4 = ...  # type: float\n      v5 = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e': 'x: float.*x: str'})

    def test_bad_return(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Check, "\n      from typing import AnyStr, Dict\n\n      class Foo:\n        def f(self) -> AnyStr: return __any_object__\n        def g(self) -> Dict[AnyStr, Dict[AnyStr, AnyStr]]:\n          return {'foo': {'bar': self.f()}}\n    ")

    def test_optional_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import Optional, TypeVar\n      T = TypeVar("T", bound=str)\n      def f() -> Optional[T]:\n        return 42 if __random__ else None  # bad-return-type[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'Optional\\[str\\].*int'})

    def test_unicode_literals(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from __future__ import unicode_literals\n      import typing\n      T = typing.TypeVar("T")\n      def f(x: T) -> T:\n        return __any_object__\n      v = f(42)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      from typing import Any\n      T = TypeVar("T")\n      def f(x: T) -> T: ...\n      v = ...  # type: int\n    ')

    def test_any_as_bound(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Any, TypeVar\n      T = TypeVar("T", bound=Any)\n      def f(x: T) -> T:\n        return x\n      f(42)\n    ')

    def test_any_as_constraint(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Any, TypeVar\n      T = TypeVar("T", str, Any)\n      def f(x: T) -> T:\n        return x\n      f(42)\n    ')

    def test_name_reuse(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Generic, TypeVar\n      T = TypeVar("T", int, float)\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n      def f(foo: Foo[T]) -> T:\n        return foo.x\n    ')

    def test_property_type_param(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import TypeVar, Generic\n      T = TypeVar('T')\n      class A(Generic[T]):\n          def __init__(self, foo: T):\n              self._foo = foo\n          @property\n          def foo(self) -> T:\n              return self._foo\n          @foo.setter\n          def foo(self, foo: T) -> None:\n              self._foo = foo\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import TypeVar, Generic, Any, Annotated\n      T = TypeVar('T')\n      class A(Generic[T]):\n          _foo: T\n          foo: Annotated[T, 'property']\n          def __init__(self, foo: T) -> None:\n            self = A[T]\n    ")

    @test_base.skip('Needs improvements to matcher.py to detect error.')
    def test_return_typevar(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import TypeVar\n      T = TypeVar('T')\n      def f(x: T) -> T:\n        return T  # bad-return-type[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*T.*Actual.*TypeVar'})

    def test_typevar_in_union_alias(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import Dict, List, TypeVar, Union\n      T = TypeVar("T")\n      U = TypeVar("U")\n      Foo = Union[T, List[T], Dict[T, List[U]], complex]\n      def f(x: Foo[int, str]): ...\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, TypeVar, Union\n      T = TypeVar("T")\n      U = TypeVar("U")\n      Foo = Union[T, List[T], Dict[T, List[U]], complex]\n      def f(x: Union[Dict[int, List[str]], List[int], complex, int]) -> None: ...\n    ')

    def test_typevar_in_union_alias_error(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      from typing import Dict, List, TypeVar, Union\n      T = TypeVar("T")\n      U = TypeVar("U")\n      Foo = Union[T, List[T], Dict[T, List[U]], complex]\n      def f(x: Foo[int]): ...  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(err, {'e': 'Union.*2.*got.*1'})

    def test_cast_generic_tuple(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Tuple, TypeVar, cast\n      T = TypeVar('T')\n      def f(x: T, y: T):\n        return cast(Tuple[T, ...], x)\n      assert_type(f(0, 1), Tuple[int, ...])\n    ")

    def test_cast_in_instance_method(self):
        if False:
            return 10
        self.Check("\n      from typing import TypeVar, cast\n      T = TypeVar('T', bound='Base')\n      class Base:\n        def clone(self: T) -> T:\n          return cast(T, __any_object__)\n      class Child(Base):\n        pass\n      Child().clone()\n    ")

    def test_typevar_in_nested_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import TypeVar\n      T = TypeVar('T')\n      def f(x: T):\n        def wrapper(x: T):\n          pass\n        return wrapper\n    ")

    def test_typevar_in_nested_function_in_instance_method(self):
        if False:
            return 10
        self.Check("\n      from typing import TypeVar\n      T = TypeVar('T')\n      class Foo:\n        def f(self, x: T):\n          def g(x: T):\n            pass\n    ")

    def test_pass_through_class(self):
        if False:
            return 10
        self.Check("\n      from typing import Type, TypeVar\n      T = TypeVar('T')\n      def f(cls: Type[T]) -> Type[T]:\n        return cls\n    ")

    @test_base.skip('Requires completing TODO in annotation_utils.deformalize')
    def test_type_of_typevar(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Type, TypeVar\n      T = TypeVar('T', bound=int)\n      class Foo:\n        def f(self, x: T) -> Type[T]:\n          return type(x)\n        def g(self):\n          assert_type(self.f(0), Type[int])\n    ")

    def test_instantiate_unsubstituted_typevar(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Type, TypeVar\n      T = TypeVar('T', bound=int)\n      def f() -> Type[T]:\n        return int\n      def g():\n        return f().__name__\n    ")

    def test_class_typevar_in_nested_method(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n        def f(self):\n          def g() -> T:\n            return self.x\n          return g()\n      assert_type(Foo(0).f(), int)\n    ")

    def test_self_annotation_in_base_class(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import TypeVar\n      T = TypeVar('T', bound='Base')\n      class Base:\n        def resolve(self: T) -> T:\n          return self\n      class Child(Base):\n        def resolve(self: T) -> T:\n          assert_type(Base().resolve(), Base)\n          return self\n      assert_type(Child().resolve(), Child)\n    ")

    def test_union_against_typevar(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Callable, Iterable, TypeVar, Union\n      T = TypeVar('T')\n      def f(x: Callable[[T], int], y: Iterable[T]):\n        pass\n\n      def g(x: Union[int, str]):\n        return 0\n\n      f(g, [0, ''])\n    ")

    def test_callable_instance_against_callable(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing import Any, Callable, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2', bound=int)\n\n      def f() -> Callable[[T2], T2]:\n        return __any_object__\n\n      # Passing f() to g is an error because g expects a callable with an\n      # unconstrained parameter type.\n      def g(x: Callable[[T1], T1]):\n        pass\n      g(f())  # wrong-arg-types\n\n      # Passing f() to h is okay because T1 in this Callable is just being used\n      # to save the parameter type for h's return type.\n      def h(x: Callable[[T1], Any]) -> T1:\n        return __any_object__\n      h(f())\n    ")

    def test_future_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from __future__ import annotations\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      x: Callable[[T], T] = lambda x: x\n    ")

    def test_imported_typevar_in_scope(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', "\n      from typing import TypeVar\n      T = TypeVar('T')\n    ")]):
            self.Check('\n        import foo\n        def f(x: foo.T) -> foo.T:\n          y: foo.T = x\n          return y\n      ')

    def test_bad_typevar_in_pyi(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', "\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      class C:\n        f: Callable[..., T]\n    ")]):
            self.assertNoCrash(self.Check, '\n        import foo\n        class C(foo.C):\n          def g(self):\n            return self.f(0)\n      ')

class GenericTypeAliasTest(test_base.BaseTest):
    """Tests for generic type aliases ("type macros")."""

    def test_homogeneous_tuple(self):
        if False:
            return 10
        ty = self.Infer("\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[T, ...]\n\n      def f(x: X[int]):\n        pass\n\n      f((0, 1, 2))  # should not raise an error\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[T, ...]\n\n      def f(x: Tuple[int, ...]) -> None: ...\n    ")

    def test_heterogeneous_tuple(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors("\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[T]\n      def f(x: X[int]):\n        pass\n      f((0, 1, 2))  # wrong-arg-types\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Tuple, TypeVar\n      T = TypeVar('T')\n      X = Tuple[T]\n      def f(x: Tuple[int]) -> None: ...\n    ")

    def test_substitute_typevar(self):
        if False:
            print('Hello World!')
        foo_ty = self.Infer("\n      from typing import List, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      X = List[T1]\n      def f(x: X[T2]) -> T2:\n        return x[0]\n    ")
        self.assertTypesMatchPytd(foo_ty, "\n      from typing import List, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      X = List[T1]\n      def f(x: List[T2]) -> T2: ...\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        def f(x: T) -> foo.X[T]:\n          return [x]\n      ", pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import foo\n        from typing import List, TypeVar\n        T = TypeVar('T')\n        def f(x: T) -> List[T]: ...\n      ")

    def test_substitute_value(self):
        if False:
            for i in range(10):
                print('nop')
        foo_ty = self.Infer("\n      from typing import List, TypeVar\n      T = TypeVar('T')\n      X = List[T]\n      def f(x: X[int]) -> int:\n        return x[0]\n    ")
        self.assertTypesMatchPytd(foo_ty, "\n      from typing import List, TypeVar\n      T = TypeVar('T')\n      X = List[T]\n      def f(x: List[int]) -> int: ...\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer('\n        import foo\n        def f(x: int) -> foo.X[int]:\n          return [x]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import List\n        def f(x: int) -> List[int]: ...\n      ')

    def test_partial_substitution(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import Dict, TypeVar\n      T = TypeVar('T')\n      X = Dict[T, str]\n      def f(x: X[int]) -> int:\n        return next(iter(x.keys()))\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Dict, TypeVar\n      T = TypeVar('T')\n      X = Dict[T, str]\n      def f(x: Dict[int, str]) -> int: ...\n    ")

    def test_callable(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      X = Callable[[T], str]\n      def f() -> X[int]:\n        def g(x: int):\n          return str(x)\n        return g\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      X = Callable[[T], str]\n      def f() -> Callable[[int], str]: ...\n    ")

    def test_import_callable(self):
        if False:
            print('Hello World!')
        foo = self.Infer("\n      from typing import TypeVar\n      T = TypeVar('T')\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            bar = self.Infer('\n        import foo\n        from typing import Callable\n        X = Callable[[foo.T], foo.T]\n      ', pythonpath=[d.path])
            d.create_file('bar.pyi', pytd_utils.Print(bar))
            ty = self.Infer('\n        import foo\n        import bar\n        def f(x: foo.T, y: bar.X[foo.T]):\n          pass\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import bar\n        import foo\n        from typing import Callable, TypeVar\n        T = TypeVar('T')\n        def f(x: T, y: Callable[[T], T]) -> None: ...\n      ")

    def test_union_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import TypeVar, Union\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      X = Union[int, T1]\n      def f(x: X[T2], y: T2):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import TypeVar, Union\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      X = Union[int, T1]\n      def f(x: Union[int, T2], y: T2) -> None: ...\n    ")

    def test_union_value(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import TypeVar, Union\n      T = TypeVar('T')\n      X = Union[int, T]\n      def f(x: X[str]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Union, TypeVar\n      T = TypeVar('T')\n      X = Union[int, T]\n      def f(x: Union[int, str]) -> None: ...\n    ")

    def test_extra_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import Dict, TypeVar\n      T = TypeVar('T')\n      X = Dict[T, T]\n      def f(x: X[int, str]):  # invalid-annotation[e]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected 1 parameter, got 2'})

    def test_missing_parameter(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors("\n      from typing import Dict, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      X = Dict[T1, T2]\n      def f(x: X[int]):  # invalid-annotation[e]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected 2 parameters, got 1'})

    def test_nested_typevars(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: X[float, str]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: Callable[[int], Dict[float, str]]) -> None: ...\n    ")

    def test_extra_nested_parameter(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors("\n      from typing import Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: X[float, str, complex]):  # invalid-annotation[e]\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: Callable[[int], Dict[float, str]]) -> None: ...\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected 2 parameters, got 3'})

    def test_missing_nested_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors("\n      from typing import Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: X[float]):  # invalid-annotation[e]\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Callable, Dict, TypeVar\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Callable[[int], Dict[K, V]]\n      def f(x: Callable[[int], Dict[float, Any]]) -> None: ...\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected 2 parameters, got 1'})

    def test_reingest_union(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer("\n      from typing import Optional, TypeVar\n      T = TypeVar('T')\n      X = Optional[T]\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        def f1(x: foo.X[int]):\n          pass\n        def f2(x: foo.X[T]) -> T:\n          assert x\n          return x\n      ", pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, "\n      import foo\n      from typing import Optional, TypeVar\n      T = TypeVar('T')\n      def f1(x: Optional[int]) -> None: ...\n      def f2(x: Optional[T]) -> T: ...\n    ")

    def test_multiple_options(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import Any, Mapping, Sequence, TypeVar, Union\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Union[Sequence, Mapping[K, Any], V]\n      try:\n        Y = X[str, V]\n      except TypeError:\n        Y = Union[Sequence, Mapping[str, Any], V]\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Mapping, Sequence, TypeVar, Union\n      K = TypeVar('K')\n      V = TypeVar('V')\n      X = Union[Sequence, Mapping[K, Any], V]\n      Y = Union[Sequence, Mapping[str, Any], V]\n    ")

    def test_multiple_typevar_options(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import TypeVar\n      if __random__:\n        T1 = TypeVar('T1')\n        T2 = TypeVar('T2')\n      else:\n        T1 = TypeVar('T1')\n        T2 = TypeVar('T2', bound=str)\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, TypeVar\n      T1 = TypeVar('T1')\n      T2: Any\n    ")

    def test_unparameterized_typevar_alias(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors("\n      from typing import TypeVar\n      T = TypeVar('T')\n      U = TypeVar('U')\n      Foo = list[T]\n      Bar = dict[T, U]\n      def f(x: Foo) -> int:  # invalid-annotation[e]\n        return 42\n      def g(x: Foo) -> T:\n        return 42\n      def h(x: Foo[T]) -> int:  # invalid-annotation\n        return 42\n      def h(x: Foo, y: Bar) -> int:  # invalid-annotation\n        return 42\n      def j(x: Foo, y: Bar, z: U) -> int:\n        return 42\n    ")
        if sys.version_info[:2] >= (3, 9):
            self.assertErrorSequences(err, {'e': ['Foo is a generic alias']})

class TypeVarTestPy3(test_base.BaseTest):
    """Tests for TypeVar in Python 3."""

    def test_use_constraints_from_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import AnyStr, TypeVar\n        T = TypeVar("T", int, float)\n        def f(x: T) -> T: ...\n        def g(x: AnyStr) -> AnyStr: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.f("")  # wrong-arg-types[e1]\n        foo.g(0)  # wrong-arg-types[e2]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'Union\\[float, int\\].*str', 'e2': 'Union\\[bytes, str\\].*int'})

    def test_subprocess(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import subprocess\n      from typing import List\n      def run(args: List[str]):\n        result = subprocess.run(\n          args,\n          stdout=subprocess.PIPE,\n          stderr=subprocess.PIPE,\n          universal_newlines=True)\n        if result.returncode:\n          raise subprocess.CalledProcessError(\n              result.returncode, args, result.stdout)\n        return result.stdout\n    ')
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      from typing import List\n      def run(args: List[str]) -> str: ...\n    ')

    def test_abstract_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from abc import ABC, abstractmethod\n      from typing import Type, TypeVar\n\n      T = TypeVar('T', bound='Foo')\n\n      class Foo(ABC):\n        @classmethod\n        @abstractmethod\n        def f(cls: Type[T]) -> T:\n          return cls()\n    ")

    def test_split(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import AnyStr, Generic\n      class Foo(Generic[AnyStr]):\n        def __init__(self, x: AnyStr):\n          if isinstance(x, str):\n            self.x = x\n          else:\n            self.x = x.decode('utf-8')\n    ")

    def test_typevar_in_variable_annotation(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import TypeVar\n      T = TypeVar('T')\n      def f(x: T):\n        y: T = x\n    ")

    def test_none_constraint(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing import TypeVar\n      T = TypeVar(\'T\', float, None)\n      def f(x: T) -> T:\n        return x\n      f(0.0)\n      f(None)\n      f("oops")  # wrong-arg-types\n    ')
if __name__ == '__main__':
    test_base.main()