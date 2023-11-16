"""Tests for overriding."""
from pytype.tests import test_base

class OverridingTest(test_base.BaseTest):
    """Tests for overridden and overriding methods signature match."""

    def test_positional_or_keyword_match(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def f(self, a: int, b: str) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, b: str = "", c: int = 1, *, d: int = 2) -> None:\n          pass\n    ')

    def test_positional_or_keyword_underscore_match(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def f(self, a: int, _: str) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, _: int, b: str) -> None:\n          pass\n    ')

    def test_positional_or_keyword_name_mismatch(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, b: int) -> None:\n          pass\n    ')

    def test_positional_or_keyword_name_and_type_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def f(self, a: int, b: str) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, b: int, c: int) -> None:\n          pass\n    ')

    def test_positional_or_keyword_name_and_count_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int, b: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, b: int) -> None:  # signature-mismatch\n          pass\n\n      class Baz(Foo):\n        def f(self, b: int, c:int, d: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_positional_or_keyword_to_keyword_only_mismatch(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, a: int) -> None:  # signature-mismatch[e]\n          pass\n    ')
        self.assertErrorSequences(errors, {'e': ['Overriding method signature mismatch', 'Base signature: ', 'Subclass signature: ', 'Not enough positional parameters in overriding method.']})

    def test_keyword_only_match(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def f(self, *, a: int, b: int, c: int = 0) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, *, b: int, c: int = 0, d: int = 1) -> None:\n          pass\n    ')

    def test_keyword_only_name_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, b: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_keyword_only_name_mismatch_twice(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, a: int) -> None:\n          pass\n\n        def g(self, *, c: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, b: int) -> None:  # signature-mismatch\n          pass\n\n        def g(self, *, d: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_keyword_only_count_mismatch(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, a: int, b: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, a: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_default_to_non_default_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int = 0) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_default_to_default_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def f(self, a: int = 0, *, c: int = 2) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int = 0, b: int = 1, * , c: int = 2, d: int = 3) -> None:\n          pass\n    ')

    def test_keyword_default_value_mismatch(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def f(self, *, t: int = 0) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, t: int = 1) -> None:  # signature-mismatch[e]\n          pass\n    ')
        self.assertErrorSequences(errors, {'e': ['t: int = 0', 't: int = 1']})

    def test_default_value_imported_class(self):
        if False:
            return 10
        with self.DepTree([('foo.py', '\n      class Foo:\n        def f(self, x: int = 0):\n          pass\n    ')]):
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          def f(self, x: int = 0):\n            pass\n      ')

    def test_partial_annotations(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def f(self, t, g: int) -> str:\n          return ""\n\n      class Bar(Foo):\n        def f(self, t: int, g: int):\n          pass\n    ')

    def test_parameter_type_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, t: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: str) -> None:  # signature-mismatch\n          pass\n    ')

    def test_return_type_mismatch(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      class Foo:\n        def f(self) -> int:\n          return 0\n\n      class Bar(Foo):\n        def f(self) -> str:  # signature-mismatch\n          return ''\n    ")

    def test_none_return_type_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      class Foo:\n        def f(self) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self) -> str:  # signature-mismatch\n          return ''\n    ")

    def test_return_type_matches_empty(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.py', '\n      class Foo:\n        def f(self):\n          raise NotImplementedError()\n    ')]):
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          def f(self) -> None:\n            pass\n      ')

    def test_pytdclass_signature_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo(list):\n        def clear(self) -> None:\n          pass\n    ')

    def test_pytdclass_parameter_type_mismatch(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Foo(list):\n        def clear(self, x: int) -> None:  # signature-mismatch[e]\n          pass\n    ')
        self.assertErrorSequences(errors, {'e': ['list.clear(self)']})

    def test_pytdclass_return_type_mismatch(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class Foo(list):\n        def clear(self) -> str:  # signature-mismatch\n          return ""\n    ')

    def test_pytdclass_default_value_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import unittest\n\n      class A(unittest.case.TestCase):\n        def assertDictEqual(self, d1, d2, msg=None):\n          pass\n    ')

    def test_pytdclass_default_value_mismatch(self):
        if False:
            return 10
        self.Check('\n      import unittest\n\n      class A(unittest.case.TestCase):\n        def assertDictEqual(self, d1, d2, msg=""):\n          pass\n    ')

    def test_subclass_subclass_signature_match(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def f(self, t: int) -> None:\n          pass\n\n      class Bar(Foo):\n        pass\n\n      class Baz(Bar):\n        def f(self, t: int) -> None:\n          pass\n  ')

    def test_subclass_subclass_parameter_type_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, t: int) -> None:\n          pass\n\n      class Bar(Foo):\n        pass\n\n      class Baz(Bar):\n        def f(self, t: str) -> None:  # signature-mismatch\n          pass\n  ')

    def test_keyword_type_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, t: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, t: str) -> None:  # signature-mismatch\n          pass\n  ')

    def test_keyword_to_positional_type_mismatch(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, t: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: str) -> None:  # signature-mismatch\n          pass\n  ')

    def test_subclass_parameter_type_match(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t: B) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: A) -> None:\n          pass\n    ')

    def test_subclass_parameter_type_mismatch(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t: A) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: B) -> None:  # signature-mismatch\n          pass\n    ')

    def test_subclass_return_type_match(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t) -> A:\n          return A()\n\n      class Bar(Foo):\n        def f(self, t) -> B:\n          return B()\n    ')

    def test_subclass_return_type_mismatch(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t) -> B:\n          return B()\n\n      class Bar(Foo):\n        def f(self, t) -> A:  # signature-mismatch\n          return A()\n    ')

    def test_multiple_inheritance_parameter_type_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class C(A):\n        pass\n\n      class Foo:\n        def f(self, t: B) -> None:\n          pass\n\n      class Bar:\n        def f(self, t: C) -> None:\n          pass\n\n      class Baz(Foo, Bar):\n        def f(self, t: A) -> None:\n          pass\n    ')

    def test_multiple_inheritance_parameter_type_mismatch(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class C(B):\n        pass\n\n      class Foo:\n        def f(self, t: A) -> None:\n          pass\n\n      class Bar:\n        def f(self, t: C) -> None:\n          pass\n\n      class Baz(Foo, Bar):\n        def f(self, t: B) -> None:  # signature-mismatch\n          pass\n    ')

    def test_multiple_inheritance_return_type_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        pass\n\n      class B:\n        pass\n\n      class C(A, B):\n        pass\n\n      class Foo:\n        def f(self, t) -> A:\n          return A()\n\n      class Bar:\n        def f(self, t) -> B:\n          return B()\n\n      class Baz(Foo, Bar):\n        def f(self, t) -> C:\n          return C()\n    ')

    def test_multiple_inheritance_return_type_mismatch(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class C(B):\n        pass\n\n      class Foo:\n        def f(self, t) -> A:\n          return C()\n\n      class Bar:\n        def f(self, t) -> C:\n          return C()\n\n      class Baz(Foo, Bar):\n        def f(self, t) -> B:  # signature-mismatch\n          return C()\n    ')

    def test_multiple_inheritance_base_parameter_type_mismatch(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        pass\n\n      class Baz:\n        def f(self, a: int, b: int) -> None:\n          pass\n\n      class Qux(Bar, Baz):  # signature-mismatch\n        pass\n    ')

    def test_generic_type_match(self):
        if False:
            return 10
        self.Check('\n      from typing import Callable, Sequence\n\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t: Callable[[A], B]) -> Sequence[Callable[[B], A]]:\n          return []\n\n      class Bar(Foo):\n        def f(self, t: Callable[[B], A]) -> Sequence[Callable[[A], B]]:\n          return []\n    ')

    def test_covariant_generic_parameter_type_mismatch(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Sequence, Iterable\n\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t: Iterable[A]) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: Iterable[B]) -> None:  # signature-mismatch\n          pass\n    ')

    def test_contravariant_generic_parameter_type_mismatch(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Callable\n\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t: Callable[[B], None]) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, t: Callable[[A], None]) -> None:  # signature-mismatch\n          pass\n    ')

    def test_covariant_generic_return_type_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import Sequence\n\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class Foo:\n        def f(self, t) -> Sequence[B]:\n          return [B()]\n\n      class Bar(Foo):\n        def f(self, t) -> Sequence[A]:  # signature-mismatch\n          return [A()]\n    ')

    def test_subclass_of_generic_for_builtin_types(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n\n      class A(Generic[T]):\n        def f(self, t: T) -> None:\n          pass\n\n        def g(self, t: int) -> None:\n          pass\n\n      class B(A[int]):\n        def f(self, t: str) -> None:  # signature-mismatch\n          pass\n\n        def g(self, t: str) -> None:  # signature-mismatch\n          pass\n\n      class C(A[list]):\n        def f(self, t: list) -> None:\n          pass\n\n        def g(self, t: int) -> None:\n          pass\n    ")

    def test_subclass_of_generic_for_simple_types(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T, U]):\n        def f(self, t: T) -> U:\n          pass\n\n      class Y:\n        pass\n\n      class X(Y):\n        pass\n\n      class B(A[X, Y]):\n        def f(self, t: X) -> Y:\n          return Y()\n\n      class C(A[X, Y]):\n        def f(self, t: Y) -> X:\n          return X()\n\n      class D(A[Y, X]):\n        def f(self, t: X) -> X:  # signature-mismatch\n          return X()\n\n      class E(A[Y, X]):\n        def f(self, t: Y) -> Y:  # signature-mismatch\n          return Y()\n    ")

    def test_subclass_of_generic_for_bound_types(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n\n      class X:\n        pass\n\n      T = TypeVar('T', bound=X)\n\n      class A(Generic[T]):\n        def f(self, t: T) -> T:\n          return T()\n\n      class Y(X):\n        pass\n\n      class B(A[Y]):\n        def f(self, t: Y) -> Y:\n          return Y()\n\n      class C(A[Y]):\n        def f(self, t: X) -> Y:\n          return Y()\n\n      class D(A[Y]):\n        def f(self, t: Y) -> X:  # signature-mismatch\n          return X()\n    ")

    def test_subclass_of_generic_match_for_generic_types(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Generic, List, Sequence, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T, U]):\n        def f(self, t: List[T]) -> Sequence[U]:\n          return []\n\n      class X:\n        pass\n\n      class Y:\n        pass\n\n      class B(A[X, Y]):\n        def f(self, t: Sequence[X]) -> List[Y]:\n          return []\n\n      class Z(X):\n        pass\n\n      class C(A[Z, X]):\n        def f(self, t: List[X]) -> Sequence[Z]:\n          return []\n    ")

    def test_subclass_of_generic_mismatch_for_generic_types(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Generic, List, Sequence, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T, U]):\n        def f(self, t: Sequence[T]) -> List[U]:\n          return []\n\n      class X:\n        pass\n\n      class Y:\n        pass\n\n      class B(A[X, Y]):\n        def f(self, t: List[X]) -> List[Y]:  # signature-mismatch\n          return []\n\n      class C(A[X, Y]):\n        def f(self, t: Sequence[X]) -> Sequence[Y]:  # signature-mismatch\n          return []\n\n      class Z(X):\n        pass\n\n      class D(A[X, Z]):\n        def f(self, t: Sequence[Z]) -> List[Z]:  # signature-mismatch\n          return []\n\n      class E(A[X, Z]):\n        def f(self, t: Sequence[X]) -> List[X]:  # signature-mismatch\n          return []\n    ")

    def test_nested_generic_types(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Callable, Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')\n\n      class A(Generic[T, U]):\n        def f(self, t: Callable[[T], U]) -> None:\n          pass\n\n      class Super:\n        pass\n\n      class Sub(Super):\n        pass\n\n      class B(A[Sub, Super]):\n        def f(self, t: Callable[[Sub], Super]) -> None:\n          pass\n\n      class C(A[Sub, Super]):\n        def f(self, t: Callable[[Super], Super]) -> None:  # signature-mismatch\n          pass\n\n      class D(A[Sub, Super]):\n        def f(self, t: Callable[[Sub], Sub]) -> None:  # signature-mismatch\n          pass\n    ")

    def test_nested_generic_types2(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Callable, Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')  # not in the class template\n\n      class A(Generic[T, U]):\n        def f(self, t: Callable[[T, Callable[[T], V]], U]) -> None:\n          pass\n\n      class Super:\n        pass\n\n      class Sub(Super):\n        pass\n\n      class B(Generic[T], A[Sub, T]):\n        pass\n\n      class C(B[Super]):\n        def f(self, t: Callable[[Sub, Callable[[Sub], V]], Super]) -> None:\n          pass\n\n      class D(B[Super]):\n        def f(self, t: Callable[[Sub, Callable[[Super], Sub]], Super]) -> None:\n          pass\n\n      class E(B[Super]):\n        def f(self, t: Callable[[Super, Callable[[Sub], V]], Super]) -> None:  # signature-mismatch\n          pass\n\n      class F(Generic[T], B[T]):\n        def f(self, t: Callable[[Sub, Callable[[Sub], V]], T]) -> None:\n          pass\n\n      class G(Generic[T], B[T]):\n        def f(self, t: Callable[[Sub, Callable[[Super], Super]], T]) -> None:\n          pass\n\n      class H(Generic[T], B[T]):\n        def f(self, t: Callable[[Super, Callable[[Sub], V]], T]) -> None:  # signature-mismatch\n          pass\n    ")

    def test_subclass_of_generic_for_renamed_type_parameters(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T]):\n        def f(self, t: T) -> None:\n          pass\n\n      class B(Generic[U], A[U]):\n        pass\n\n      class X:\n        pass\n\n      class C(B[X]):\n        def f(self, t: X) -> None:\n          pass\n    ")

    def test_subclass_of_generic_for_renamed_type_parameters2(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T, U]):\n        def f(self, t: T) -> U:\n          return U()\n\n      class X:\n        pass\n\n      class B(Generic[T], A[X, T]):\n        pass\n\n      class Y:\n        pass\n\n      class C(B[Y]):\n        def f(self, t: X) -> Y:\n          return Y()\n\n      class D(B[Y]):\n        def f(self, t: X) -> X:  # signature-mismatch\n          return X()\n    ")

    def test_subclass_of_generic_for_generic_method(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T]):\n        def f(self, t: T, u: U) -> U:\n          return U()\n\n      class Y:\n        pass\n\n      class X(Y):\n        pass\n\n      class B(A[X]):\n        def f(self, t: X, u: U) -> U:\n          return U()\n\n      class C(A[X]):\n        def f(self, t: Y, u: U) -> U:\n          return U()\n\n      class D(A[Y]):\n        def f(self, t: X, u: U) -> U:  # signature-mismatch\n          return U()\n    ")

    def test_varargs_match(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def f(self, a: int, b: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, *args: int) -> None:\n          pass\n    ')

    def test_varargs_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int, b: str) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, *args: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_varargs_count_match(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, *args: str) -> None:\n          pass\n    ')

    def test_pytd_varargs_not_annotated(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n        class Foo:\n          def f(self, *args):\n            pass\n      ')]):
            self.Check('\n        import foo\n\n        class Bar(foo.Foo):\n          def f(self, x: int):\n            pass\n      ')

    def test_kwargs_match(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def f(self, a: int, *, b: int, c: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, **kwargs: int) -> None:\n          pass\n    ')

    def test_kwargs_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int, *, b: str) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, **kwargs: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_kwargs_count_match(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def f(self, a: int, *, b: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, *, b: int, **kwargs: str) -> None:\n          pass\n    ')

    def test_default_value_to_varargs(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def call(self, x: str, y: int = 0) -> None:\n          pass\n\n      class Bar(Foo):\n        def call(self, x, *args) -> None:\n          pass\n    ')

    def test_default_value_to_kwargs(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def call(self, x: int, *, y: int, z: int = 0) -> None:\n          pass\n\n      class Bar(Foo):\n        def call(self, x: int, **kwargs) -> None:\n          pass\n    ')

    def test_class_and_static_methods(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar:\n        @classmethod\n        def f(cls, b: str) -> None:\n          pass\n\n      class Baz:\n        @staticmethod\n        def f(c: list) -> None:\n          pass\n    ')

    def test_self_name(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(this, self: int) -> None:\n          pass\n    ')

    def test_keyword_only_double_underscore_name_mismatch(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, __a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, *, __a: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_positional_only_match(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo:\n        def f(self, a: int, b: str, c: int = 0, /) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, d: int, / , e: str, f: int = 0, g: int = 1) -> None:\n          pass\n    ')

    def test_positional_only_to_keyword_only(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int, /) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, * , a: int) -> None:  # signature-mismatch\n          pass\n    ')

    def test_positional_or_keyword_to_positional_only_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        def f(self, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, /) -> None:  # signature-mismatch\n          pass\n    ')

    def test_keyword_only_to_positional_only_mismatch(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, /) -> None:  # signature-mismatch\n          pass\n    ')

    def test_keyword_only_to_positional_only_count_mismatch(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      class Foo:\n        def f(self, *, a: int) -> None:\n          pass\n\n      class Bar(Foo):\n        def f(self, a: int, /) -> None:  # signature-mismatch\n          pass\n    ')

    def test_callable_multiple_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable\n      class Foo:\n        def __call__(self, x: int, *, y: str):\n          pass\n      class Bar(Callable, Foo):\n        pass\n    ')

    def test_async(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      class Foo:\n        async def f(self) -> int:\n          return 0\n        def g(self) -> int:\n          return 0\n    ')]):
            self.CheckWithErrors("\n        import foo\n        class Good(foo.Foo):\n          async def f(self) -> int:\n            return 0\n        class Bad(foo.Foo):\n          async def f(self) -> str:  # signature-mismatch\n            return ''\n          # Test that we catch the non-async/async mismatch even without a\n          # return annotation.\n          async def g(self):  # signature-mismatch\n            return 0\n      ")

    def test_disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo:\n        def f(self, x) -> int:\n          return 0\n      class Bar(Foo):\n        def f(  # pytype: disable=signature-mismatch\n            self, x) -> str:\n          return "0"\n      class Baz(Foo):\n        def f(\n            self, x) -> str:  # pytype: disable=signature-mismatch\n          return "0"\n      class Qux(Foo):\n        def f(\n            self,  # pytype: disable=signature-mismatch\n            x) -> str:\n          return "0"\n    ')

class TypingOverrideTest(test_base.BaseTest):
    """Tests for @typing.override."""

    def test_valid_override(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import override\n      class A:\n        def f(self):\n          pass\n      class B(A):\n        @override\n        def f(self):\n          pass\n    ')

    def test_invalid_override(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import override\n      class A:\n        def f(self):\n          pass\n      class B(A):\n        @override\n        def g(self):  # override-error[e]\n          pass\n    ')
        self.assertErrorSequences(errors, {'e': ["Attribute 'g' not found on any parent class"]})

    def test_multiple_inheritance(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing_extensions import override\n      class A:\n        def f(self):\n          pass\n      class B:\n        def g(self):\n          pass\n      class C(A, B):\n        @override\n        def f(self):\n          pass\n        @override\n        def g(self):\n          pass\n        @override\n        def h(self):  # override-error\n          pass\n    ')

    def test_nested_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing_extensions import override\n      class A:\n        class B:\n          pass\n      class C(A):\n        @override\n        class B:\n          pass\n        @override\n        class B2:  # override-error\n          pass\n    ')

    def test_strict_mode(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      # pytype: features=require-override-decorator\n      from typing_extensions import override\n      class A:\n        def f(self):\n          pass\n        def g(self):\n          pass\n      class B(A):\n        @override\n        def f(self):\n          pass\n        def g(self):  # override-error[e]\n          pass\n        def h(self):\n          pass\n    ')
        self.assertErrorSequences(errors, {'e': ["Missing @typing.override decorator for 'g', which overrides 'A.g'"]})
if __name__ == '__main__':
    test_base.main()