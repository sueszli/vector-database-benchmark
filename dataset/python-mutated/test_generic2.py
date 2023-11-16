"""Tests for handling GenericType."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class GenericBasicTest(test_base.BaseTest):
    """Tests for User-defined Generic Type."""

    def test_generic_type_params_error(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import Generic\n\n      class A(Generic[int]):  # invalid-annotation[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'Parameters.*Generic.*must.*type variables'})

    def test_mro_error(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors("\n      from typing import Generic, Iterator, Generator, TypeVar\n\n      T = TypeVar('T')\n\n      class A(Generic[T],  Iterator[T], Generator):  # mro-error\n        pass\n    ")

    def test_template_order_error(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import Generic, TypeVar\n\n      T1 = TypeVar(\'T1\')\n      S1 = TypeVar(\'S1\')\n      T2 = TypeVar(\'T2\')\n      S2 = TypeVar(\'S2\')\n      T3 = TypeVar(\'T3\')\n      S3 = TypeVar(\'S3\')\n      K1 = TypeVar(\'K1\')\n      V1 = TypeVar(\'V1\')\n      K2 = TypeVar(\'K2\')\n      V2 = TypeVar(\'V2\')\n\n      class DictA(Generic[T1, S1]): pass\n      class DictB(Generic[T2, S2]): pass\n      class DictC(Generic[T3, S3]): pass\n\n      # type parameter sequences: K2, K1, V1, V2\n      class ClassA(DictA[K1, V1], DictB[K2, V2], DictC[K2, K1]):\n        def func(self, x: K1, y: K2):\n          pass\n\n      # type parameter sequences: K1, K2, V1, V2\n      class ClassB(Generic[K1, K2, V1, V2], DictA[K1, V1],\n                   DictB[K2, V2], DictC[K2, K1]):\n        def func(self, x: K1, y: K2):\n          pass\n\n      A = ClassA[int, str, int, int]()\n      B = ClassB[int, str, int, int]()\n      A.func(5, "5") # wrong-arg-types[e1]\n      A.func("5", 5) # OK\n      B.func(5, "5") # OK\n      B.func("5", 5) # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'int.*str'})

    def test_type_erasure_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import Optional, TypeVar, Generic\n\n      T = TypeVar(\'T\', int, float)\n      S = TypeVar(\'S\')\n\n      class MyClass(Generic[T, S]):\n        def __init__(self, x: Optional[T] = None, y: Optional[S] = None):\n            pass\n\n        def fun(self, x: T, y: S):\n            pass\n\n      o1 = MyClass[str, str]()  # bad-concrete-type[e1]\n      o2 = MyClass[int, int]()\n      o2.fun("5", 5)  # wrong-arg-types[e2]\n      o2.fun(5, "5")  # wrong-arg-types[e3]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Union\\[float, int\\].*str', 'e2': 'x: int.*x: str', 'e3': 'y: int.*y: str'})

    def test_inheric_plain_generic_error(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n     from typing import Generic\n\n     class A(Generic):  # invalid-annotation[e]\n       pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'Cannot inherit.*plain Generic'})

    def test_generic_with_dup_type_error(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Generic, TypeVar\n\n        T = TypeVar('T')\n        class A(Generic[T, T]): ...\n      ")
            (_, errors) = self.InferWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'Duplicate.*T.*a.A'})

    def test_multi_generic_error(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Generic, TypeVar\n\n        T = TypeVar('T')\n        V = TypeVar('V')\n        class A(Generic[T], Generic[V]): ...\n      ")
            (_, errors) = self.InferWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'Cannot inherit.*Generic.*multiple times'})

    def test_generic_with_type_miss_error(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import Generic, TypeVar, Dict\n\n        K = TypeVar('K')\n        V = TypeVar('V')\n        class A(Dict[K, V], Generic[K]): ...\n      ")
            (_, errors) = self.InferWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'V.*are not listed in Generic.*a.A'})

    def test_class_in_func_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors("\n      from typing import TypeVar, Generic, Union\n\n      T = TypeVar('T')\n      S = TypeVar('S')\n\n      def func(x: T, y: S) -> Union[T, S]:\n        class InnerCls1(Generic[T]):  # invalid-annotation[e1]  # invalid-annotation[e2]\n          class InnerCls2(Generic[S]):\n            pass\n\n        return x + y\n    ")
        self.assertErrorRegexes(errors, {'e1': 'func.*InnerCls2.*S', 'e2': 'func.*InnerCls1.*T'})

    def test_class_in_class_error(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors("\n     from typing import Optional, TypeVar, Generic, Iterator\n\n     T = TypeVar('T', int, float, str)\n     S = TypeVar('S')\n\n     class MyClass(Generic[T, S]):  # invalid-annotation[e1]\n       def __init__(self, x: Optional[T] = None, y: Optional[S] = None):\n         pass\n\n       def f(self, x: T, y: S):\n         pass\n\n       class InnerClass1(Iterator[T]):\n         pass\n\n     class A(Generic[T]):  # invalid-annotation[e2]\n       class B(Generic[S]):\n         class C(Generic[T]):\n           pass\n    ")
        self.assertErrorRegexes(errors, {'e1': 'MyClass.*InnerClass1.*T', 'e2': 'A.*C.*T'})

    def test_signature_type_param(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors("\n      from typing import Optional, TypeVar, Generic\n\n      T = TypeVar('T', int, float, str)\n      S = TypeVar('S')\n      V = TypeVar('V')\n\n      class MyClass(Generic[T, S]):\n        def __init__(self, x: Optional[T] = None, y: Optional[S] = None):\n            pass\n\n        def func1(self, x: T, y: S): pass\n\n        def func2(self, x: V): pass  # invalid-annotation[e1]\n\n      def func1(x: S): pass  # invalid-annotation[e2]\n\n      def func2(x: S) -> S:\n        return x\n\n      def func3(x: T): pass\n    ")
        self.assertErrorRegexes(errors, {'e1': "Invalid type annotation 'V'", 'e2': "Invalid type annotation 'S'"})

    def test_pyi_output(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Optional, TypeVar, Generic\n\n      S = TypeVar('S')\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')\n\n      class MyClass(Generic[T, S]):\n        def __init__(self, x: Optional[T] = None, y: Optional[S] = None):\n            pass\n\n        def fun(self, x: T, y: S):\n            pass\n\n      x = MyClass[int, int]()\n      y = MyClass(5, 5)\n\n      class A(Generic[T, S]):\n        pass\n\n      class B(Generic[T, S]):\n        pass\n\n      class C(Generic[U, V], A[U, V], B[U, V]):\n        pass\n\n      z = C()\n\n      class D(A[V, U]):\n        pass\n\n      a = D()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, Optional, TypeVar, Union\n\n      a: D[nothing, nothing]\n      x: MyClass[int, int]\n      y: MyClass[int, int]\n      z: C[nothing, nothing]\n\n      S = TypeVar('S')\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')\n\n      class A(Generic[T, S]):\n          pass\n\n      class B(Generic[T, S]):\n          pass\n\n      class C(Generic[U, V], A[U, V], B[U, V]):\n          pass\n\n      class D(A[V, U]):\n          pass\n\n      class MyClass(Generic[T, S]):\n          def __init__(self, x: Optional[T] = ..., y: Optional[S] = ...) -> None:\n            self = MyClass[T, S]\n          def fun(self, x: T, y: S) -> None: ...\n    ")

    def test_signature_type_error(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      V = TypeVar('V')\n\n      class MyClass(Generic[T]):\n        def __init__(self, x: T, y: V):  # invalid-annotation[e]\n          pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'V.*appears only once in the function signature'})

    def test_type_parameter_without_substitution(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('base.pyi', "\n        from typing import Generic, Type, TypeVar\n\n        T = TypeVar('T')\n\n        class MyClass(Generic[T]):\n          @classmethod\n          def ProtoClass(cls) -> Type[T]: ...\n      ")
            self.Check('\n        from base import MyClass\n\n        class SubClass(MyClass):\n          def func(self):\n            self.ProtoClass()\n      ', pythonpath=[d.path])

    def test_pytd_class_instantiation(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]):\n          def get(self) -> T: ...\n          def put(self, elem: T): ...\n      ')
            ty = self.Infer('\n        import a\n        b = a.A[int]()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n\n        b = ...  # type: a.A[int]\n      ')

    def test_func_match_for_interpreter_class_error(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import TypeVar, Generic\n\n      T1 = TypeVar(\'T1\')\n      S1 = TypeVar(\'S1\')\n      T2 = TypeVar(\'T2\')\n      S2 = TypeVar(\'S2\')\n      T = TypeVar(\'T\')\n      S = TypeVar(\'S\')\n\n      class A(Generic[T1, S1]):\n        def fun1(self, x: T1, y: S1):\n            pass\n\n      class B(Generic[T2, S2]):\n        def fun2(self, x: T2, y: S2):\n            pass\n\n      class C(Generic[T, S], A[T, S], B[T, S]):\n        def fun3(self, x: T, y: S):\n            pass\n\n      o = C[int, int]()\n      o.fun1("5", "5")  # wrong-arg-types[e1]\n      o.fun2("5", "5")  # wrong-arg-types[e2]\n      o.fun3("5", "5")  # wrong-arg-types[e3]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'int.*str', 'e2': 'int.*str', 'e3': 'int.*str'})

    def test_func_match_for_pytd_class_error(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', "\n        from typing import TypeVar, Generic\n\n        T1 = TypeVar('T1')\n        S1 = TypeVar('S1')\n        T2 = TypeVar('T2')\n        S2 = TypeVar('S2')\n        T = TypeVar('T')\n        S = TypeVar('S')\n\n        class A(Generic[T1, S1]):\n          def fun1(self, x: T1, y: S1): ...\n\n        class B(Generic[T2, S2]):\n          def fun2(self, x: T2, y: S2): ...\n\n        class C(A[T, S], B[T, S], Generic[T, S]):\n          def fun3(self, x: T, y: S): ...\n      ")
            (_, errors) = self.InferWithErrors('\n        import a\n\n        o = a.C[int, int]()\n\n        o.fun1("5", "5")  # wrong-arg-types[e1]\n        o.fun2("5", "5")  # wrong-arg-types[e2]\n        o.fun3("5", "5")  # wrong-arg-types[e3]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'int.*str', 'e2': 'int.*str', 'e3': 'int.*str'})

    def test_type_renaming_error(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T', int, float)\n      V = TypeVar('V')\n      S = TypeVar('S')\n      U = TypeVar('U', bound=int)\n      W = TypeVar('W')\n\n      class A(Generic[T]): pass\n      class B(A[V]): pass  # bad-concrete-type[e1]\n\n      class C(Generic[V]): pass\n      class D(C[T]): pass\n      class E(D[S]): pass  # bad-concrete-type[e2]\n\n      class F(Generic[U]): pass\n      class G(F[W]): pass  # bad-concrete-type[e3]\n    ")
        self.assertErrorSequences(errors, {'e1': ['Expected: T', 'Actually passed: V', 'T and V have incompatible'], 'e2': ['Expected: T', 'Actually passed: S', 'T and S have incompatible'], 'e3': ['Expected: U', 'Actually passed: W', 'U and W have incompatible']})

    def test_type_parameter_conflict_error(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      V = TypeVar('V')\n      S = TypeVar('S')\n      U = TypeVar('U')\n\n      class A(Generic[T]): pass\n      class B(A[V]): pass\n\n      class D(B[S], A[U]): pass\n      class E(D[int, str]): pass  # invalid-annotation[e1]\n\n      d = D[int, str]()  # invalid-annotation[e2]\n      e = E()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Generic, TypeVar\n\n      d = ...  # type: Any\n      e = ...  # type: E\n\n      S = TypeVar('S')\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')\n\n      class A(Generic[T]):\n          pass\n\n      class B(A[V]):\n          pass\n\n      class D(B[S], A[U]):\n          pass\n\n      class E(Any):\n          pass\n     ")
        self.assertErrorRegexes(errors, {'e1': 'Conflicting value for TypeVar', 'e2': 'Conflicting value for TypeVar'})

    def test_unbound_type_parameter_error(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n      U = TypeVar('U')\n\n      class A(Generic[T]): pass\n      class B(A): pass\n      class D(B, A[U]): pass  # invalid-annotation[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Conflicting value for TypeVar D.U'})

    def test_self_type_parameter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Sequence, Typing, Generic\n\n        AT = TypeVar("AT", bound=A)\n        BT = TypeVar("BT", bound=B)\n        CT = TypeVar("CT", bound=C)\n        T = TypeVar("T")\n\n        class A(Sequence[AT]): ...\n        class B(A, Sequence[BT]): ...\n        class C(B, Sequence[CT]): ...\n\n        class D(Sequence[D]): ...\n        class E(D, Sequence[E]): ...\n        class F(E, Sequence[F]): ...\n\n        class G(Sequence[G[int]], Generic[T]): ...\n      ')
            self.Check('\n        import a\n\n        c = a.C()\n        f = a.F()\n        g = a.G[int]()\n      ', pythonpath=[d.path])

    def test_any_match_all_types(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      import collections, typing\n\n      class DictA(collections.OrderedDict, typing.MutableMapping[int, int]):\n        pass\n\n      class DictB(typing.MutableMapping[int, int]):\n        pass\n\n      class DictC(collections.OrderedDict, DictB):\n        pass\n\n      d1 = collections.OrderedDict()\n      d2 = DictA()\n      d3 = DictC()\n      x = d1["123"]\n      y = d2["123"]  # unsupported-operands[e1]\n      z = d3["123"]  # unsupported-operands[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'str.*int'})

    def test_no_self_annot(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Any, Generic, List, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, children: List['Foo[Any]']):\n          pass\n    ")

    def test_illegal_self_annot(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import Any, Generic, List, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self: 'Foo', children: List['Foo[Any]']):\n          pass  # invalid-annotation[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'self.*__init__'})

    def test_parameterized_forward_reference(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Generic, TypeVar\n      T = TypeVar(\'T\')\n\n      v = None  # type: "Foo[int]"\n\n      class Foo(Generic[T]):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      v: Foo[int]\n      class Foo(Generic[T]): ...\n    ")

    def test_bad_parameterized_forward_reference(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Generic, TypeVar\n      T = TypeVar(\'T\')\n\n      v = None  # type: "Foo[int, str]"  # invalid-annotation[e]\n\n      class Foo(Generic[T]):\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': '1.*2'})

    def test_recursive_class(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import List\n      class Foo(List["Foo"]):\n        pass\n    ')

    def test_late_annotations(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Generic, TypeVar\n\n      T = TypeVar('T')\n\n      class A(Generic[T]): ...\n      class B(Generic[T]): ...\n\n      class C(A['C']): ...\n      class D(A['B[D]']): ...\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n\n      class A(Generic[T]): ...\n      class B(Generic[T]): ...\n\n      class C(A[C]): ...\n      class D(A[B[D]]): ...\n    ")

    def test_type_parameter_count(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Generic, List, TypeVar\n\n      T = TypeVar('T')\n      SomeAlias = List[T]\n\n      class Foo(Generic[T]):\n        def __init__(self, x: T, y: SomeAlias):\n          pass\n\n      def f(x: T) -> SomeAlias:\n        return [x]\n    ")

    def test_return_type_param(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n        def f(self) -> T:\n          return self.x\n      def g():\n        return Foo(0).f()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n        def f(self) -> T: ...\n      def g() -> int: ...\n    ")

    def test_generic_function_in_generic_class(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Generic, Tuple, TypeVar\n      S = TypeVar('S')\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n        def f(self, x: S) -> Tuple[S, T]:\n          return (x, self.x)\n      def g(x):\n        return Foo(0).f('hello world')\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Generic, Tuple, TypeVar\n      S = TypeVar('S')\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n        def f(self, x: S) -> Tuple[S, T]: ...\n      def g(x) -> Tuple[str, int]: ...\n    ")

    def test_generic_abc_with_getitem(self):
        if False:
            return 10
        self.Check("\n      import abc\n      from typing import Any, Generic, Optional, Tuple, TypeVar\n\n      T = TypeVar('T')\n\n      class Filterable(Generic[T], abc.ABC):\n        @abc.abstractmethod\n        def get_filtered(self) -> T:\n          pass\n\n      class SequenceHolder(Generic[T], Filterable[Any]):\n        def __init__(self, *sequence: Optional[T]) -> None:\n          self._sequence = sequence\n\n        def __getitem__(self, key: int) -> Optional[T]:\n          return self._sequence[key]\n\n        def get_filtered(self) -> 'SequenceHolder[T]':\n          filtered_sequence = tuple(\n              item for item in self._sequence if item is not None)\n          return SequenceHolder(*filtered_sequence)\n\n      sequence_holder = SequenceHolder('Hello', None, 'World')\n    ")

    def test_check_class_param(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Generic, Tuple, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n        def f(self, x: T):\n          pass\n      foo = Foo(0)\n      foo.f(1)  # okay\n      foo.f('1')  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Expected.*int.*Actual.*str'})

    def test_instantiate_parameterized_class(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import Any, Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n      def f(x: Foo[int]):\n        return x.x\n      def g(x: Any):\n        return Foo[int](x)\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n      def f(x: Foo[int]) -> int: ...\n      def g(x: Any) -> Foo[int]: ...\n    ")

    def test_constructor_typevar_container(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Generic, List, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: List[T]):\n          self.x = x\n          self.y = x[0]\n        def f(self) -> T:\n          return self.y\n      def g():\n        return Foo([0]).f()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, List, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: List[T]\n        y: T\n        def __init__(self, x: List[T]) -> None:\n          self = Foo[T]\n        def f(self) -> T: ...\n      def g() -> int: ...\n    ")

    def test_reinherit_generic(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n      # Inheriting from Foo (unparameterized) is equivalent to inheriting from\n      # Foo[Any]. This is likely a mistake, but we should still do something\n      # reasonable.\n      class Bar(Foo, Generic[T]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n      class Bar(Foo, Generic[T]):\n        x: Any\n    ")

    def test_generic_substitution(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar\n\n        AD = TypeVar('AD', bound=AsDictable)\n        T = TypeVar('T')\n\n        class AsDictable(Protocol):\n          def _asdict(self) -> Dict[str, Any]: ...\n        class AsDictableListField(Field[List[AD]]): ...\n        class Field(Generic[T]):\n          def __call__(self) -> T: ...\n        class FieldDeclaration(Generic[T]):\n          def __call__(self) -> T: ...\n      ")
            d.create_file('bar.pyi', '\n        import foo\n        from typing import Any, Dict\n\n        BarFieldDeclaration: foo.FieldDeclaration[foo.AsDictableListField[X]]\n\n        class X:\n          def _asdict(self) -> Dict[str, Any]: ...\n      ')
            self.Check('\n        import bar\n        from typing import Sequence\n\n        def f(x: Sequence[bar.X]):\n          pass\n        def g():\n          f(bar.BarFieldDeclaration()())\n      ', pythonpath=[d.path])

    def test_subclass_typevar(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Stack(Generic[T]):\n        def peek(self) -> T:\n          return __any_object__\n      class IntStack(Stack[int]):\n        pass\n      x = IntStack().peek()\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Stack(Generic[T]):\n        def peek(self) -> T: ...\n      class IntStack(Stack[int]): ...\n      x: int\n    ")

    def test_inference_with_subclass(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T', int, str)\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n      class Bar(Foo[int]): ...\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T', int, str)\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n      class Bar(Foo[int]):\n        x: int\n    ")

    def test_rename_bounded_typevar(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Callable, Generic, TypeVar\n\n      T = TypeVar('T', bound=int)\n      No = TypeVar('No', bound=float)\n      Ok = TypeVar('Ok', bound=bool)\n\n      class Box(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n        def error(self, f: Callable[[T], No]) -> 'Box[No]':  # bad-concrete-type\n          return Box(f(self.x))  # wrong-arg-types\n        def good(self, f: Callable[[T], Ok]) -> 'Box[Ok]':\n          return Box(f(self.x))\n    ")

    def test_property(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Generic, TypeVar, Union\n      T = TypeVar('T', bound=Union[int, str])\n      class Foo(Generic[T]):\n        @property\n        def foo(self) -> T:\n          return __any_object__\n      x: Foo[int]\n      assert_type(x.foo, int)\n    ")

    def test_property_with_init_parameter(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, TypeVar, Union\n      T = TypeVar('T', bound=Union[int, str])\n      class Foo(Generic[T]):\n        def __init__(self, foo: T):\n          self._foo = foo\n        @property\n        def foo(self) -> T:\n          return self._foo\n      x = Foo(0)\n      assert_type(x.foo, int)\n    ")

    def test_property_with_inheritance(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Generic, TypeVar, Union\n      T = TypeVar('T', bound=Union[int, str])\n      class Foo(Generic[T]):\n        def __init__(self, foo: T):\n          self._foo = foo\n        @property\n        def foo(self) -> T:\n          return self._foo\n      class Bar(Foo[int]):\n        pass\n      x: Bar\n      assert_type(x.foo, int)\n    ")

    def test_pyi_property(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', "\n        from typing import Generic, TypeVar, Union\n        T = TypeVar('T', bound=Union[int, str])\n        class Foo(Generic[T]):\n          @property\n          def foo(self) -> T:\n            return __any_object__\n    ")]):
            self.Check('\n        import foo\n        x: foo.Foo[int]\n        assert_type(x.foo, int)\n      ')

    def test_pyi_property_with_inheritance(self):
        if False:
            return 10
        with self.DepTree([('foo.py', "\n      from typing import Generic, Type, TypeVar\n      T = TypeVar('T')\n      class Base(Generic[T]):\n        @property\n        def x(self) -> Type[T]:\n          return __any_object__\n      class Foo(Base[T]):\n        pass\n    ")]):
            self.Check('\n        import foo\n        def f(x: foo.Foo):\n          return x.x\n      ')

    def test_pyi_property_setter(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', "\n      from typing import Annotated, Any, Callable, Generic, TypeVar\n      ValueType = TypeVar('ValueType')\n      class Data(Generic[ValueType]):\n        value: Annotated[ValueType, 'property']\n      class Manager:\n        def get_data(\n            self, x: Callable[[ValueType], Any], y: Data[ValueType]\n        ) -> Data[ValueType]: ...\n    ")]):
            self.Check('\n        import foo\n        class Bar:\n          def __init__(self, x: foo.Manager):\n            self.data = x.get_data(__any_object__, __any_object__)\n            self.data.value = None\n      ')

    def test_parameterize_generic_with_generic(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, TypeVar, Union\n      class A: ...\n      class B: ...\n      T = TypeVar('T', bound=Union[A, B])\n      class Foo(Generic[T]): ...\n    ")]):
            self.CheckWithErrors("\n        from typing import Any, Generic, TypeVar\n        import foo\n\n        T = TypeVar('T')\n        class C(Generic[T]):\n          pass\n\n        class Bar(foo.Foo[C[Any]]):  # bad-concrete-type\n          def __init__(self):\n            pass\n          def f(self, c: C[Any]):\n            pass\n      ")

class GenericFeatureTest(test_base.BaseTest):
    """Tests for User-defined Generic Type."""

    def test_type_parameter_duplicated(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, Dict\n        T = TypeVar("T")\n        class A(Dict[T, T], Generic[T]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.A()\n          x[1] = 2\n          return x\n\n        d = None  # type: a.A[int]\n        ks, vs = d.keys(), d.values()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n\n        d = ...  # type: a.A[int]\n        ks = ...  # type: dict_keys[int]\n        vs = ...  # type: dict_values[int]\n\n        def f() -> a.A[int]: ...\n      ')

    def test_typevar_under_decorator(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import abc\n      from typing import Generic, Tuple, TypeVar\n      T = TypeVar('T')\n      class Foo(abc.ABC, Generic[T]):\n        @abc.abstractmethod\n        def parse(self) -> Tuple[T]:\n          raise NotImplementedError()\n    ")

    def test_typevar_in_class_attribute(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n      x = Foo[int]().x\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n      x: int\n    ")

    def test_bad_typevar_in_class_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      from typing import Generic, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      class Foo(Generic[T1]):\n        x: T2  # invalid-annotation[e]\n    ")
        self.assertErrorRegexes(errors, {'e': "TypeVar\\(s\\) 'T2' not in scope for class 'Foo'"})

    def test_typevar_in_instance_attribute(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x, y):\n          self.x: T = x\n          self.y = y  # type: T\n      foo = Foo[int](__any_object__, __any_object__)\n      x, y = foo.x, foo.y\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        x: T\n        y: T\n        def __init__(self, x, y) -> None: ...\n      foo: Foo[int]\n      x: int\n      y: int\n    ")

    def test_bad_typevar_in_instance_attribute(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors("\n      from typing import Generic, TypeVar\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      class Foo(Generic[T1]):\n        def __init__(self, x, y):\n          self.x: T2 = x  # invalid-annotation[e1]\n          self.y = y  # type: T2  # invalid-annotation[e2]\n    ")
        self.assertErrorRegexes(errors, {'e1': "TypeVar\\(s\\) 'T2' not in scope for class 'Foo'", 'e2': "TypeVar\\(s\\) 'T2' not in scope for class 'Foo'"})

    def test_typevar_in_classmethod(self):
        if False:
            return 10
        self.Check("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class X(Generic[T]):\n        @classmethod\n        def f(cls, x: T) -> T:\n          y: T = x\n          return y\n    ")

    def test_reingest_generic(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self, x: T):\n          self.x = x\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        x1 = foo.Foo(0).x\n        x2 = foo.Foo[str](__any_object__).x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x1: int\n        x2: str\n      ')

    def test_inherit_from_nested_generic(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo:\n        class Bar(Generic[T]):\n          pass\n        class Baz(Bar[T]):\n          pass\n      class Qux(Foo.Bar[T]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo:\n        class Bar(Generic[T]): ...\n        class Baz(Foo.Bar[T]): ...\n      class Qux(Foo.Bar[T]): ...\n    ")

    def test_mutation_to_unknown(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, TypeVar, overload\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      class A(Generic[T1, T2]):\n        @overload\n        def f(self, x: str) -> None:\n          self = A[bytes, T2]\n        @overload\n        def f(self, x: int) -> None:\n          self = A[float, T2]\n    ")]):
            self.Check('\n        import foo\n        from typing import Any\n        a = foo.A[int, int]()\n        a.f(__any_object__)\n        assert_type(a, foo.A[Any, int])\n      ')

    def test_invalid_mutation(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('_typing.pyi', '\n            from typing import Any\n            NDArray: Any\n         '), ('my_numpy.pyi', '\n            from _typing import NDArray\n            from typing import Any, Generic, TypeVar\n\n            _T1 = TypeVar("_T1")\n            _T2 = TypeVar("_T2")\n\n            class ndarray(Generic[_T1, _T2]):\n                def __getitem__(self: NDArray[Any], key: str) -> NDArray[Any]: ...\n        ')]):
            err = self.CheckWithErrors('\n        import my_numpy as np\n\n        def aggregate_on_columns(matrix: np.ndarray):\n          matrix = matrix[None, :]  # invalid-signature-mutation[e]\n      ')
            self.assertErrorSequences(err, {'e': ['ndarray.__getitem__', 'self = Any']})

    def test_class_name_prefix(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Alpha(Generic[T]):\n        def __init__(self, x: T):\n          pass\n      class Alphabet(Alpha[str]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Alpha(Generic[T]):\n        def __init__(self, x: T):\n          self = Alpha[T]\n      class Alphabet(Alpha[str]): ...\n    ")

    def test_inherit_generic_namedtuple(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import AnyStr, Generic, NamedTuple\n      class Base(NamedTuple, Generic[AnyStr]):\n        x: AnyStr\n      class Child(Base[str]):\n        pass\n      c: Child\n      assert_type(c.x, str)\n    ')

    def test_inherit_generic_namedtuple_pyi(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      from typing import AnyStr, Generic, NamedTuple\n      class Base(NamedTuple, Generic[AnyStr]):\n        x: AnyStr\n      class Child(Base[str]): ...\n    ')]):
            self.Check('\n        import foo\n        c: foo.Child\n        assert_type(c.x, str)\n      ')

    def test_generic_signature(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, TypeVar, Union\n      T = TypeVar('T', bound=Union[int, str])\n      class A(Generic[T]):\n        def f(self, x: T): ...\n    ")]):
            self.Check('\n        import foo\n        class B(foo.A[str]):\n          def f(self, x: str):\n            pass\n      ')

    def test_classmethod(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, Type, TypeVar\n      T = TypeVar('T')\n      class X(Generic[T]):\n        @classmethod\n        def f(cls) -> Type[T]:\n          return __any_object__\n      class Y(X[str]):\n        pass\n      assert_type(Y.f(), Type[str])\n      assert_type(Y().f(), Type[str])\n    ")

    def test_classmethod_pyi(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class X(Generic[T]):\n        @classmethod\n        def f(cls) -> type[T]: ...\n    ")]):
            self.Check('\n        import foo\n        from typing import Type\n        class Y(foo.X[str]):\n          pass\n        assert_type(Y.f(), Type[str])\n        assert_type(Y().f(), Type[str])\n      ')

    def test_classmethod_reingest(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.py', "\n      from typing import Generic, Type, TypeVar\n      T = TypeVar('T')\n      class X(Generic[T]):\n        @classmethod\n        def f(cls) -> Type[T]:\n          return __any_object__\n    ")]):
            self.Check('\n        import foo\n        from typing import Type\n        class Y(foo.X[str]):\n          pass\n        assert_type(Y.f(), Type[str])\n        assert_type(Y().f(), Type[str])\n      ')

    def test_annotated_cls(self):
        if False:
            return 10
        self.Check("\n      from typing import Generic, Type, TypeVar\n      T = TypeVar('T', int, str)\n      class A(Generic[T]):\n        @classmethod\n        def f(cls: Type['A[T]'], x: T) -> T:\n          return x\n      def f() -> str:\n        return A.f('')\n    ")

    @test_base.skip('TODO(b/297390011): Support this.')
    def test_annotated_cls_pyi(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', "\n       from typing import Generic, Type, TypeVar\n       T = TypeVar('T', int, str)\n       class A(Generic[T]):\n         @classmethod\n         def f(cls: Type[A[T]], x: T) -> T: ...\n     ")]):
            self.Check("\n         import foo\n         def f() -> str:\n           return foo.A.f('')\n      ")

    def test_generic_staticmethod(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Any, Callable, Generic, TypeVar, Union\n\n      T = TypeVar('T')\n\n      class Expr(Generic[T]):\n\n        def __call__(self, *args: Any, **kwargs: Any) -> T:\n          return __any_object__\n\n        @staticmethod\n        def make_unbound(\n            init: Union[Callable[..., T], 'Expr[T]'],\n        ) -> 'Expr[T]':\n          return Expr()\n\n\n      def expr_var(initial_expr: Expr[T]) -> Expr[T]:\n        return Expr.make_unbound(init=initial_expr)\n    ")

    def test_inherit_from_generic_class_with_generic_instance_method(self):
        if False:
            return 10
        ty = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Base(Generic[T]):\n        def __init__(self, x: T):\n          self.x: T = x\n      class Child(Base[bool]):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Base(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Base[T]\n      class Child(Base[bool]):\n        x: bool\n    ")
if __name__ == '__main__':
    test_base.main()