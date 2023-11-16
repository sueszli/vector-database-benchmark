"""Tests for handling GenericType."""
from pytype.tests import test_base
from pytype.tests import test_utils

class GenericTest(test_base.BaseTest):
    """Tests for GenericType."""

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]): pass\n        def f() -> A[int]: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> a.A[int]: ...\n      ')

    def test_binop(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]): pass\n      ')
            ty = self.Infer('\n        from a import A\n        def f():\n          return A() + [42]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List, Type\n        A = ...  # type: Type[a.A]\n        def f() -> List[int]: ...\n      ')

    def test_specialized(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class A(Dict[K, V]): pass\n        class B(A[str, int]): pass\n      ')
            ty = self.Infer('\n        import a\n        def foo():\n          return a.B()\n        def bar():\n          x = foo()\n          return {list(x.keys())[0]: list(x.values())[0]}\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def foo() -> a.B: ...\n        def bar() -> dict[str, int]: ...\n      ')

    def test_specialized_mutation(self):
        if False:
            return 10
        with test_utils.Tempdir() as d1:
            with test_utils.Tempdir() as d2:
                d1.create_file('a.pyi', '\n          from typing import List, TypeVar\n          T = TypeVar("T")\n          class A(List[T]): pass\n        ')
                d2.create_file('b.pyi', '\n          import a\n          class B(a.A[int]): pass\n        ')
                ty = self.Infer('\n          import b\n          def foo():\n            x = b.B()\n            x.extend(["str"])\n            return x\n          def bar():\n            return foo()[0]\n        ', pythonpath=[d1.path, d2.path])
                self.assertTypesMatchPytd(ty, '\n          import b\n          from typing import Union\n          def foo() -> b.B: ...\n          def bar() -> Union[int, str]: ...\n        ')

    def test_specialized_partial(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, TypeVar\n        V = TypeVar("V")\n        class A(Dict[str, V]): pass\n        class B(A[int]): pass\n      ')
            ty = self.Infer('\n        import a\n        def foo():\n          return a.A()\n        def bar():\n          return list(foo().keys())\n        def baz():\n          return a.B()\n        def qux():\n          return list(baz().items())\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List, Tuple\n        import a\n        def foo() -> a.A[nothing]: ...\n        def bar() -> List[str]: ...\n        def baz() -> a.B: ...\n        def qux() -> List[Tuple[str, int]]: ...\n      ')

    def test_type_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]):\n          def bar(self) -> T: ...\n        class B(A[int]): ...\n      ')
            ty = self.Infer('\n        import foo\n        def f():\n          return foo.B().bar()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> int: ...\n      ')

    def test_type_parameter_renaming(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        U = TypeVar("U")\n        class A(List[U]): pass\n        class B(A[int]): pass\n      ')
            ty = self.Infer('\n        import a\n        def foo():\n          return a.A()\n        def bar():\n          return a.B()[0]\n        def baz():\n          x = a.B()\n          x.extend(["str"])\n          return x[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Union\n        import a\n        def foo() -> a.A[nothing]: ...\n        def bar() -> int: ...\n        def baz() -> Union[int, str]: ...\n      ')

    def test_type_parameter_renaming_chain(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, Set, TypeVar, Union\n        A = TypeVar("A")\n        B = TypeVar("B")\n        class Foo(List[A]):\n          def foo(self) -> None:\n            self = Foo[Union[A, complex]]\n        class Bar(Foo[B], Set[B]):\n          def bar(self) -> B: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.Bar([42])\n          x.foo()\n          x.extend(["str"])\n          x.add(float(3))\n          return x.bar()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Union\n        import a\n        def f() -> Union[int, float, complex, str]: ...\n      ')

    def test_type_parameter_renaming_conflict1(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, Tuple, TypeVar\n        T1 = TypeVar("T1")\n        T2 = TypeVar("T2")\n        T3 = TypeVar("T3")\n        class A(Generic[T1]):\n          def f(self) -> T1: ...\n        class B(Generic[T1]):\n          def g(self) -> T1: ...\n        class C(A[T2], B[T3]):\n          def __init__(self):\n            self = C[int, str]\n          def h(self) -> Tuple[T2, T3]: ...\n      ')
            ty = self.Infer('\n        import a\n        v1 = a.C().f()\n        v2 = a.C().g()\n        v3 = a.C().h()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any, Tuple\n        import a\n        v1 = ...  # type: int\n        v2 = ...  # type: str\n        v3 = ...  # type: Tuple[int, str]\n      ')

    def test_type_parameter_renaming_conflict2(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T1 = TypeVar("T1")\n        T2 = TypeVar("T2")\n        T3 = TypeVar("T3")\n        class A(Generic[T1]):\n          def f(self) -> T1: ...\n        class B(Generic[T2]):\n          def g(self) -> T2: ...\n        class C(A[T3], B[T3]):\n          def __init__(self):\n            self = C[str]\n      ')
            ty = self.Infer('\n        import a\n        v = a.C().f()\n        w = a.C().g()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        v = ...  # type: str\n        w = ...  # type: str\n      ')

    def test_change_multiply_renamed_type_parameter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T1 = TypeVar("T1")\n        T2 = TypeVar("T2")\n        T3 = TypeVar("T3")\n        class A(Generic[T1]):\n          def f(self):\n            self = A[str]\n        class B(Generic[T1]): ...\n        class C(A[T2], B[T3]):\n          def g(self):\n            self= C[int, float]\n      ')
            ty = self.Infer('\n        import a\n        v = a.C()\n        v.f()\n        v.g()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        # T1, T2, and T3 are all set to Any due to T1 being an alias for both\n        # T2 and T3.\n        v: a.C[int, float]\n      ')

    def test_type_parameter_deep(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        U = TypeVar("U")\n        V = TypeVar("V")\n        class A(Generic[U]):\n          def bar(self) -> U: ...\n        class B(A[V], Generic[V]): ...\n        def baz() -> B[int]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f():\n          return foo.baz().bar()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> int: ...\n      ')

    def test_type_parameter_import(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d1:
            d1.create_file('a.pyi', '\n        T = TypeVar("T")\n      ')
            with test_utils.Tempdir() as d2:
                d2.create_file('b.pyi', '\n          from typing import Generic, Union\n          from a import T\n          class A(Generic[T]):\n            def __init__(self, x: T) -> None:\n              self = A[Union[int, T]]\n            def a(self) -> T: ...\n        ')
                ty = self.Infer('\n          import b\n          def f():\n            return b.A("hello world")\n          def g():\n            return b.A(3.14).a()\n        ', pythonpath=[d1.path, d2.path])
                self.assertTypesMatchPytd(ty, '\n          import b\n          from typing import Union\n          def f() -> b.A[Union[int, str]]: ...\n          def g() -> Union[int, float]: ...\n        ')

    def test_type_parameter_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class MyIterable(Generic[T]): pass\n        class MyList(MyIterable[T]): pass\n        class MyDict(MyIterable[K], Generic[K, V]): pass\n        class Custom(MyDict[K, V], MyList[V]): pass\n      ')
            ty = self.Infer('\n        import a\n        x = a.Custom()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: a.Custom[nothing, nothing]\n      ')

    def test_type_parameter_ambiguous(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List\n        T = TypeVar("T")\n        class A(Generic[T]): pass\n        class B(A[int]): pass\n        class C(List[T], B): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.C()\n          return x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> a.C[nothing]: ...\n      ')

    def test_type_parameter_duplicated(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, Dict\n        T = TypeVar("T")\n        class A(Dict[T, T], Generic[T]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.A()\n          x[1] = 2\n          return x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n\n        import a\n        def f() -> a.A[int]: ...\n      ')

    def test_union(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, Union\n        class A(List[Union[int, str]]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.A()\n        def g():\n          return f()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Union\n        import a\n        def f() -> a.A: ...\n        def g() -> Union[int, str]: ...\n      ')

    def test_multiple_templates(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class MyDict(Generic[K, V]): pass\n        class A(MyDict[K, V], List[V]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.A()\n          x.extend([42])\n          return x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> a.A[nothing, int]: ...\n      ')

    def test_multiple_templates_flipped(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, Generic, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class MyList(Generic[V]):\n          def __getitem__(self, x: int) -> V: ...\n        class A(MyList[V], Dict[K, V]):\n          def a(self) -> K: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.A()\n          x.update({"hello": 0})\n          return x\n        def g():\n          return f().a()\n        def h():\n          return f()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> a.A[str, int]: ...\n        def g() -> str: ...\n        def h() -> int: ...\n      ')

    def test_type_parameter_empty(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]):\n          def f(self) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.A().f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List\n        import a\n        def f() -> List[nothing]: ...\n      ')

    @test_base.skip('Needs better GenericType support')
    def test_type_parameter_limits(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import AnyStr, Generic\n        class A(Generic[AnyStr]):\n          def f(self) -> AnyStr: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.A().f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Union\n        import a\n        def f() -> Union[str, unicode]: ...\n      ')

    def test_prevent_infinite_loop_on_type_param_collision(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class Foo(List[T]): pass\n      ')
            self.assertNoCrash(self.Check, '\n        import a\n        def f():\n          x = a.Foo()\n          x.append(42)\n          return x\n        g = lambda y: y+1\n      ', pythonpath=[d.path])

    def test_template_construction(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, Generic, List, TypeVar\n        T = TypeVar("T")\n        U = TypeVar("U")\n        class A(Dict[T, U], List[T], Generic[T, U]):\n          def f(self) -> None:\n            self = A[int, str]\n          def g(self) -> T: ...\n          def h(self) -> U: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          x = a.A()\n          x.f()\n          return x\n        def g():\n          return f().g()\n        def h():\n          return f().h()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        # T was made unsolvable by an AliasingDictConflictError.\n        def f() -> a.A[int, str]: ...\n        def g() -> int: ...\n        def h() -> str: ...\n      ')

    def test_aliasing_dict_conflict_error(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, Generic, List, TypeVar\n        T = TypeVar("T")\n        U = TypeVar("U")\n        class A(Dict[T, U], List[T], Generic[T, U]): ...\n      ')
            ty = self.Infer('\n        import a\n        v = a.A()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        v = ...  # type: a.A[nothing, nothing]\n      ')

    def test_recursive_container(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List\n        class A(List[A]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.A()[0]\n        def g():\n          return a.A()[0][0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> a.A: ...\n        def g() -> a.A: ...\n      ')

    def test_pytd_subclass(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]):\n          def __init__(self) -> None:\n            self = A[str]\n          def f(self) -> T: ...\n        class B(A): pass\n      ')
            ty = self.Infer('\n        import a\n        def foo():\n          return a.B().f()\n        def bar():\n          return a.B()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def foo() -> str: ...\n        def bar() -> str: ...\n      ')

    def test_interpreter_subclass(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]):\n          def __init__(self) -> None:\n            self = A[str]\n          def f(self) -> T: ...\n      ')
            ty = self.Infer('\n        import a\n        class B(a.A): pass\n        def foo():\n          return B().f()\n        def bar():\n          return B()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        class B(a.A): pass\n        def foo() -> str: ...\n        def bar() -> str: ...\n      ')

    def test_instance_attribute(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict, TypeVar\n        T1 = TypeVar("T1", int, float)\n        T2 = TypeVar("T2", bound=complex)\n        class A(Dict[T1, T2]):\n          x: T1\n          y: T2\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          v = a.A()\n          return (v.x, v.y)\n        def g():\n          v = a.A({0: 4.2})\n          return (v.x, v.y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Tuple, Union\n        def f() -> Tuple[Union[int, float], complex]: ...\n        def g() -> Tuple[int, float]: ...\n      ')

    def test_instance_attribute_visible(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class MyPattern(Generic[T]):\n          pattern = ...  # type: T\n          def __init__(self, x: T):\n            self = MyPattern[T]\n      ')
            ty = self.Infer('\n        import a\n        RE = a.MyPattern("")\n        def f(x):\n          if x:\n            raise ValueError(RE.pattern)\n        def g():\n          return RE.pattern\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        RE = ...  # type: a.MyPattern[str]\n        def f(x) -> None: ...\n        def g() -> str: ...\n      ')

    def test_instance_attribute_change(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        N = TypeVar("N")\n        class A(Generic[T]):\n          x = ...  # type: T\n          def f(self, x: N) -> None:\n            self = A[N]\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          inst = a.A()\n          inst.f(0)\n          inst.x\n          inst.f("")\n          return inst.x\n        def g():\n          inst = a.A()\n          inst.f(0)\n          inst.x = True\n          inst.f("")\n          return inst.x\n        def h():\n          inst = a.A()\n          inst.f(0)\n          x = inst.x\n          inst.f("")\n          return x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> str: ...\n        def g() -> bool: ...\n        def h() -> int: ...\n      ')

    def test_instance_attribute_inherited(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T", int, float)\n        class A(List[T]):\n          x = ...  # type: T\n      ')
            ty = self.Infer('\n        from typing import TypeVar\n        import a\n        T = TypeVar("T")\n        class B(a.A[T]): pass\n        def f():\n          return B().x\n        def g():\n          return B([42]).x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any, TypeVar, Union\n        import a\n        T = TypeVar("T")\n        class B(a.A[T]):\n          x = ...  # type: Union[int, float]\n        def f() -> Union[int, float]: ...\n        def g() -> int: ...\n      ')

    def test_instance_attribute_set(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]):\n          def f(self) -> T: ...\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          inst = a.A()\n          inst.x = inst.f()\n          return inst.x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        def f() -> Any: ...\n      ')

    def test_instance_attribute_conditional(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]):\n          x = ...  # type: T\n      ')
            ty = self.Infer('\n        import a\n        def f(x):\n          inst = a.A([4.2])\n          if x:\n            inst.x = 42\n          return inst.x\n        def g(x):\n          inst = a.A([4.2])\n          if x:\n            inst.x = 42\n          else:\n            return inst.x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Optional, Union\n        import a\n        def f(x) -> Union[int, float]: ...\n        def g(x) -> Optional[float]: ...\n      ')

    def test_instance_attribute_method(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        class A(List[T]):\n          x = ...  # type: T\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return abs(a.A([42]).x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def f() -> int: ...\n      ')

    def test_inherited_type_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A1(Generic[T]):\n          def f(self) -> T: ...\n        class A2(A1): pass\n      ')
            ty = self.Infer('\n        import a\n        def f(x):\n          return x.f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        def f(x) -> Any: ...\n      ')

    def test_attribute_on_anything_type_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can access an attribute on "Any".'
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any, List\n        class A(List[Any]): pass\n      ')
            ty = self.Infer('\n        import a\n        def f():\n          return a.A()[0].someproperty\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        def f() -> Any: ...\n      ')

    def test_match_anything_type_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can match "Any" against a formal function argument.'
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any, List\n        class A(List[Any]): pass\n      ')
            ty = self.Infer('\n        import a\n        n = len(a.A()[0])\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        n = ...  # type: int\n      ')

    def test_renamed_type_parameter_match(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Iterable, TypeVar\n        Q = TypeVar("Q")\n        def f(x: Iterable[Q]) -> Q: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.f({True: "false"})\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: bool\n      ')

    def test_type_parameter_union(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar, Union\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class Foo(List[Union[K, V]]):\n          def __init__(self):\n            self = Foo[int, str]\n      ')
            ty = self.Infer('\n        import foo\n        v = list(foo.Foo())\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        v = ...  # type: list[Union[int, str]]\n      ')

    def test_type_parameter_subclass(self):
        if False:
            while True:
                i = 10
        'Test subclassing A[T] with T undefined and a type that depends on T.'
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List\n        T = TypeVar("T")\n        class A(Generic[T]):\n          data = ...  # type: List[T]\n      ')
            ty = self.Infer('\n        import a\n        class B(a.A):\n          def foo(self):\n            return self.data\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        class B(a.A):\n          data = ...  # type: list\n          def foo(self) -> list: ...\n      ')

    def test_constrained_type_parameter_subclass(self):
        if False:
            return 10
        'Test subclassing A[T] with T undefined and a type that depends on T.'
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List\n        T = TypeVar("T", int, str)\n        class A(Generic[T]):\n          data = ...  # type: List[T]\n      ')
            ty = self.Infer('\n        import a\n        class B(a.A):\n          def foo(self):\n            return self.data\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List, Union\n        import a\n        class B(a.A):\n          data = ...  # type: List[Union[int, str]]\n          def foo(self) -> List[Union[int, str]]: ...\n      ')

    def test_bounded_type_parameter_subclass(self):
        if False:
            while True:
                i = 10
        'Test subclassing A[T] with T undefined and a type that depends on T.'
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, List\n        T = TypeVar("T", bound=complex)\n        class A(List[T], Generic[T]):\n          data = ...  # type: List[T]\n      ')
            ty = self.Infer('\n        import a\n        class B(a.A):\n          def foo(self):\n            return self.data\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List\n        import a\n        class B(a.A):\n          data = ...  # type: List[complex]\n          def foo(self) -> List[complex]: ...\n      ')

    def test_constrained_type_parameter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T", int, float)\n        class A(Generic[T]):\n          v = ...  # type: T\n        def make_A() -> A: ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.make_A().v\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        v = ...  # type: Union[int, float]\n      ')

    def test_bounded_type_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T", bound=float)\n        class A(Generic[T]):\n          v = ...  # type: T\n        def make_A() -> A: ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.make_A().v\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        v = ...  # type: float\n      ')

    def test_mutate_call(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        _T = TypeVar("_T")\n        class A(Generic[_T]):\n          def to_str(self):\n            self = A[str]\n          def to_int(self):\n            self = A[int]\n      ')
            ty = self.Infer('\n        import foo\n        a = foo.A()\n        a.to_str()\n        a.to_int()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        a = ...  # type: foo.A[int]\n      ')

    def test_override_inherited_method(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class Base(Generic[T]):\n          def __init__(self, x: T) -> None: ...\n      ')
            ty = self.Infer('\n        import a\n        class Derived(a.Base):\n          def __init__(self):\n            pass\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import a\n        class Derived(a.Base):\n          def __init__(self) -> None: ...\n      ')

    def test_bad_mro(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Iterator, TypeVar\n      T = TypeVar('T')\n      U = TypeVar('U')\n      V = TypeVar('V')\n\n      class A(Iterator[T]):\n        pass\n      class B(Iterator[U], A[V]):   # mro-error\n        pass\n    ")

    def test_generic_class_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def copy(self):\n          cls = self.__class__\n          return cls()\n    ")

    @test_base.skip('b/169446275: TypeVar currently checks for any common parent')
    def test_generic_classes_enforce_types(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Generic, TypeVar, Union\n      _T = TypeVar("_T")\n      class Clz(Generic[_T]):\n        def set(self, val: _T): ...\n\n      class Base: ...\n\n      class SubA(Base): ...\n      class SubB(Base): ...\n\n      clz_a: Clz[SubA]\n      # TODO(b/169446275): remove this note and test_base.skip() once fixed.\n      clz_a.set(SubB())  # wrong-arg-types\n      # Safety check (this already works): only common superclass is \'object\'.\n      clz_a.set(123)  # wrong-arg-types\n\n      # Regression test: subclasses should be allowed.\n      clz_base: Clz[Base]\n      clz_base.set(SubB())\n      # Regression test: Unions should allow all members.\n      clz_union: Clz[Union[SubA, SubB]]\n      clz_union.set(SubA())\n      clz_union.set(SubB())\n      # But still prevent incorrect types, including parents.\n      clz_union.set(123)  # wrong-arg-types\n      # TODO(b/169446275): remove this note and test_base.skip() once fixed.\n      clz_union.set(Base())  # wrong-arg-types\n    ')
if __name__ == '__main__':
    test_base.main()