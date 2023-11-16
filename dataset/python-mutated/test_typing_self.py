"""Tests for typing.Self."""
import textwrap
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class SelfTest(test_base.BaseTest):
    """Tests for typing.Self."""

    def test_instance_method_return(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing_extensions import Self\n      class A:\n        def f(self) -> Self:\n          return self\n      class B(A):\n        pass\n      assert_type(A().f(), A)\n      assert_type(B().f(), B)\n    ')

    def test_parameterized_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import List\n      from typing_extensions import Self\n      class A:\n        def f(self) -> List[Self]:\n          return [self]\n      class B(A):\n        pass\n      assert_type(A().f(), "List[A]")\n      assert_type(B().f(), "List[B]")\n    ')

    def test_parameter(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing_extensions import Self\n      class A:\n        def f(self, other: Self) -> bool:\n          return False\n      class B(A):\n        pass\n      B().f(B())  # ok\n      B().f(0)  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Expected', 'B', 'Actual', 'int']})

    def test_nested_class(self):
        if False:
            return 10
        self.Check('\n      from typing_extensions import Self\n      class A:\n        class B:\n          def f(self) -> Self:\n            return self\n      class C(A.B):\n        pass\n      assert_type(A.B().f(), A.B)\n      assert_type(C().f(), C)\n    ')

    @test_utils.skipBeforePy((3, 11), 'typing.Self is new in 3.11')
    def test_import_from_typing(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Self\n      class A:\n        def f(self) -> Self:\n          return self\n      class B(A):\n        pass\n      assert_type(A().f(), A)\n      assert_type(B().f(), B)\n    ')

    def test_classmethod(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing_extensions import Self\n      class A:\n        @classmethod\n        def build(cls) -> Self:\n          return cls()\n      class B(A):\n        pass\n      assert_type(A.build(), A)\n      assert_type(B.build(), B)\n    ')

    def test_new(self):
        if False:
            return 10
        self.Check('\n      from typing_extensions import Self\n      class A:\n        def __new__(cls) -> Self:\n          return super().__new__(cls)\n      class B(A):\n        pass\n      assert_type(A(), A)\n      assert_type(B(), B)\n    ')

    def test_generic_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Generic, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class A(Generic[T]):\n        def copy(self) -> Self:\n          return self\n      class B(A[T]):\n        pass\n      assert_type(A[int]().copy(), A[int])\n      assert_type(B[str]().copy(), B[str])\n    ")

    def test_protocol(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      from typing import Protocol, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class MyProtocol(Protocol[T]):\n        def f(self) -> Self:\n          return self\n      class Ok1:\n        def f(self) -> MyProtocol:\n          return self\n      class Ok2:\n        def f(self) -> 'Ok2':\n          return self\n      class Ok3:\n        def f(self) -> Self:\n          return self\n      class Bad:\n        def f(self) -> int:\n          return 0\n      def f(x: MyProtocol[str]):\n        pass\n      f(Ok1())\n      f(Ok2())\n      f(Ok3())\n      f(Bad())  # wrong-arg-types\n    ")

    def test_protocol_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing import Protocol, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class MyProtocol(Protocol[T]):\n        @classmethod\n        def build(cls) -> Self:\n          return cls()\n      class Ok:\n        @classmethod\n        def build(cls) -> 'Ok':\n          return cls()\n      class Bad:\n        @classmethod\n        def build(cls) -> int:\n          return 0\n      def f(x: MyProtocol[str]):\n        pass\n      f(Ok())\n      f(Bad())  # wrong-arg-types\n    ")

    def test_signature_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing_extensions import Self\n      class Foo:\n        def f(self) -> Self:\n          return self\n      class Ok(Foo):\n        def f(self) -> 'Ok':\n          return self\n      class Bad(Foo):\n        def f(self) -> int:  # signature-mismatch\n          return 0\n    ")

    def test_class_attribute(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing_extensions import Self\n      class Foo:\n        x: Self\n      class Bar(Foo):\n        pass\n      assert_type(Foo.x, Foo)\n      assert_type(Foo().x, Foo)\n      assert_type(Bar.x, Bar)\n      assert_type(Bar().x, Bar)\n    ')

    def test_instance_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Self\n      class Foo:\n        def __init__(self, x: Self):\n          self.x = x\n          self.y: Self = __any_object__\n      class Bar(Foo):\n        pass\n      assert_type(Foo(__any_object__).x, Foo)\n      assert_type(Foo(__any_object__).y, Foo)\n      assert_type(Bar(__any_object__).x, Bar)\n      assert_type(Bar(__any_object__).y, Bar)\n    ')

    def test_cast(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import cast\n      from typing_extensions import Self\n      class Foo:\n        def f(self):\n          return cast(Self, __any_object__)\n      class Bar(Foo):\n        pass\n      assert_type(Foo().f(), Foo)\n      assert_type(Bar().f(), Bar)\n    ')

    def test_generic_attribute(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Generic, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class C(Generic[T]):\n        x: Self\n      class D(C[T]):\n        pass\n      assert_type(C[int].x, C[int])\n      assert_type(C[int]().x, C[int])\n      assert_type(D[str].x, D[str])\n      assert_type(D[str]().x, D[str])\n    ")

    def test_attribute_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing import Protocol\n      from typing_extensions import Self\n      class C(Protocol):\n        x: Self\n      class Ok:\n        x: 'Ok'\n      class Bad:\n        x: int\n      def f(c: C):\n        pass\n      f(Ok())\n      f(Bad())  # wrong-arg-types\n    ")

class SelfPyiTest(test_base.BaseTest):
    """Tests for typing.Self usage in type stubs."""

    def test_instance_method_return(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        def f(self) -> Self: ...\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A().f(), foo.A)\n        assert_type(B().f(), B)\n      ')

    def test_classmethod_return(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        @classmethod\n        def f(cls) -> Self: ...\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A.f(), foo.A)\n        assert_type(B.f(), B)\n      ')

    def test_new_return(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        def __new__(cls) -> Self: ...\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A(), foo.A)\n        assert_type(B(), B)\n      ')

    def test_parameterized_return(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        def f(self) -> list[Self]: ...\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A().f(), "List[foo.A]")\n        assert_type(B().f(), "List[B]")\n      ')

    def test_parameter(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        def f(self, other: Self) -> bool: ...\n    ')]):
            errors = self.CheckWithErrors('\n        import foo\n        class B(foo.A):\n          pass\n        B().f(B())  # ok\n        B().f(0)  # wrong-arg-types[e]\n      ')
            self.assertErrorSequences(errors, {'e': ['Expected', 'B', 'Actual', 'int']})

    def test_nested_class(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        class B:\n          def f(self) -> Self: ...\n    ')]):
            self.Check('\n        import foo\n        class C(foo.A.B):\n          pass\n        assert_type(foo.A.B().f(), foo.A.B)\n        assert_type(C().f(), C)\n      ')

    def test_generic_class(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, Self, TypeVar\n      T = TypeVar('T')\n      class A(Generic[T]):\n        def copy(self) -> Self: ...\n    ")]):
            self.Check("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        class B(foo.A[T]):\n          pass\n        assert_type(foo.A[int]().copy(), foo.A[int])\n        assert_type(B[str]().copy(), B[str])\n      ")

    def test_protocol(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', "\n      from typing import Protocol, Self, TypeVar\n      T = TypeVar('T')\n      class MyProtocol(Protocol[T]):\n        @classmethod\n        def build(cls) -> Self: ...\n    ")]):
            self.CheckWithErrors("\n        import foo\n        class Ok:\n          @classmethod\n          def build(cls) -> 'Ok':\n            return cls()\n        class Bad:\n          @classmethod\n          def build(cls) -> int:\n            return 0\n        def f(x: foo.MyProtocol[str]):\n          pass\n        f(Ok())\n        f(Bad())  # wrong-arg-types\n      ")

    def test_signature_mismatch(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        def f(self) -> Self: ...\n    ')]):
            self.CheckWithErrors("\n        import foo\n        class Ok(foo.A):\n          def f(self) -> 'Ok':\n            return self\n        class Bad(foo.A):\n          def f(self) -> int:  # signature-mismatch\n            return 0\n      ")

    def test_attribute(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import Self\n      class A:\n        x: Self\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A.x, foo.A)\n        assert_type(foo.A().x, foo.A)\n        assert_type(B.x, B)\n        assert_type(B().x, B)\n      ')

    def test_generic_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', "\n      from typing import Generic, Self, TypeVar\n      T = TypeVar('T')\n      class A(Generic[T]):\n        x: Self\n    ")]):
            self.Check("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        class B(foo.A[T]):\n          pass\n        assert_type(foo.A[str].x, foo.A[str])\n        assert_type(foo.A[int]().x, foo.A[int])\n        assert_type(B[int].x, B[int])\n        assert_type(B[str]().x, B[str])\n      ")

    def test_attribute_mismatch(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import Protocol, Self\n      class C(Protocol):\n        x: Self\n    ')]):
            self.CheckWithErrors("\n        import foo\n        class Ok:\n          x: 'Ok'\n        class Bad:\n          x: str\n        def f(c: foo.C):\n          pass\n        f(Ok())\n        f(Bad())  # wrong-arg-types\n      ")

class SelfReingestTest(test_base.BaseTest):
    """Tests for outputting typing.Self to a stub and reading the stub back in."""

    def test_output(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import Self\n      class A:\n        def f(self) -> Self:\n          return self\n    ')
        expected = textwrap.dedent('      from typing import Self\n\n      class A:\n          def f(self) -> Self: ...')
        actual = pytd_utils.Print(ty)
        self.assertMultiLineEqual(expected, actual)

    def test_attribute_output(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing_extensions import Self\n      class A:\n        x: Self\n        def __init__(self):\n          self.y: Self = __any_object__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Self\n      class A:\n        x: Self\n        y: Self\n        def __init__(self) -> None: ...\n    ')

    def test_instance_method_return(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        def f(self) -> Self:\n          return self\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A().f(), foo.A)\n        assert_type(B().f(), B)\n      ')

    def test_parameterized_return(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      from typing import List\n      from typing_extensions import Self\n      class A:\n        def f(self) -> List[Self]:\n          return [self]\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A().f(), "List[foo.A]")\n        assert_type(B().f(), "List[B]")\n      ')

    def test_parameter(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        def f(self, other: Self) -> bool:\n          return False\n    ')]):
            errors = self.CheckWithErrors('\n        import foo\n        class B(foo.A):\n          pass\n        B().f(B())  # ok\n        B().f(0)  # wrong-arg-types[e]\n      ')
            self.assertErrorSequences(errors, {'e': ['Expected', 'B', 'Actual', 'int']})

    def test_nested_class(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        class B:\n          def f(self) -> Self:\n            return self\n    ')]):
            self.Check('\n        import foo\n        class C(foo.A.B):\n          pass\n        assert_type(foo.A.B().f(), foo.A.B)\n        assert_type(C().f(), C)\n      ')

    @test_utils.skipBeforePy((3, 11), 'typing.Self is new in 3.11')
    def test_import_from_typing(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      from typing import Self\n      class A:\n        def f(self) -> Self:\n          return self\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A().f(), foo.A)\n        assert_type(B().f(), B)\n      ')

    def test_classmethod(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        @classmethod\n        def build(cls) -> Self:\n          return cls()\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A.build(), foo.A)\n        assert_type(B.build(), B)\n      ')

    def test_new(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        def __new__(cls) -> Self:\n          return super().__new__(cls)\n    ')]):
            self.Check('\n        import foo\n        class B(foo.A):\n          pass\n        assert_type(foo.A(), foo.A)\n        assert_type(B(), B)\n      ')

    def test_generic_class(self):
        if False:
            return 10
        with self.DepTree([('foo.py', "\n      from typing import Generic, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class A(Generic[T]):\n        def copy(self) -> Self:\n          return self\n    ")]):
            self.Check("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        class B(foo.A[T]):\n          pass\n        assert_type(foo.A[int]().copy(), foo.A[int])\n        assert_type(B[str]().copy(), B[str])\n      ")

    def test_protocol(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', "\n      from typing import Protocol, TypeVar\n      from typing_extensions import Self\n      T = TypeVar('T')\n      class MyProtocol(Protocol[T]):\n        def f(self) -> Self:\n          return self\n    ")]):
            self.CheckWithErrors('\n        import foo\n        from typing_extensions import Self\n        class Ok:\n          def f(self) -> Self:\n            return self\n        class Bad:\n          def f(self) -> int:\n            return 0\n        def f(x: foo.MyProtocol[str]):\n          pass\n        f(Ok())\n        f(Bad())  # wrong-arg-types\n      ')

    def test_signature_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import Self\n      class A:\n        def f(self) -> Self:\n          return self\n    ')]):
            self.CheckWithErrors('\n        import foo\n        class Ok(foo.A):\n          def f(self) -> foo.A:\n            return self\n        class Bad(foo.A):\n          def f(self) -> int:  # signature-mismatch\n            return 0\n      ')

class IllegalLocationTest(test_base.BaseTest):
    """Tests for typing.Self in illegal locations."""

    def test_function_annotation_not_in_class(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import Self\n      def f(x) -> Self:  # invalid-annotation[e]\n        return x\n    ')
        self.assertErrorSequences(errors, {'e': ["'typing.Self' outside of a class"]})

    def test_variable_annotation_not_in_class(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import Self\n      x: Self  # invalid-annotation[e1]\n      y = ...  # type: Self  # invalid-annotation[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ["'Self' not in scope"], 'e2': ["'Self' not in scope"]})
if __name__ == '__main__':
    test_base.main()