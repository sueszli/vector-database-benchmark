"""Tests for the @typing.dataclass_transform decorator."""
from pytype.tests import test_base

class TestDecorator(test_base.BaseTest):
    """Tests for the @dataclass_transform decorator."""

    def test_invalid_target(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing_extensions import dataclass_transform\n      x = 10\n      dataclass_transform()(x) # dataclass-error\n    ')

    def test_args(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing_extensions import dataclass_transform\n      dataclass_transform(eq_default=True)  # not-supported-yet\n      def f(cls):\n        return cls\n    ')

    def test_pyi_args(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      from typing import dataclass_transform\n\n      @dataclass_transform(eq_default=True)\n      def dc(cls): ...\n    ')]):
            self.Check('\n        import foo\n\n        @foo.dc\n        class A:\n          x: int\n\n        a = A(x=10)\n      ')

class TestFunction(test_base.BaseTest):
    """Tests for @dataclass_transform on functions."""

    def test_py_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing_extensions import dataclass_transform\n\n      # NOTE: The decorator overrides the function body and makes `dc` a\n      # dataclass decorator.\n      @dataclass_transform()\n      def dc(cls):\n        return cls\n\n      @dc\n      class A:\n        x: int\n\n      a = A(x=10)\n      assert_type(a, A)\n    ')

    def test_write_pyi(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform(eq_default=True)  # not-supported-yet\n      def dc(f):\n        return f\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import TypeVar, dataclass_transform\n\n      _T0 = TypeVar('_T0')\n\n      @dataclass_transform\n      def dc(f: _T0) -> _T0: ...\n    ")

    def test_pyi_function(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', "\n      from typing import TypeVar, dataclass_transform\n\n      _T0 = TypeVar('_T0')\n\n      @dataclass_transform\n      def dc(cls: _T0) -> _T0: ...\n    ")]):
            self.CheckWithErrors('\n        import foo\n\n        @foo.dc\n        class A:\n          x: int\n\n        a = A(x=10)\n        b = A() # missing-parameter\n      ')

    def test_reingest(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.py', '\n      from typing import TypeVar\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      def dc(f):\n        return f\n    ')]):
            self.CheckWithErrors('\n        import foo\n\n        @foo.dc\n        class A:\n          x: int\n\n        a = A(x=10)\n        b = A() # missing-parameter\n      ')

    def test_function_with_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import dataclass_transform\n      @dataclass_transform()\n      def dc(cls, *, init=True, repr=True, eq=True, order=False,\n             unsafe_hash=False, frozen=False, kw_only=False):\n        return cls\n    ')

class TestClass(test_base.BaseTest):
    """Tests for @dataclass_transform on classes."""

    def test_single_inheritance(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing_extensions import dataclass_transform\n      @dataclass_transform()\n      class Base: ...\n\n      class A(Base):\n          x: int\n          y: str\n\n      class B(A):\n          z: int\n\n      a = B(1, '2', 3)\n      b = B(1, 2)  # missing-parameter\n      c = B(1, 2, 3)  # wrong-arg-types\n    ")

    def test_multiple_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing_extensions import dataclass_transform\n      @dataclass_transform()\n      class Mixin: ...\n\n      class Base:\n        pass\n\n      class A(Base, Mixin):\n          x: int\n          y: str\n\n      class B(A):\n          z: int\n\n      a = B(1, '2', 3)\n      b = B(1, 2)  # missing-parameter\n      c = B(1, 2, 3)  # wrong-arg-types\n    ")

    def test_redundant_decorator(self):
        if False:
            return 10
        self.CheckWithErrors("\n      import dataclasses\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class Base: ...\n\n      @dataclasses.dataclass\n      class A(Base):\n          x: int\n          y: str\n\n      class B(A):\n          z: int\n\n      a = B(1, '2', 3)\n      b = B(1, 2)  # missing-parameter\n      c = B(1, 2, 3)  # wrong-arg-types\n    ")

    def test_redundant_decorator_pyi(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import dataclasses\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class A:\n        pass\n\n      @dataclasses.dataclass\n      class B(A):\n        x: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import dataclasses\n      from typing import dataclass_transform\n\n      @dataclass_transform\n      class A: ...\n\n      @dataclasses.dataclass\n      class B(A):\n        x: int\n        def __init__(self, x: int) -> None: ...\n    ')

    def test_write_pyi(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors('\n      from typing_extensions import dataclass_transform\n      @dataclass_transform()\n      class Mixin: ...\n\n      class Base:\n        pass\n\n      class A(Base, Mixin):\n          x: int\n          y: str\n\n      class B(A):\n          z: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import dataclasses\n      from typing import dataclass_transform\n\n      @dataclasses.dataclass\n      class A(Base, Mixin):\n          x: int\n          y: str\n          def __init__(self, x: int, y: str) -> None: ...\n\n      @dataclasses.dataclass\n      class B(A):\n          z: int\n          def __init__(self, x: int, y: str, z: int) -> None: ...\n\n      class Base: ...\n\n      @dataclass_transform\n      class Mixin: ...\n    ')

    def test_pyi_class(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      from typing import dataclass_transform\n\n      @dataclass_transform\n      class Mixin:\n        ...\n    ')]):
            self.CheckWithErrors("\n        import foo\n\n        class Base(foo.Mixin):\n          x: int\n\n        class A(Base):\n          y: str\n\n        a = A(x=10, y='foo')\n        b = A(10) # missing-parameter\n        c = A(10, 20) # wrong-arg-types\n      ")

    def test_reingest(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class Mixin:\n        pass\n    ')]):
            self.CheckWithErrors("\n        import foo\n\n        class Base(foo.Mixin):\n          x: int\n\n        class A(Base):\n          y: str\n\n        a = A(x=10, y='foo')\n        b = A(10) # missing-parameter\n        c = A(10, 20) # wrong-arg-types\n      ")

    def test_init_subclass_impl(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      import dataclasses\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class X:\n        def __init_subclass__(cls):\n          return dataclasses.dataclass(cls)\n    ')]):
            self.CheckWithErrors("\n        import foo\n        class Y(foo.X):\n          x: int\n        Y()  # missing-parameter\n        Y(x=0)  # ok\n        Y(x='')  # wrong-arg-types\n      ")

class TestMetaclass(test_base.BaseTest):
    """Tests for @dataclass_transform on metaclasses."""

    def test_py_metaclass(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class Meta(type): ...\n\n      class Base(metaclass=Meta): ...\n\n      class A(Base):\n        x: int\n        y: str\n\n      a = A(1, '2')\n      a = A(1, 2)  # wrong-arg-types\n    ")

    def test_pyi_class(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      from typing import dataclass_transform\n\n      @dataclass_transform\n      class Meta(type):\n        ...\n\n      class Base(metaclass=Meta): ...\n    ')]):
            self.CheckWithErrors("\n        import foo\n\n        class A(foo.Base):\n          x: int\n          y: str\n\n        a = A(x=10, y='foo')\n        b = A(10) # missing-parameter\n        c = A(10, 20) # wrong-arg-types\n      ")

    def test_reingest(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.py', '\n      from typing_extensions import dataclass_transform\n\n      @dataclass_transform()\n      class Meta(type):\n        ...\n\n      class Base(metaclass=Meta): ...\n    ')]):
            self.CheckWithErrors("\n        import foo\n\n        class A(foo.Base):\n          x: int\n          y: str\n\n        a = A(x=10, y='foo')\n        b = A(10) # missing-parameter\n        c = A(10, 20) # wrong-arg-types\n      ")
if __name__ == '__main__':
    test_base.main()