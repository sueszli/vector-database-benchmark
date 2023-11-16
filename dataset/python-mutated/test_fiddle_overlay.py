"""Tests for the fiddle overlay."""
from pytype.tests import test_base
_FIDDLE_PYI = '\nfrom typing import Callable, Generic, Type, TypeVar, Union\n\nT = TypeVar("T")\n\nclass Buildable(Generic[T], metaclass=abc.ABCMeta):\n  def __init__(self, fn_or_cls: Union[Buildable, Type[T], Callable[..., T]], *args, **kwargs) -> None:\n    self = Buildable[T]\n\nclass Config(Generic[T], Buildable[T]):\n  ...\n\nclass Partial(Generic[T], Buildable[T]):\n  def __call__(self, *args, **kwargs): ...\n'

class TestDataclassConfig(test_base.BaseTest):
    """Tests for Config wrapping a dataclass."""

    @property
    def buildable_type_name(self) -> str:
        if False:
            print('Hello World!')
        return 'Config'

    def test_basic(self):
        if False:
            print('Hello World!')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f'\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        a = fiddle.{self.buildable_type_name}(Simple)\n        a.x = 1\n        a.y = 2  # annotation-type-mismatch\n      ')

    def test_return_type(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check(f'\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        def f() -> fiddle.{self.buildable_type_name}[Simple]:\n          a = fiddle.{self.buildable_type_name}(Simple)\n          a.x = 1\n          return a\n      ')

    def test_pyi(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', f'\n            import dataclasses\n            import fiddle\n\n            @dataclasses.dataclass\n            class Simple:\n              x: int\n              y: str\n\n            a: fiddle.{self.buildable_type_name}[Simple]\n         ')]):
            self.CheckWithErrors('\n        import foo\n        a = foo.a\n        a.x = 1\n        a.y = 2  # annotation-type-mismatch\n      ')

    def test_nested_dataclasses(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f'\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        @dataclasses.dataclass\n        class Complex:\n          x: Simple\n          y: str\n\n        a = fiddle.{self.buildable_type_name}(Complex)\n        a.x.x = 1\n        a.x.y = 2  # annotation-type-mismatch\n      ')

    def test_frozen_dataclasses(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f'\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass(frozen=True)\n        class Simple:\n          x: int\n          y: str\n\n        @dataclasses.dataclass(frozen=True)\n        class Complex:\n          x: Simple\n          y: str\n\n        a = fiddle.{self.buildable_type_name}(Complex)\n        a.x.x = 1\n        a.x.y = 2  # annotation-type-mismatch\n      ')

    def test_nested_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check(f"\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class DataClass:\n          x: int\n          y: str\n\n        class RegularClass:\n          def __init__(self, a, b):\n            self.a = a\n            self.b = b\n\n        @dataclasses.dataclass\n        class Parent:\n          child_data: DataClass\n          child_regular: RegularClass\n\n        child_data = fiddle.Config(DataClass, x=1, y='y')\n        child_regular = fiddle.Config(RegularClass, 1, 2)\n        c = fiddle.{self.buildable_type_name}(Parent, child_data, child_regular)\n      ")

    def test_nested_object_assignment(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check(f"\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class DataClass:\n          x: int\n          y: str\n\n        class RegularClass:\n          def __init__(self, a, b):\n            self.a = a\n            self.b = b\n\n        @dataclasses.dataclass\n        class Parent:\n          child_data: DataClass\n          child_regular: RegularClass\n\n        c = fiddle.{self.buildable_type_name}(Parent)\n        c.child_data = fiddle.Config(DataClass)\n        c.child_data = DataClass(x=1, y='y')\n        c.child_regular = fiddle.Config(RegularClass)\n        c.child_regular = RegularClass(1, 2)\n      ")

    def test_init_args(self):
        if False:
            print('Hello World!')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f"\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        a = fiddle.{self.buildable_type_name}(Simple, x=1, y='2')\n        b = fiddle.{self.buildable_type_name}(Simple, 1, '2')\n        c = fiddle.{self.buildable_type_name}(Simple, 1, y='2')\n        d = fiddle.{self.buildable_type_name}(Simple, x='a', y='2')  # wrong-arg-types\n        e = fiddle.{self.buildable_type_name}(Simple, x=1)  # partial initialization is fine\n        f = fiddle.{self.buildable_type_name}(Simple, x=1, z=3)  # wrong-keyword-args\n        g = fiddle.{self.buildable_type_name}(Simple, 1, '2', 3)  # wrong-arg-count\n      ")

    def test_pyi_underlying_class(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', '\n        import dataclasses\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n         ')]):
            self.CheckWithErrors(f"\n        import fiddle\n        from foo import Simple\n\n        a = fiddle.{self.buildable_type_name}(Simple, x=1, y='2')\n        b = fiddle.{self.buildable_type_name}(Simple, 1, '2')\n        c = fiddle.{self.buildable_type_name}(Simple, 1, y='2')\n        d = fiddle.{self.buildable_type_name}(Simple, x='a', y='2')  # wrong-arg-types\n        e = fiddle.{self.buildable_type_name}(Simple, x=1)  # partial initialization is fine\n        f = fiddle.{self.buildable_type_name}(Simple, x=1, z=3)  # wrong-keyword-args\n        g = fiddle.{self.buildable_type_name}(Simple, 1, '2', 3)  # wrong-arg-count\n      ")

    def test_explicit_init(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', '\n        import dataclasses\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n          def __init__(self: Simple, x: int, y: str): ...\n         ')]):
            self.CheckWithErrors(f"\n        import fiddle\n        from foo import Simple\n        a = fiddle.{self.buildable_type_name}(Simple, x=1, y='2')\n        b = fiddle.{self.buildable_type_name}(Simple, 1, '2')\n        c = fiddle.{self.buildable_type_name}(Simple, 1, y='2')\n        d = fiddle.{self.buildable_type_name}(Simple, x='a', y='2')  # wrong-arg-types\n        e = fiddle.{self.buildable_type_name}(Simple, x=1)  # partial initialization is fine\n        f = fiddle.{self.buildable_type_name}(Simple, x=1, z=3)  # wrong-keyword-args\n        g = fiddle.{self.buildable_type_name}(Simple, 1, '2', 3)  # wrong-arg-count\n      ")

    def test_typevar(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f"\n        import dataclasses\n        import fiddle\n        from typing import TypeVar\n\n        _T = TypeVar('_T')\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        def passthrough(conf: fiddle.{self.buildable_type_name}[_T]) -> fiddle.{self.buildable_type_name}[_T]:\n          return conf\n\n        a = fiddle.{self.buildable_type_name}(Simple)\n        x = passthrough(a)\n        assert_type(x, fiddle.{self.buildable_type_name}[Simple])\n    ")

    def test_pyi_typevar(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', f"\n          import fiddle\n          from typing import TypeVar\n\n          _T = TypeVar('_T')\n\n          def build(buildable: fiddle.{self.buildable_type_name}[_T]) -> _T: ...\n         ")]):
            self.Check(f"\n        import dataclasses\n        import fiddle\n        import foo\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        a = fiddle.{self.buildable_type_name}(Simple, x=1, y='2')\n        b = foo.build(a)\n        assert_type(b, Simple)\n      ")

    def test_bare_type(self):
        if False:
            print('Hello World!')
        'Check that we can match fiddle.Config against fiddle.Config[A].'
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check(f'\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class Simple:\n          x: int\n          y: str\n\n        def f() -> fiddle.{self.buildable_type_name}:\n          a = fiddle.{self.buildable_type_name}(Simple)\n          a.x = 1\n          return a\n      ')

    def test_generic_dataclass(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f"\n        from typing import Generic, TypeVar\n        import dataclasses\n        import fiddle\n\n        T = TypeVar('T')\n\n        @dataclasses.dataclass\n        class D(Generic[T]):\n          x: T\n\n        a = fiddle.{self.buildable_type_name}(D)\n        a.x = 1\n        b = fiddle.{self.buildable_type_name}(D[int])\n        b.x = 1\n        c = fiddle.{self.buildable_type_name}(D[str])\n        c.x = 1  # annotation-type-mismatch\n      ")

    def test_dataclass_error_detection(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors(f'\n        import dataclasses\n        import fiddle\n        @dataclasses.dataclass\n        class A:\n          x: int\n          y: str\n        A(x=0)  # missing-parameter\n        fiddle.{self.buildable_type_name}(A, x=0)\n        A(x=0)  # missing-parameter\n      ')

    def test_dataclass_error_detection_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', '\n      import dataclasses\n      @dataclasses.dataclass\n      class Foo:\n        x: int\n        y: str\n        def __init__(self, x: int, y: str) -> None: ...\n    ')]):
            self.CheckWithErrors(f'\n        import fiddle\n        import foo\n        foo.Foo(x=0)  # missing-parameter\n        fiddle.{self.buildable_type_name}(foo.Foo, x=0)\n        foo.Foo(x=0)  # missing-parameter\n      ')

    def test_imported_dataclass(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI), ('foo.pyi', '\n      import dataclasses\n      @dataclasses.dataclass\n      class Foo:\n        x: int\n        y: str\n        def __init__(self, x: int, y: str) -> None: ...\n    ')]):
            errors = self.CheckWithErrors(f"\n        import fiddle\n        import foo\n        fiddle.{self.buildable_type_name}(foo.Foo, x='')  # wrong-arg-types[e]\n      ")
            self.assertErrorSequences(errors, {'e': ['Expected', 'x: int', 'Actual', 'x: str']})

class TestDataclassPartial(TestDataclassConfig):
    """Test fiddle.Partial over dataclasses."""

    @property
    def buildable_type_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Partial'

    def test_nested_partial_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check("\n        import dataclasses\n        import fiddle\n        from typing import Callable\n\n        @dataclasses.dataclass\n        class DataClass:\n          x: int\n          y: str\n\n        class RegularClass:\n          def __init__(self, a, b):\n            self.a = a\n            self.b = b\n\n        @dataclasses.dataclass\n        class Parent:\n          data_factory: Callable[..., DataClass]\n          regular_factory: Callable[..., RegularClass]\n\n        def data_builder(x: int = 1) -> DataClass:\n          return DataClass(x=x, y='y')\n\n        def regular_builder() -> RegularClass:\n          return RegularClass(1, 2)\n\n        c = fiddle.Partial(Parent)\n        c.child_data = data_builder\n        c.child_data = fiddle.Partial(DataClass)\n        c.regular_factory = regular_builder\n        c.regular_factory = fiddle.Partial(RegularClass)\n      ")

    def test_config_partial_mismatch(self):
        if False:
            print('Hello World!')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.CheckWithErrors('\n        import dataclasses\n        import fiddle\n\n        @dataclasses.dataclass\n        class DataClass:\n          x: int\n          y: str\n\n        def f() -> fiddle.Config:\n          return fiddle.Partial(DataClass)  # bad-return-type\n      ')

class TestClassConfig(test_base.BaseTest):
    """Tests for Config wrapping a regular python class."""

    def test_basic(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check('\n        import fiddle\n\n        class Simple:\n          x: int\n          y: str\n\n        a = fiddle.Config(Simple)\n        a.x = 1\n        a.y = 2\n      ')

    def test_init_args(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check('\n        import fiddle\n\n        class Simple:\n          x: int\n          y: str\n\n        a = fiddle.Config(Simple, 1)\n        b = fiddle.Config(Simple, 1, 2)  # no type checking yet\n      ')

class TestFunctionConfig(test_base.BaseTest):
    """Tests for Config wrapping a function."""

    def test_basic(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check('\n        import fiddle\n\n        def Simple(x: int, y: str):\n          pass\n\n        a = fiddle.Config(Simple)\n        a.x = 1\n        a.y = 2\n      ')

    def test_init_args(self):
        if False:
            print('Hello World!')
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check('\n        import fiddle\n\n        def Simple(x: int, y: str):\n          pass\n\n        a = fiddle.Config(Simple, 1)\n        b = fiddle.Config(Simple, 1, 2)  # no type checking yet\n        b = fiddle.Config(Simple, 1, 2, 3)  # no arg checking yet\n      ')

    def test_matching(self):
        if False:
            return 10
        with self.DepTree([('fiddle.pyi', _FIDDLE_PYI)]):
            self.Check('\n        import fiddle\n\n        def Simple(x: int, y: str):\n          pass\n\n        def f() -> fiddle.Config[Simple]:\n          return fiddle.Config(Simple, 1)\n      ')
if __name__ == '__main__':
    test_base.main()