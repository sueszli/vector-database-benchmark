"""Tests for reloading generated pyi."""
from pytype.tests import test_base

class ReingestTest(test_base.BaseTest):
    """Tests for reloading the pyi we generate."""

    def test_type_parameter_bound(self):
        if False:
            for i in range(10):
                print('nop')
        foo = '\n      from typing import TypeVar\n      T = TypeVar("T", bound=float)\n      def f(x: T) -> T: return x\n    '
        with self.DepTree([('foo.py', foo, dict(deep=False))]):
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.f("")  # wrong-arg-types[e]\n      ')
            self.assertErrorRegexes(errors, {'e': 'float.*str'})

    def test_default_argument_type(self):
        if False:
            return 10
        foo = '\n      from typing import Any, Callable, TypeVar\n      T = TypeVar("T")\n      def f(x):\n        return True\n      def g(x: Callable[[T], Any]) -> T: ...\n    '
        with self.DepTree([('foo.py', foo)]):
            self.Check('\n        import foo\n        foo.g(foo.f).upper()\n      ')

    def test_duplicate_anystr_import(self):
        if False:
            while True:
                i = 10
        dep1 = '\n      from typing import AnyStr\n      def f(x: AnyStr) -> AnyStr:\n        return x\n    '
        dep2 = '\n      from typing import AnyStr\n      from dep1 import f\n      def g(x: AnyStr) -> AnyStr:\n        return x\n    '
        deps = [('dep1.py', dep1), ('dep2.py', dep2)]
        with self.DepTree(deps):
            self.Check('import dep2')

class ReingestTestPy3(test_base.BaseTest):
    """Python 3 tests for reloading the pyi we generate."""

    def test_instantiate_pyi_class(self):
        if False:
            for i in range(10):
                print('nop')
        foo = '\n      import abc\n      class Foo(metaclass=abc.ABCMeta):\n        @abc.abstractmethod\n        def foo(self):\n          pass\n      class Bar(Foo):\n        def foo(self):\n          pass\n    '
        with self.DepTree([('foo.py', foo)]):
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.Foo()  # not-instantiable[e]\n        foo.Bar()\n      ')
            self.assertErrorRegexes(errors, {'e': 'foo\\.Foo.*foo'})

    def test_use_class_attribute_from_annotated_new(self):
        if False:
            i = 10
            return i + 15
        foo = '\n      class Foo:\n        def __new__(cls) -> "Foo":\n          return cls()\n      class Bar:\n        FOO = Foo()\n    '
        with self.DepTree([('foo.py', foo)]):
            self.Check('\n        import foo\n        print(foo.Bar.FOO)\n      ')
if __name__ == '__main__':
    test_base.main()