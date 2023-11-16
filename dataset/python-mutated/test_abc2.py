"""Tests for @abc.abstractmethod in abc_overlay.py."""
from pytype.tests import test_base
from pytype.tests import test_utils

class AbstractMethodTests(test_base.BaseTest):
    """Tests for @abc.abstractmethod."""

    def test_no_skip_call(self):
        if False:
            print('Hello World!')
        self.Check('\n      import abc\n      class Example(metaclass=abc.ABCMeta):\n        @abc.abstractmethod\n        def foo(self) -> int:\n          return None\n    ', skip_repeat_calls=False)

    def test_multiple_inheritance_builtins(self):
        if False:
            return 10
        self.Check('\n      import abc\n      class Foo(object, metaclass=abc.ABCMeta):\n        pass\n      class Bar1(Foo, tuple):\n        pass\n      class Bar2(Foo, bytes):\n        pass\n      class Bar3(Foo, str):\n        pass\n      class Bar4(Foo, bytearray):\n        pass\n      class Bar5(Foo, dict):\n        pass\n      class Bar6(Foo, list):\n        pass\n      class Bar7(Foo, set):\n        pass\n      class Bar8(Foo, frozenset):\n        pass\n      class Bar9(Foo, memoryview):\n        pass\n      class BarA(Foo, range):\n        pass\n      Bar1()\n      Bar2()\n      Bar3()\n      Bar4()\n      Bar5()\n      Bar6()\n      Bar7()\n      Bar8()\n      Bar9(b"")\n      BarA(0)\n    ')

    def test_abstractproperty(self):
        if False:
            while True:
                i = 10
        (ty, errors) = self.InferWithErrors('\n      import abc\n      class Foo(metaclass=abc.ABCMeta):\n        @abc.abstractproperty\n        def foo(self):\n          return 42\n      class Bar(Foo):\n        @property\n        def foo(self):\n          return super(Bar, self).foo\n      v1 = Foo().foo  # not-instantiable[e]\n      v2 = Bar().foo\n    ')
        self.assertTypesMatchPytd(ty, "\n      import abc\n      from typing import Annotated, Any\n      v1 = ...  # type: Any\n      v2 = ...  # type: int\n      class Bar(Foo):\n        foo = ...  # type: Annotated[int, 'property']\n      class Foo(metaclass=abc.ABCMeta):\n        foo = ...  # type: Annotated[Any, 'property']\n    ")
        self.assertErrorRegexes(errors, {'e': 'Foo.*foo'})

    def test_dictviews(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from collections import abc\n      from typing import Dict\n      d: Dict[str, int] = {}\n      abc.ItemsView(d)\n      abc.KeysView(d)\n      abc.ValuesView(d)\n    ')

    def test_instantiate_abstract_class_annotation(self):
        if False:
            print('Hello World!')
        self.Check('\n      import abc\n      from typing import Type\n      class A(metaclass=abc.ABCMeta):\n        @abc.abstractmethod\n        def a(self):\n          pass\n      def f(x: Type[A]):\n        return x()\n    ')

    def test_instantiate_abstract_pytdclass_annotation(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n        class A(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          def a(self) -> None: ...\n      ')
            self.Check('\n        import foo\n        from typing import Type\n        def f(x: Type[foo.A]):\n          return x()\n      ', pythonpath=[d.path])

    def test_instantiate_generic_abstract_class(self):
        if False:
            print('Hello World!')
        self.Check("\n      import abc\n      from typing import Generic, Type, TypeVar\n      T = TypeVar('T')\n      class A(Generic[T], abc.ABC):\n        @abc.abstractmethod\n        def a(self): ...\n      def f(x: Type[A[int]]):\n        return x()\n    ")

    def test_instantiate_abstract_class_in_own_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import abc\n      class Foo(abc.ABC):\n        @abc.abstractmethod\n        def f(self): ...\n        @classmethod\n        def g(cls):\n          return cls()\n    ')

    def test_abstract_classmethod(self):
        if False:
            return 10
        self.Check('\n      import abc\n      class Foo(abc.ABC):\n        @classmethod\n        @abc.abstractmethod\n        def f(cls) -> str: ...\n    ')

    def test_bad_abstract_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      import abc\n      class Foo:  # ignored-abstractmethod[e]\n        @classmethod\n        @abc.abstractmethod\n        def f(cls) -> str: ...  # bad-return-type\n    ')
        self.assertErrorSequences(errors, {'e': ['on method Foo.f']})

    def test_bad_abstract_pyi_method(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      import abc\n      class Foo(abc.ABC):\n        @abc.abstractmethod\n        def f(self) -> int: ...\n    ')]):
            self.CheckWithErrors('\n        import foo\n        class Bar:  # ignored-abstractmethod\n          f = foo.Foo.f\n      ')

    def test_abstract_property(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("\n      import abc\n      class Foo(abc.ABC):\n        @abc.abstractmethod\n        @property\n        def f(self) -> str:  # wrong-arg-types[e]\n          return 'a'\n\n        @property\n        @abc.abstractmethod\n        def g(self) -> str:\n          return 'a'\n    ")
        self.assertErrorSequences(errors, {'e': ['Expected', 'Callable', 'Actual', 'property']})

    def test_instantiate_abcmeta(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import abc\n      ABC = abc.ABCMeta('ABC', (object,), {})\n      class Foo(ABC):\n        @abc.abstractmethod\n        def f(self):\n          pass\n    ")

    def test_ignored_abstractmethod_nested(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import abc\n      def f():\n        class C:  # ignored-abstractmethod\n          @abc.abstractmethod\n          def f(self):\n            pass\n    ')

    def test_abstractmethod_variants(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import abc\n      class C(abc.ABC):\n        @abc.abstractclassmethod\n        def f(cls) -> int: ...\n        @abc.abstractstaticmethod\n        def g(): ...\n    ')

    def test_inference(self):
        if False:
            return 10
        ty = self.Infer('\n      from abc import abstractclassmethod\n      from abc import abstractmethod\n      from abc import abstractproperty\n      from abc import abstractstaticmethod\n    ')
        self.assertTypesMatchPytd(ty, "\n      import abc\n      from typing import Callable, Type, TypeVar\n\n      abstractclassmethod: Type[abc.abstractclassmethod]\n      abstractproperty: Type[abc.abstractproperty]\n      abstractstaticmethod: Type[abc.abstractstaticmethod]\n\n      _FuncT = TypeVar('_FuncT', bound=Callable)\n      def abstractmethod(funcobj: _FuncT) -> _FuncT: ...\n    ")
if __name__ == '__main__':
    test_base.main()