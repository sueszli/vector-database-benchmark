"""Test methods."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestMethods(test_base.BaseTest):
    """Tests for class methods."""

    def test_function_init(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def __init__(self: int):\n        return self\n    ')
        self.assertTypesMatchPytd(ty, '\n      def __init__(self: int) -> int: ...\n    ')

    def test_annotated_self(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(x: int):\n          pass  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*x'})

    def test_late_annotated_self(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(x: "X"):\n          pass  # invalid-annotation[e]\n      class X:\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'X.*x'})

    def test_attribute_with_annotated_self(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(self: int):\n          self.x = 3  # invalid-annotation[e]\n        def foo(self):\n          return self.x\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*self'})

    def test_attribute_with_annotated_self_and_function_init(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(self: int):\n          self.x = 3  # invalid-annotation[e]\n      def __init__(self: int):\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*self'})

    def test_use_abstract_classmethod(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n\n        class Foo(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          @classmethod\n          def foo(cls, value) -> int: ...\n      ')
            self.Check('\n        import collections\n        import foo\n\n        class Bar:\n          def __init__(self, **kwargs):\n            for k, v in self.f().items():\n              v.foo(kwargs[k])\n\n          def f(self) -> collections.OrderedDict[str, foo.Foo]:\n            return __any_object__\n      ', pythonpath=[d.path])

    def test_max_depth(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Any, Union\n\n      class A:\n        def __init__(self, x: int):\n          self.x = 1\n          self.FromInt(x)\n\n        def cmp(self, other: 'A') -> bool:\n          return self.Upper() < other.Upper()\n\n        def FromInt(self, x: int) -> None:\n          self.x = 'x'\n\n        def Upper(self) -> str:\n          return self.x.upper()  # attribute-error\n    ", maximum_depth=2)

    def test_call_dispatch(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union\n      class Foo:\n        def __call__(self):\n          pass\n      class Bar:\n        def __call__(self, x):\n          pass\n      def f(x: Union[Foo, Bar]):\n        if isinstance(x, Foo):\n          return x()\n    ')

    def test_lookup_on_dynamic_class(self):
        if False:
            return 10
        self.Check("\n      class Foo:\n        _HAS_DYNAMIC_ATTRIBUTES = True\n        def f(self) -> str:\n          return ''\n        def g(self):\n          assert_type(self.f(), str)\n    ")

class TestMethodsPy3(test_base.BaseTest):
    """Test python3-specific method features."""

    def test_init_subclass_classmethod(self):
        if False:
            print('Hello World!')
        '__init_subclass__ should be promoted to a classmethod.'
        self.Check("\n      from typing import Type\n\n      _REGISTERED_BUILDERS = {}\n\n      class A():\n        def __init_subclass__(cls, **kwargs):\n          _REGISTERED_BUILDERS['name'] = cls\n\n      def get_builder(name: str) -> Type[A]:\n        return _REGISTERED_BUILDERS[name]\n    ")

    def test_pass_through_typevar(self):
        if False:
            return 10
        self.Check("\n      from typing import TypeVar\n      F = TypeVar('F')\n      def f(x: F) -> F:\n        return x\n      class A:\n        def f(self, x: float) -> float:\n          return x\n      g = f(A().f)\n      assert_type(g(0), float)\n    ")

    def test_dunder_self(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Type\n      class A:\n        def foo(self):\n          return 42\n\n        @classmethod\n        def bar(cls):\n          return cls()\n\n      a = A().foo.__self__\n      b = A.bar.__self__\n      assert_type(a, A)\n      assert_type(b, Type[A])\n    ')

    def test_signature_inference(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class C:\n        def __init__(self, fn1, fn2):\n          self._fn1 = fn1\n          self._fn2 = fn2\n        def f(self, x):\n          self._fn1(x)\n          self._fn2(x=x)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class C:\n        def __init__(self, fn1, fn2) -> None: ...\n        def f(self, x) -> None: ...\n        def _fn1(self, _1) -> Any: ...\n        def _fn2(self, x) -> Any: ...\n    ')
if __name__ == '__main__':
    test_base.main()