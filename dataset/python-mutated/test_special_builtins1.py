"""Tests of special builtins (special_builtins.py)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class SpecialBuiltinsTest(test_base.BaseTest):
    """Tests for special_builtins.py."""

    def test_next(self):
        if False:
            while True:
                i = 10
        (ty, _) = self.InferWithErrors('\n      a = iter([1, 2, 3])\n      b = next(a)\n      c = next(42) # wrong-arg-types\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      a: listiterator[int]\n      b: int\n      c: Any\n    ')

    def test_next_none(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      next(None)\n    ')

    def test_next_ambiguous(self):
        if False:
            return 10
        self.assertNoCrash(self.Check, '\n      class Foo:\n        def a(self):\n          self._foo = None\n        def b(self):\n          self._foo = __any_object__\n        def c(self):\n          next(self._foo)\n    ')

    def test_abs(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Check, '\n      abs(None)\n    ')

    def test_property_matching(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class A():\n        def setter(self, other):\n          pass\n        def getter(self):\n          return 42\n        def create_property(self, cls, property_name):\n          setattr(cls, property_name, property(self.getter, self.setter))\n    ')

    def test_property_from_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          def get_foo(self) -> int: ...\n      ')
            ty = self.Infer('\n        import foo\n        class Bar(foo.Foo):\n          foo = property(fget=foo.Foo.get_foo)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import foo\n        from typing import Annotated\n        class Bar(foo.Foo):\n          foo = ...  # type: Annotated[int, 'property']\n      ")

    def test_property_from_native_function(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo(dict):\n        foo = property(fget=dict.__getitem__)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated, Any\n      class Foo(dict):\n        foo = ...  # type: Annotated[Any, 'property']\n    ")

    def test_property_from_pyi_with_type_parameter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Union\n        class Foo:\n          def get_foo(self) -> Union[str, int]: ...\n      ')
            ty = self.Infer('\n        import foo\n        class Bar(foo.Foo):\n          foo = property(fget=foo.Foo.get_foo)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import foo\n        from typing import Annotated, Union\n        class Bar(foo.Foo):\n          foo = ...  # type: Annotated[Union[int, str], 'property']\n      ")

    def test_callable_if_splitting(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def foo(x):\n        if callable(x):\n          return x(42)\n        else:\n          return False\n      f = lambda x: 10\n      a = foo(f)\n      b = foo(10)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      a = ...  # type: int\n      b = ...  # type: bool\n      def f(x) -> int: ...\n      def foo(x) -> Any: ...\n    ')

    def test_callable(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A():\n        def __call__(self):\n          pass\n        def foo(self):\n          pass\n        @staticmethod\n        def bar(self):\n          pass\n        @classmethod\n        def baz():\n          pass\n        @property\n        def quux(self):\n          pass\n      class B():\n        pass\n      def fun(x):\n        pass\n      obj = A()\n      # All these should be defined.\n      if callable(fun): a = 1\n      if callable(A): b = 1\n      if callable(obj): c = 1\n      if callable(obj.foo): d = 1\n      if callable(A.bar): e = 1\n      if callable(A.baz): f = 1\n      if callable(max): g = 1\n      if callable({}.setdefault): h = 1\n      if callable(hasattr): i = 1\n      if callable(callable): j = 1\n      # All these should not be defined.\n      if callable(obj.quux): w = 1\n      if callable(1): x = 1\n      if callable([]): y = 1\n      if callable(B()): z = 1\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated\n      obj = ...  # type: A\n      a = ...  # type: int\n      b = ...  # type: int\n      c = ...  # type: int\n      d = ...  # type: int\n      e = ...  # type: int\n      f = ...  # type: int\n      g = ...  # type: int\n      h = ...  # type: int\n      i = ...  # type: int\n      j = ...  # type: int\n      def fun(x) -> None: ...\n      class A:\n          quux = ...  # type: Annotated[None, 'property']\n          def __call__(self) -> None: ...\n          @staticmethod\n          def bar(self) -> None: ...\n          @classmethod\n          def baz() -> None: ...\n          def foo(self) -> None: ...\n      class B:\n          pass\n    ")

    def test_property_change(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __init__(self):\n          self.foo = 42\n        @property\n        def bar(self):\n          return self.foo\n      def f():\n        foo = Foo()\n        x = foo.bar\n        foo.foo = "hello world"\n        y = foo.bar\n        return (x, y)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated, Any, Tuple, Union\n      class Foo:\n        foo = ...  # type: Union[int, str]\n        bar = ...  # type: Annotated[int, 'property']\n        def __init__(self) -> None: ...\n      def f() -> Tuple[int, str]: ...\n    ")

    def test_different_property_instances(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(self):\n          self._foo = 42 if __random__ else "hello world"\n        @property\n        def foo(self):\n          return self._foo\n      foo1 = Foo()\n      foo2 = Foo()\n      if isinstance(foo1.foo, str):\n        x = foo2.foo.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*int'})

    def test_property_on_class(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        x = 0\n        @property\n        def foo(self):\n          return self.x\n      foo = Foo.foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Annotated\n      class Foo:\n        x: int\n        foo: Annotated[int, "property"]\n      foo: property\n    ')
if __name__ == '__main__':
    test_base.main()