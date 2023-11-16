"""Tests for super()."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestSuperPython3Feature(test_base.BaseTest):
    """Tests for super()."""

    def test_super_without_args(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Callable\n      class A:\n        def m_a(self, x: int, y: int) -> int:\n          return x + y\n      class B(A):\n        def m_b(self, x: int, y: int) -> int:\n          return super().m_a(x, y)\n      b = B()\n      i = b.m_b(1, 2)\n      class C(A):\n        def m_c(self, x: int, y: int) -> Callable[["C"], int]:\n          def f(this: "C") -> int:\n            return super().m_a(x, y)\n          return f\n      def call_m_c(c: C, x: int, y: int) -> int:\n        f = c.m_c(x, y)\n        return f(c)\n      i = call_m_c(C(), i, i + 1)\n      def make_my_c() -> C:\n        class MyC(C):\n          def m_c(self, x: int, y: int) -> Callable[[C], int]:\n            def f(this: C) -> int:\n              super_f = super().m_c(x, y)\n              return super_f(self)\n            return f\n        return MyC()\n      def call_my_c(x: int, y: int) -> int:\n        c = make_my_c()\n        f = c.m_c(x, y)\n        return f(c)\n      i = call_my_c(i, i + 2)\n      class Outer:\n        class InnerA(A):\n          def m_a(self, x: int, y: int) -> int:\n            return 2 * super().m_a(x, y)\n      def call_inner(a: Outer.InnerA) -> int:\n        return a.m_a(1, 2)\n      i = call_inner(Outer.InnerA())\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Callable\n    class A:\n      def m_a(self, x: int, y: int) -> int: ...\n    class B(A):\n      def m_b(self, x: int, y: int) -> int: ...\n    class C(A):\n      def m_c(self, x: int, y: int) -> Callable[[C], int]: ...\n    def call_m_c(c: C, x: int, y: int) -> int: ...\n    def make_my_c() -> C: ...\n    def call_my_c(x: int, y: int) -> int: ...\n    class Outer:\n      class InnerA(A):\n        def m_a(self, x: int, y: int) -> int: ...\n    def call_inner(a: Outer.InnerA) -> int: ...\n    b = ...  # type: B\n    i = ...  # type: int\n    ')

    def test_super_without_args_error(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      class A:\n        def m(self):\n          pass\n      class B(A):\n        def m(self):\n          def f():\n            super().m()  # invalid-super-call[e1]\n          f()\n      def func(x: int):\n        super().m()  # invalid-super-call[e2]\n      ')
        self.assertErrorRegexes(errors, {'e1': ".*Missing 'self' argument.*", 'e2': '.*Missing __class__ closure.*'})

    def test_mixin(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Mixin:\n        def __init__(self, x, **kwargs):\n          super().__init__(**kwargs)\n          self.x = x\n\n      class Foo:\n        def __init__(self, y):\n          self.y = y\n\n      class Bar(Mixin, Foo):\n        def __init__(self, x, y):\n          return super().__init__(x=x, y=y)\n    ')

    def test_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import abc\n\n      class Foo(metaclass=abc.ABCMeta):\n        pass\n\n      class Bar(Foo):\n        def __new__(cls):\n          return super().__new__(cls)\n    ')

    def test_metaclass(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Meta(type): ...\n        class Foo(metaclass=Meta):\n          @classmethod\n          def hook(cls): ...\n      ')
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          @classmethod\n          def hook(cls):\n            return super().hook()\n        class Baz(Bar):\n          @classmethod\n          def hook(cls):\n            return super().hook()\n      ', pythonpath=[d.path])

    def test_metaclass_calling_super(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Meta(type):\n        def __init__(cls, name, bases, dct):\n          super(Meta, cls).__init__(name, bases, dct)\n          cls.hook()  # pytype: disable=attribute-error\n      class Foo(metaclass=Meta):\n        @classmethod\n        def hook(cls):\n          pass\n      class Bar(Foo):\n        @classmethod\n        def hook(cls):\n          super(Bar, cls).hook()  # pytype: disable=name-error\n    ')

    def test_generic_class(self):
        if False:
            return 10
        self.Check("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        pass\n      class Bar(Foo[T]):\n        def __init__(self):\n          super().__init__()\n      class Baz(Bar[T]):\n        pass\n    ")

    def test_nested_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Parent1:\n        def hook(self):\n          pass\n      class Parent2(Parent1):\n        pass\n      def _BuildChild(parent):\n        class Child(parent):\n          def hook(self):\n            return super().hook()\n        return Child\n      Child1 = _BuildChild(Parent1)\n      Child2 = _BuildChild(Parent2)\n    ')

    def test_namedtuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import NamedTuple\n      class Foo(NamedTuple('Foo', [('x', int)])):\n        def replace(self, *args, **kwargs):\n          return super()._replace(*args, **kwargs)\n    ")

    def test_list_comprehension(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def f(self) -> int:\n          return 42\n\n      class Bar(Foo):\n        def f(self) -> int:\n          return [x for x in\n               [super().f() for _ in range(1)]][0]\n    ')

    def test_keyword_arg(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        def f(self):\n          return 42\n      class Bar(Foo):\n        def f(self):\n          return super().f()\n      x = Bar.f(self=Bar())\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def f(self) -> int: ...\n      class Bar(Foo):\n        def f(self) -> int: ...\n      x: int\n    ')

    def test_classmethod_inheritance_chain(self):
        if False:
            return 10
        with self.DepTree([('base.py', "\n      from typing import Type, TypeVar\n      BaseT = TypeVar('BaseT', bound='Base')\n      class Base:\n        @classmethod\n        def test(cls: Type[BaseT]) -> BaseT:\n          return cls()\n    ")]):
            self.Check('\n        import base\n        class Foo(base.Base):\n          @classmethod\n          def test(cls):\n            return super().test()\n        class Bar(Foo):\n          @classmethod\n          def test(cls):\n            return super().test()\n      ')

    def test_type_subclass_with_own_new(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      class A(type):\n        def __new__(cls) -> 'A':\n          return __any_object__\n\n      class B(A):\n        def __new__(cls):\n          C = A.__new__(cls)\n\n          def __init__(self):\n            super(C, self).__init__()\n\n          C.__init__ = __init__\n    ")
if __name__ == '__main__':
    test_base.main()