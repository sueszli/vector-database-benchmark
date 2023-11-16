"""Tests for super()."""
from pytype.tests import test_base
from pytype.tests import test_utils

class SuperTest(test_base.BaseTest):
    """Tests for super()."""

    def test_set_attr(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def foo(self, name, value):\n          super(Foo, self).__setattr__(name, value)\n    ')

    def test_str(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def foo(self, name, value):\n          super(Foo, self).__str__()\n    ')

    def test_get(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def foo(self, name, value):\n          super(Foo, self).__get__(name)\n    ')

    def test_inherited_get(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __get__(self, obj, objtype):\n          return 42\n      class Bar(Foo):\n        def __get__(self, obj, objtype):\n          return super(Bar, self).__get__(obj, objtype)\n      class Baz:\n        x = Bar()\n      Baz().x + 1\n    ')

    def test_inherited_get_grandparent(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def __get__(self, obj, objtype):\n          return 42\n      class Mid(Foo):\n        pass\n      class Bar(Mid):\n        def __get__(self, obj, objtype):\n          return super(Bar, self).__get__(obj, objtype)\n      class Baz:\n        x = Bar()\n      Baz().x + 1\n    ')

    def test_inherited_get_multiple(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __get__(self, obj, objtype):\n          return 42\n      class Quux:\n        pass\n      class Bar(Quux, Foo):\n        def __get__(self, obj, objtype):\n          return super(Bar, self).__get__(obj, objtype)\n      class Baz:\n        x = Bar()\n      Baz().x + 1\n    ')

    def test_set(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def foo(self, name, value):\n          super(Foo, self).__set__(name, value)  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '__set__.*super'})

    def test_inherited_set(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def __init__(self):\n          self.foo = 1\n        def __set__(self, name, value):\n          self.foo = value\n      class Bar(Foo):\n        def __set__(self, name, value):\n          super(Bar, self).__set__(name, value)\n      class Baz():\n        x = Bar()\n      y = Baz()\n      y.x = 42\n    ')

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def foo(self, name, value):\n          super(Foo, self).__init__()\n    ')

    def test_getattr(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def hello(self, name):\n          getattr(super(Foo, self), name)\n    ')

    def test_getattr_multiple_inheritance(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class X:\n        pass\n\n      class Y:\n        bla = 123\n\n      class Foo(X, Y):\n        def hello(self):\n          getattr(super(Foo, self), "bla")\n    ')

    def test_getattr_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Y:\n        bla = 123\n\n      class Foo(Y):\n        def hello(self):\n          getattr(super(Foo, self), "bla")\n    ')

    def test_isinstance(self):
        if False:
            return 10
        self.Check('\n      class Y:\n        pass\n\n      class Foo(Y):\n        def hello(self):\n          return isinstance(super(Foo, self), Y)\n    ')

    def test_call_super(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errorlog) = self.InferWithErrors('\n      class Y:\n        pass\n\n      class Foo(Y):\n        def hello(self):\n          return super(Foo, self)()  # not-callable[e]\n    ')
        self.assertErrorRegexes(errorlog, {'e': 'super'})

    def test_super_type(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        pass\n      x = super(type, A)\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        pass\n      x = ...  # type: super\n    ')

    def test_super_with_ambiguous_base(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Grandparent:\n          def f(self) -> int: ...\n      ')
            ty = self.Infer('\n        import foo\n        class Parent(foo.Grandparent):\n          pass\n        OtherParent = __any_object__\n        class Child(OtherParent, Parent):\n          def f(self):\n            return super(Parent, self).f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        class Parent(foo.Grandparent): ...\n        OtherParent = ...  # type: Any\n        class Child(Any, Parent):\n          def f(self) -> int: ...\n      ')

    def test_super_with_any(self):
        if False:
            while True:
                i = 10
        self.Check('\n      super(__any_object__, __any_object__)\n    ')

    def test_single_argument_super(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      super(object)\n      super(object())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'cls: type.*cls: object'})

    def test_method_on_single_argument_super(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      sup = super(object)\n      sup.foo  # attribute-error[e1]\n      sup.__new__(object)  # wrong-arg-types[e2]\n      v = sup.__new__(super)\n    ')
        self.assertTypesMatchPytd(ty, '\n      sup = ...  # type: super\n      v = ...  # type: super\n    ')
        self.assertErrorRegexes(errors, {'e1': "'foo' on super", 'e2': 'Type\\[super\\].*Type\\[object\\]'})

    def test_super_under_decorator(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def decorate(cls):\n        return __any_object__\n      class Parent:\n        def Hello(self):\n          pass\n      @decorate\n      class Child(Parent):\n        def Hello(self):\n          return super(Child, self).Hello()\n    ')

    def test_super_set_attr(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __init__(self):\n          super(Foo, self).foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'super'})

    def test_super_subclass_set_attr(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Foo: pass\n      class Bar(Foo):\n        def __init__(self):\n          super(Bar, self).foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'super'})

    def test_super_nothing_set_attr(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo(nothing): ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        class Bar(foo.Foo):\n          def __init__(self):\n            super(foo.Foo, self).foo = 42  # not-writable[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'super'})

    def test_super_any_set_attr(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      class Foo(__any_object__):\n        def __init__(self):\n          super(Foo, self).foo = 42  # not-writable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'super'})

    @test_base.skip('pytype thinks the two Foo classes are the same')
    def test_duplicate_class_names(self):
        if False:
            return 10
        self.Check("\n      class Foo:\n        def __new__(self, *args, **kwargs):\n          typ = type('Foo', (Foo,), {})\n          return super(Foo, typ).__new__(typ)\n        def __init__(self, x):\n          super(Foo, self).__init__()\n    ")
if __name__ == '__main__':
    test_base.main()