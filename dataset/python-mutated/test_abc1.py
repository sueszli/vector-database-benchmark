"""Tests for @abc.abstractmethod in abc_overlay.py."""
from pytype.tests import test_base
from pytype.tests import test_utils

class AbstractMethodTests(test_base.BaseTest):
    """Tests for @abc.abstractmethod."""

    def test_instantiate_pyi_abstract_class(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n        class Example(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          def foo(self) -> None: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.Example()  # not-instantiable[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'foo\\.Example.*foo'})

    def test_stray_abstractmethod(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      import abc\n      class Example:  # ignored-abstractmethod[e]\n        @abc.abstractmethod\n        def foo(self):\n          pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo.*Example'})

    def test_multiple_inheritance_implementation_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n        class Interface(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          def foo(self): ...\n        class X(Interface): ...\n        class Implementation(Interface):\n          def foo(self) -> int: ...\n        class Foo(X, Implementation): ...\n      ')
            self.Check('\n        import foo\n        foo.Foo().foo()\n      ', pythonpath=[d.path])

    def test_multiple_inheritance_error_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n        class X: ...\n        class Interface(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          def foo(self): ...\n        class Foo(X, Interface): ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.Foo().foo()  # not-instantiable[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'foo\\.Foo.*foo'})

    def test_abc_metaclass_from_decorator(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('six.pyi', "\n        from typing import TypeVar, Callable\n        T = TypeVar('T')\n        def add_metaclass(metaclass: type) -> Callable[[T], T]: ...\n      ")
            self.Check('\n        import abc\n        import six\n        @six.add_metaclass(abc.ABCMeta)\n        class Foo:\n          @abc.abstractmethod\n          def foo(self):\n            pass\n      ', pythonpath=[d.path])

    def test_abc_child_metaclass(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('six.pyi', "\n        from typing import TypeVar, Callable\n        T = TypeVar('T')\n        def add_metaclass(metaclass: type) -> Callable[[T], T]: ...\n      ")
            self.Check('\n        import abc\n        import six\n        class ABCChild(abc.ABCMeta):\n          pass\n        @six.add_metaclass(ABCChild)\n        class Foo:\n          @abc.abstractmethod\n          def foo(self):\n            pass\n      ', pythonpath=[d.path])

    def test_misplaced_abstractproperty(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      import abc\n      @abc.abstractproperty\n      class Example:\n        pass\n      Example()  # not-callable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': "'abstractproperty' object"})
if __name__ == '__main__':
    test_base.main()