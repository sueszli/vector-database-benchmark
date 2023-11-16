"""Tests for methods in six_overlay.py."""
from pytype.tests import test_base

class SixTests(test_base.BaseTest):
    """Tests for six and six_overlay."""

    def test_six_moves_import(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import six\n      def use_range():\n        for x in six.moves.range(1, 10):\n          x\n    ')

    def test_add_metaclass(self):
        if False:
            return 10
        'Like the test in test_abc but without a fake six.pyi.'
        self.Check('\n      import abc\n      import six\n      class A:\n        def __init__(self):\n          self.foo = "hello"\n      @six.add_metaclass(abc.ABCMeta)\n      class Foo(A):\n        @abc.abstractmethod\n        def get_foo(self):\n          pass\n      class Bar(Foo):\n        def get_foo(self):\n          return self.foo\n      x = Bar().get_foo()\n    ')

    def test_with_metaclass(self):
        if False:
            return 10
        self.Check('\n      import abc\n      import six\n      class A:\n        def __init__(self):\n          self.foo = "hello"\n      class B:\n        def bar(self):\n          return 42\n      class Foo(six.with_metaclass(abc.ABCMeta, A), B):\n        @abc.abstractmethod\n        def get_foo(self):\n          pass\n      class Bar(Foo):\n        def get_foo(self):\n          return self.foo\n      x = Bar().get_foo()\n      y = Bar().bar()\n    ')

    def test_with_metaclass_any(self):
        if False:
            print('Hello World!')
        self.Check('\n      import six\n      from typing import Any\n      Meta = type  # type: Any\n      class Foo(six.with_metaclass(Meta)):\n        pass\n    ')

    def test_type_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import six\n      class Foo(type):\n        def __init__(self, *args):\n          self.x = 42\n      @six.add_metaclass(Foo)\n      class Bar:\n        pass\n      x1 = Bar.x\n      x2 = Bar().x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import six\n      class Foo(type):\n        x: int\n        def __init__(self, *args) -> None: ...\n      class Bar(object, metaclass=Foo):\n        x: int\n      x1: int\n      x2: int\n    ')
if __name__ == '__main__':
    test_base.main()