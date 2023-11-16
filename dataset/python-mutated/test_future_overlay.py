"""Tests for methods in six_overlay.py."""
from pytype.tests import test_base
from pytype.tests import test_utils

class FutureUtilsTest(test_base.BaseTest):
    """Tests for future.utils and future_overlay."""

    def test_with_metaclass(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('future/utils.pyi', 'def with_metaclass(meta: type, *bases: type) -> type: ...')
            self.Check('\n          import abc\n          from future.utils import with_metaclass\n          class A:\n            def __init__(self):\n              self.foo = "hello"\n          class B:\n            def bar(self):\n              return 42\n          class Foo(with_metaclass(abc.ABCMeta, A), B):\n            @abc.abstractmethod\n            def get_foo(self):\n              pass\n          class Bar(Foo):\n            def get_foo(self):\n              return self.foo\n          x = Bar().get_foo()\n          y = Bar().bar()\n      ', pythonpath=[d.path])

    def test_missing_import(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from future.utils import iteritems  # import-error\n      from future.utils import with_metaclass  # import-error\n    ')
if __name__ == '__main__':
    test_base.main()