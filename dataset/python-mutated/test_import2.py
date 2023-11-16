"""Tests for import."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ImportTest(test_base.BaseTest):
    """Tests for import."""

    def test_module_attributes(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import os\n      f = os.__file__\n      n = os.__name__\n      d = os.__doc__\n      p = os.__package__\n      ')
        self.assertTypesMatchPytd(ty, '\n       import os\n       from typing import Optional\n       f = ...  # type: str\n       n = ...  # type: str\n       d = ...  # type: str\n       p = ...  # type: Optional[str]\n    ')

    def test_import_sys2(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import sys\n      import bad_import  # doesn't exist\n      def f():\n        return sys.stderr\n      def g():\n        return sys.maxsize\n      def h():\n        return sys.getrecursionlimit()\n    ", report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      import sys\n      from typing import Any, TextIO\n      bad_import = ...  # type: Any\n      def f() -> TextIO: ...\n      def g() -> int: ...\n      def h() -> int: ...\n    ')

    def test_relative_priority(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'x = ...  # type: int')
            d.create_file('b/a.pyi', 'x = ...  # type: complex')
            ty = self.Infer('\n        import a\n        x = a.x\n      ', deep=False, pythonpath=[d.path], module_name='b.main')
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n      ')

    def test_import_attribute_error(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      try:\n        import nonexistent  # import-error\n      except ImportError as err:\n        print(err.name)\n    ')

    def test_datetime_datetime(self):
        if False:
            return 10
        with self.DepTree([('foo.py', 'from datetime import datetime')]):
            self.Check('\n        import foo\n        assert_type(foo.datetime(1, 1, 1), "datetime.datetime")\n      ')

    def test_cycle(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('components.pyi', '\n      import loaders\n      from typing import Dict, Type\n      Foo: Type[loaders.Foo]\n      class Component:\n        def __init__(self, foos: Dict[int, loaders.Foo]) -> None: ...\n    '), ('loaders.pyi', '\n      from typing import Any, NamedTuple\n      Component: Any\n      class Foo(NamedTuple):\n        foo: int\n      def load() -> Any: ...\n    ')]):
            self.Infer('\n        from typing import Dict, NamedTuple\n        from components import Component\n        class Foo(NamedTuple):\n          foo: int\n        def load() -> Component:\n          foos: Dict[int, Foo] = {}\n          return Component(foos=foos)\n      ', module_name='loaders')
if __name__ == '__main__':
    test_base.main()