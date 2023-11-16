"""Tests for control flow cases that involve the exception table in 3.11+.

Python 3.11 changed the way exceptions and some other control structures were
compiled, and in particular some of them require examining the exception table
as well as the bytecode.
"""
from pytype.tests import test_base

class TestPy311(test_base.BaseTest):
    """Tests for python 3.11 support."""

    def test_context_manager(self):
        if False:
            while True:
                i = 10
        self.Check("\n      class A:\n        def __enter__(self):\n          pass\n        def __exit__(self, a, b, c):\n          pass\n\n      lock = A()\n\n      def f() -> str:\n        path = ''\n        with lock:\n          try:\n            pass\n          except:\n            pass\n          return path\n    ")

    def test_exception_type(self):
        if False:
            print('Hello World!')
        self.Check('\n      class FooError(Exception):\n        pass\n      try:\n        raise FooError()\n      except FooError as e:\n        assert_type(e, FooError)\n    ')

    def test_try_with(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(obj, x):\n        try:\n          with __any_object__:\n            obj.get(x)\n        except:\n          pass\n    ')

    def test_try_if_with(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Any\n      import os\n      pytz: Any\n      def f():\n        tz_env = os.environ.get('TZ')\n        try:\n          if tz_env == 'localtime':\n            with open('localtime') as localtime:\n              return pytz.tzfile.build_tzinfo('', localtime)\n        except IOError:\n          return pytz.UTC\n    ")

    def test_try_finally(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import tempfile\n      dir_ = None\n      def f():\n        global dir_\n        try:\n          if dir_:\n            return dir_\n          dir_ = tempfile.mkdtemp()\n        finally:\n          print(dir_)\n    ')

    def test_nested_try_in_for(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(x):\n        for i in x:\n          fd = __any_object__\n          try:\n            try:\n              if __random__:\n                return True\n            except ValueError:\n              continue\n          finally:\n            fd.close()\n    ')

    def test_while_and_nested_try(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(p):\n        try:\n          while __random__:\n            try:\n              return p.communicate()\n            except KeyboardInterrupt:\n              pass\n        finally:\n          pass\n    ')

    def test_while_and_nested_try_2(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f():\n        i = j = 0\n        while True:\n          try:\n            try:\n              i += 1\n            finally:\n              j += 1\n          except:\n            break\n        return\n    ')

    def test_while_and_nested_try_3(self):
        if False:
            return 10
        self.Check('\n      import os\n\n      def RmDirs(dir_name):\n        try:\n          parent_directory = os.path.dirname(dir_name)\n          while parent_directory:\n            try:\n              os.rmdir(parent_directory)\n            except OSError as err:\n              pass\n            parent_directory = os.path.dirname(parent_directory)\n        except OSError as err:\n          pass\n    ')
if __name__ == '__main__':
    test_base.main()