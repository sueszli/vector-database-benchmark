"""Tests for control flow (with statements, loops, exceptions, etc.)."""
from pytype.tests import test_base

class FlowTest(test_base.BaseTest):
    """Tests for control flow.

  These tests primarily test instruction ordering and CFG traversal of the
  bytecode interpreter, i.e., their primary focus isn't the inferred types.
  Even though they check the validity of the latter, they're mostly smoke tests.
  """

    def test_loop_and_if(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import typing\n      def foo() -> str:\n        while True:\n          y = None\n          z = None\n          if __random__:\n            y = "foo"\n            z = "foo"\n          if y:\n            return z\n        return "foo"\n    ')

    def test_cfg_cycle_singlestep(self):
        if False:
            return 10
        self.Check('\n      import typing\n      class Foo:\n        x = ...  # type: typing.Optional[int]\n        def __init__(self):\n          self.x = None\n        def X(self) -> int:\n          return self.x or 4\n        def B(self) -> None:\n          self.x = 5\n          if __random__:\n            self.x = 6\n        def C(self) -> None:\n          self.x = self.x\n    ')

    def test_unsatisfiable_in_with_block(self):
        if False:
            print('Hello World!')
        self.Check("\n      import threading\n\n      _temporaries = {}\n      _temporaries_lock = threading.RLock()\n\n      def GetResourceFilename(name: str):\n        with _temporaries_lock:\n          filename = _temporaries.get(name)\n          if filename:\n            return filename\n        return name\n\n      x = GetResourceFilename('a')\n      assert_type(x, str)\n    ")

    def test_unsatisfiable_in_except_block(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def raise_error(e):\n        raise(e)\n\n      _temporaries = {}\n\n      def f():\n        try:\n          return "hello"\n        except Exception as e:\n          filename = _temporaries.get(\'hello\')\n          if filename:\n            return filename\n          raise_error(e)\n\n      f().lower()  # f() should be str, not str|None\n    ')

    def test_finally_with_returns(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f() -> int:\n        try:\n          return 10\n        except:\n          return 42\n        finally:\n          x = None\n        return "hello world"\n      f()\n    ')
if __name__ == '__main__':
    test_base.main()