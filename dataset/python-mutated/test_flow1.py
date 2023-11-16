"""Tests for control flow (with statements, loops, exceptions, etc.)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class FlowTest(test_base.BaseTest):
    """Tests for control flow.

  These tests primarily test instruction ordering and CFG traversal of the
  bytecode interpreter, i.e., their primary focus isn't the inferred types.
  Even though they check the validity of the latter, they're mostly smoke tests.
  """

    def test_if(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      if __random__:\n        x = 3\n      else:\n        x = 3.1\n    ', deep=False, show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      x = ...  # type: Union[int, float]\n    ')

    def test_exception(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        try:\n          x = UndefinedName()\n        except Exception:\n          return 3\n      f()\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, 'def f() -> int | None: ...')

    def test_two_except_handlers(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        try:\n          x = UndefinedName()\n        except Exception:\n          return 3\n        except:\n          return 3.5\n      f()\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, 'def f() -> int | float | None: ...')

    def test_nested_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        try:\n          try:\n            UndefinedName()\n          except:\n            return 3\n        except:\n          return 3.5\n      f()\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, 'def f() -> int | float | None: ...')

    def test_raise(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        try:\n          try:\n            raise  # raises TypeError (exception not derived from BaseException)\n          except:\n            return 3\n        except:\n          return 3.5\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int | float: ...')

    def test_finally(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        try:\n          x = RaiseANameError()\n        finally:\n          return 3\n      f()\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_finally_suffix(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        try:\n          x = RaiseANameError()\n        finally:\n          x = 3\n        return x\n      f()\n    ', report_errors=False)
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_try_and_loop(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        for s in (1, 2):\n          try:\n            try:\n              pass\n            except:\n              continue\n          finally:\n            return 3\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int | None: ...')

    def test_simple_with(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        y = 1\n        with __any_object__:\n          y = 2\n        return x\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_nested_with(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        y = 1\n        with __any_object__:\n          y = 2\n          with __any_object__:\n            pass\n        return x\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_null_flow(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        if x is None:\n          return 0\n        return len(x)\n      f(__any_object__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x) -> int: ...\n    ')

    def test_continue_in_with(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        l = []\n        for i in range(3):\n          with __any_object__:\n            l.append(i)\n            if i % 2:\n               continue\n            l.append(i)\n          l.append(i)\n        return l\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> list[int]: ...')

    def test_break_in_with(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def f():\n        l = []\n        for i in range(3):\n          with __any_object__:\n            l.append('w')\n            if i % 2:\n               break\n            l.append('z')\n          l.append('e')\n        l.append('r')\n        s = ''.join(l)\n        return s\n      f()\n    ")
        self.assertTypesMatchPytd(ty, 'def f() -> str: ...')

    @test_utils.skipIfPy((3, 8), reason='Broken in 3.8')
    def test_raise_in_with(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        l = []\n        try:\n          with __any_object__:\n            l.append(\'w\')\n            raise ValueError("oops")\n            l.append(\'z\')\n          l.append(\'e\')\n        except ValueError as e:\n          assert str(e) == "oops"\n          l.append(\'x\')\n        l.append(\'r\')\n        s = \'\'.join(l)\n        return s\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> str: ...')

    def test_return_in_with(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        with __any_object__:\n          return "foo"\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> str: ...')

    def test_dead_if(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = None\n      if x is not None:\n        x.foo()\n    ')
        self.Check('\n      x = 1\n      if x is not 1:\n        x.foo()\n    ')

    def test_return_after_loop(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        x = g()\n        return x\n\n      def g():\n        while True:\n          pass\n        return 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f() -> Any: ...\n      def g() -> Any: ...\n    ')

    def test_change_boolean(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        b = True\n        while b:\n          b = False\n    ')
        if self.python_version >= (3, 10):
            expected_return_type = 'None'
        else:
            expected_return_type = 'Any'
        self.assertTypesMatchPytd(ty, f'\n      from typing import Any\n      def f() -> {expected_return_type}: ...\n    ')

    def test_independent_calls(self):
        if False:
            return 10
        ty = self.Infer('\n      class _Item:\n        def __init__(self, stack):\n          self.name = "foo"\n          self.name_list = [s.name for s in stack]\n      def foo():\n        stack = []\n        if __random__:\n          stack.append(_Item(stack))\n        else:\n          stack.append(_Item(stack))\n    ')
        self.assertTypesMatchPytd(ty, '\n      class _Item:\n        name = ...  # type: str\n        name_list = ...  # type: list\n        def __init__(self, stack) -> None: ...\n      def foo() -> None: ...\n    ')

    def test_duplicate_getproperty(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      class Foo:\n        def __init__(self):\n          self._node = __any_object__\n        def bar(self):\n          if __random__:\n            raise Exception(\n            'No node with type %s could be extracted.' % self._node)\n      Foo().bar()\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        _node = ...  # type: Any\n        def __init__(self) -> None: ...\n        def bar(self) -> NoneType: ...\n    ')

    def test_break(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def _foo():\n        while True:\n          if __random__:\n            break\n        return 3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      def _foo() -> complex: ...\n    ')

    def test_continue(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def bar():\n        while True:\n          if __random__:\n            return 3j\n          continue\n          return 3  # dead code\n    ')
        self.assertTypesMatchPytd(ty, '\n      def bar() -> complex: ...\n    ')

    def test_loop_over_list_of_lists(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n    for seq in [[1, 2, 3]]:\n        seq.append("foo")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      seq = ...  # type: List[Union[int, str]]\n    ')

    def test_call_undefined(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      def f():\n        try:\n          func = None\n        except:\n          func()  # name-error[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'func'})

    def test_nested_break(self):
        if False:
            return 10
        self.assertNoCrash(self.Infer, '\n      while True:\n        try:\n          pass\n        except:\n          break\n        while True:\n          try:\n            pass\n          except:\n            break\n    ')

    def test_nested_break2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNoCrash(self.Infer, '\n      while True:\n        for x in []:\n          pass\n        break\n    ')

    def test_loop_after_break(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNoCrash(self.Infer, '\n      for _ in ():\n        break\n      else:\n        raise\n      for _ in ():\n        break\n      else:\n        raise\n    ')

    def test_recursion(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      b = True\n      def f():\n        if b:\n          g()\n      def g():\n        global b\n        b = False\n        f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      b = ...  # type: bool\n      def f() -> None: ...\n      def g() -> None: ...\n    ')

    def test_deleted(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      def bar(y):\n        return y*y\n\n      def foo(x):\n        del x\n        y = x.y()  # name-error\n        return bar(y)\n    ')
if __name__ == '__main__':
    test_base.main()