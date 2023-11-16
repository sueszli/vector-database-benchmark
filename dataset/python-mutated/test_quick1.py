"""Tests for --quick."""
from pytype.pytd import escape
from pytype.tests import test_base

class QuickTest(test_base.BaseTest):
    """Tests for --quick."""

    def test_max_depth(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        def __init__(self, elements):\n          assert all(e for e in elements)\n          self.elements = elements\n\n        def bar(self):\n          return self.elements\n    ', quick=True)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        elements = ...  # type: Any\n        def __init__(self, elements: Any) -> None: ...\n        def bar(self) -> Any: ...\n    ')

    def test_arg_unknowns(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return 42\n    ', quick=True, show_library_calls=True)
        f = ty.Lookup('f')
        self.assertEqual(len(f.signatures), 1)
        s = f.signatures[0]
        self.assertEqual(len(s.params), 1)
        p = s.params[0]
        self.assertTrue(escape.is_unknown(p.type.name))
        _ = ty.Lookup(p.type.name)

    def test_closure(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        class A: pass\n        return {A: A()}\n    ', quick=True, maximum_depth=1)
        self.assertTypesMatchPytd(ty, '\n      def f() -> dict: ...\n    ')

    def test_init(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.real_init()\n        def real_init(self):\n          self.x = 42\n        def f(self):\n          return self.x\n      def f():\n        return A().f()\n    ', quick=True, maximum_depth=2)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n        def real_init(self) -> None: ...\n        def f(self) -> int: ...\n      def f() -> Any: ...\n    ')

    def test_analyze_annotated_max_depth(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors("\n      def make_greeting(user_id):\n        return 'hello, user' + user_id  # unsupported-operands[e]\n      def print_greeting():\n        print(make_greeting(0))\n    ", quick=True)
        self.assertErrorRegexes(errors, {'e': 'str.*int'})
if __name__ == '__main__':
    test_base.main()