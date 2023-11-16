"""Tests of builtins.tuple."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TupleTest(test_base.BaseTest):
    """Tests for builtins.tuple."""

    def test_unpack_inline_tuple(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import Tuple\n      def f(x: Tuple[str, int]):\n        return x\n      v1, v2 = f(__any_object__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      def f(x: Tuple[str, int]) -> Tuple[str, int]: ...\n      v1 = ...  # type: str\n      v2 = ...  # type: int\n    ')

    def test_unpack_tuple_or_tuple(self):
        if False:
            print('Hello World!')
        self.Check("\n      def f():\n        if __random__:\n          return (False, 'foo')\n        else:\n          return (False, 'foo')\n      def g() -> str:\n        a, b = f()\n        return b\n    ")

    def test_unpack_tuple_or_list(self):
        if False:
            while True:
                i = 10
        self.Check("\n      def f():\n        if __random__:\n          return (False, 'foo')\n        else:\n          return ['foo', 'bar']\n      def g() -> str:\n        a, b = f()\n        return b\n    ")

    def test_unpack_ambiguous_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f() -> tuple:\n        return __any_object__\n      a, b = f()\n    ')

    def test_tuple_printing(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import Tuple\n      def f(x: Tuple[str, ...]):\n        pass\n      def g(y: Tuple[str]):\n        pass\n      f((42,))  # wrong-arg-types[e1]\n      f(tuple([42]))  # wrong-arg-types[e2]\n      f(("", ""))  # okay\n      g((42,))  # wrong-arg-types[e3]\n      g(("", ""))  # wrong-arg-types[e4]\n      g(("",))  # okay\n      g(tuple([""]))  # okay\n    ')
        x = 'Tuple\\[str, \\.\\.\\.\\]'
        y = 'Tuple\\[str\\]'
        tuple_int = 'Tuple\\[int\\]'
        tuple_ints = 'Tuple\\[int, \\.\\.\\.\\]'
        tuple_str_str = 'Tuple\\[str, str\\]'
        self.assertErrorRegexes(errors, {'e1': f'{x}.*{tuple_int}', 'e2': f'{x}.*{tuple_ints}', 'e3': f'{y}.*{tuple_int}', 'e4': f'{y}.*{tuple_str_str}'})

    def test_inline_tuple(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        class A(Tuple[int, str]): ...\n      ')
            self.Check('\n        from typing import Tuple, Type\n        import foo\n        def f(x: Type[Tuple[int, str]]):\n          pass\n        def g(x: Tuple[int, str]):\n          pass\n        f(type((1, "")))\n        g((1, ""))\n        g(foo.A())\n      ', pythonpath=[d.path])

    def test_inline_tuple_error(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        class A(Tuple[str, int]): ...\n      ')
            (_, errors) = self.InferWithErrors('\n        from typing import Tuple, Type\n        import foo\n        def f(x: Type[Tuple[int, str]]):\n          pass\n        def g(x: Tuple[int, str]):\n          pass\n        f(type(("", 1)))  # wrong-arg-types[e1]\n        g(("", 1))  # wrong-arg-types[e2]\n        g(foo.A())  # wrong-arg-types[e3]\n      ', pythonpath=[d.path])
            expected = 'Tuple\\[int, str\\]'
            actual = 'Tuple\\[str, int\\]'
            self.assertErrorRegexes(errors, {'e1': f'Type\\[{expected}\\].*Type\\[{actual}\\]', 'e2': f'{expected}.*{actual}', 'e3': '%s.*foo\\.A' % expected})

    def test_tuple_combination_explosion(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Any, Dict, List, Tuple, Union\n      AlphaNum = Union[str, int]\n      def f(x: Dict[AlphaNum, Any]) -> List[Tuple]:\n        return list(sorted((k, v) for k, v in x.items() if k in {}))\n    ')

    def test_tuple_in_container(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import List, Tuple\n      def f(l: List[Tuple[int, List[int]]]):\n        line, foo = l[0]\n        return foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Tuple, TypeVar\n      def f(l: List[Tuple[int, List[int]]]) -> List[int]: ...\n    ')

    def test_mismatched_pyi_tuple(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('bar.pyi', '\n        class Bar(tuple): ...\n      ')
            errors = self.CheckWithErrors('\n        from typing import Tuple\n        import bar\n        def foo() -> Tuple[bar.Bar, bar.Bar]:\n          return bar.Bar(None, None)  # wrong-arg-count[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '1.*3'})

    def test_count(self):
        if False:
            i = 10
            return i + 15
        self.options.tweak(strict_parameter_checks=False)
        self.Check('\n      from typing import Optional\n      def f(x: Optional[str] = None, y: Optional[str] = None):\n        return (x, y).count(None)\n      def g():\n        return (0, None).count(None)\n      def h(x):\n        return (x, 0).count(None)\n    ')

    def test_empty_pyi_tuple(self):
        if False:
            while True:
                i = 10
        foo = self.Infer('\n      from typing import Tuple\n      def f(x: Tuple[()]):\n        pass\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.CheckWithErrors('\n        from typing import Any\n        import foo\n        foo.f((Any, Any))  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_match_nothing(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        def integrate() -> Tuple[nothing, nothing]: ...\n      ')
            self.CheckWithErrors('\n        import foo\n        def f(x):\n          return x[::0, 0]  # unsupported-operands\n        def g():\n          return f(foo.integrate())\n      ', pythonpath=[d.path])

    def test_empty_tuple_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = ()\n      print(x.__class__())\n    ')

class TupleTestPython3Feature(test_base.BaseTest):
    """Tests for builtins.tuple."""

    def test_iteration(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        mytuple = (1, "foo", 3j)\n        def __getitem__(self, pos):\n          return Foo.mytuple.__getitem__(pos)\n      r = [x for x in Foo()]  # Py 3 does not leak \'x\'\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Tuple, Union\n      class Foo:\n        mytuple = ...  # type: Tuple[int, str, complex]\n        def __getitem__(self, pos: int) -> Union[int, str, complex]: ...\n      r = ...  # type: List[Union[int, str, complex]]\n    ')

    def test_bad_unpacking_with_slurp(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      a, *b, c = (1,)  # bad-unpacking[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '1 value.*2 variables'})

    def test_strptime(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import time\n      (year, month, day, hour, minute) = (\n          time.strptime('', '%m %d %Y')[0:5])\n    ")
        self.assertTypesMatchPytd(ty, '\n      import time\n      from typing import Union\n      year: int\n      month: int\n      day: int\n      hour: int\n      minute: int\n    ')

    def test_parameterize_builtins_tuple(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from __future__ import annotations\n      def f(x: tuple[int, int]):\n        pass\n      f((0,))  # wrong-arg-types\n      f((0, 0))  # ok\n    ')
if __name__ == '__main__':
    test_base.main()