"""Tests for closures."""
from pytype.tests import test_base

class ClosuresTest(test_base.BaseTest):
    """Tests for closures."""

    def test_basic_closure(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        x = 3\n        def g():\n          return x\n        return g\n      def caller():\n        return f()()\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      def f() -> Callable[[], Any]: ...\n      def caller() -> int: ...\n    ')

    def test_closure_on_arg(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        def g():\n          return x\n        return g\n      def caller():\n        return f(3)()\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      def f(x: int) -> Callable[[], Any]: ...\n      def caller() -> int: ...\n    ')

    def test_closure_with_arg(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        def g(y):\n          return x[y]\n        return g\n      def caller():\n        return f([1.0])(0)\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Callable\n      def f(x: List[float]) -> Callable[[Any], Any]: ...\n      def caller() -> float: ...\n    ')

    def test_closure_same_name(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        x = 1\n        y = 2\n        def g():\n          print(y)\n          x = "foo"\n          def h():\n            return x\n          return h\n        return g\n      def caller():\n        return f()()()\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      def f() -> Callable[[], Any]: ...\n      def caller() -> str: ...\n    ')

    def test_closures_add(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        z = x+1\n        def g(y):\n          return x+y+z\n        return g\n      def caller():\n        return f(1)(2)\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      def caller() -> int: ...\n      def f(x: int) -> Callable[[Any], Any]: ...\n    ')

    def test_closures_with_defaults(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def make_adder(x, y=13, z=43):\n        def add(q, r=11):\n          return x+y+z+q+r\n        return add\n      a = make_adder(10, 17)\n      print(a(7))\n      assert a(7) == 88\n      ')

    def test_closures_with_defaults_inference(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y=13, z=43):\n        def g(q, r=11):\n          return x+y+z+q+r\n        return g\n      def t1():\n        return f(1)(1)\n      def t2():\n        return f(1, 2)(1, 2)\n      def t3():\n        return f(1, 2, 3)(1)\n      t1()\n      t2()\n      t3()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable\n      def f(x: int, y: int=..., z: int=...) -> Callable: ...\n      def t1() -> int: ...\n      def t2() -> int: ...\n      def t3() -> int: ...\n    ')

    def test_closure_scope(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        x = ["foo"]\n        def inner():\n          x[0] = "bar"\n          return x\n        return inner\n      def g(funcptr):\n        x = 5\n        def inner():\n          return x\n        y = funcptr()\n        return y\n      def caller():\n        return g(f())\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable, List\n      def caller() -> List[str]: ...\n      def f() -> Callable[[], Any]: ...\n      def g(funcptr: Callable[[], Any]) -> List[str]: ...\n    ')

    def test_deep_closures(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f1(a):\n        b = 2*a\n        def f2(c):\n          d = 2*c\n          def f3(e):\n            f = 2*e\n            def f4(g):\n              h = 2*g\n              return a+b+c+d+e+f+g+h\n            return f4\n          return f3\n        return f2\n      answer = f1(3)(4)(5)(6)\n      print(answer)\n      assert answer == 54\n      ')

    def test_deep_closures_inference(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f1(a):\n        b = a\n        def f2(c):\n          d = c\n          def f3(e):\n            f = e\n            def f4(g):\n              h = g\n              return a+b+c+d+e+f+g+h\n            return f4\n          return f3\n        return f2\n      def caller():\n        return f1(3)(4)(5)(6)\n      caller()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      def f1(a: int) -> Callable[[Any], Any]: ...\n      def caller() -> int: ...\n    ')

    def test_no_visible_bindings(self):
        if False:
            while True:
                i = 10
        self.Check("\n      def foo():\n        name = __any_object__\n        def msg():\n          return name\n        while True:\n          if __random__:\n            name = __any_object__\n            raise ValueError(msg())\n          else:\n            break\n        if __random__:\n          return {'': name}\n        return {'': name}\n    ")

    def test_undefined_var(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      def f(param):\n        pass\n\n      def outer_fn():\n        def inner_fn():\n          f(param=yet_to_be_defined)  # name-error[e]\n        inner_fn()\n        yet_to_be_defined = 0\n    ')
        self.assertErrorRegexes(err, {'e': 'yet_to_be_defined.*not.defined'})

    def test_closures(self):
        if False:
            print('Hello World!')
        self.Check('\n      def make_adder(x):\n        def add(y):\n          return x+y\n        return add\n      a = make_adder(10)\n      print(a(7))\n      assert a(7) == 17\n      ')

    def test_closures_store_deref(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def make_adder(x):\n        z = x+1\n        def add(y):\n          return x+y+z\n        return add\n      a = make_adder(10)\n      print(a(7))\n      assert a(7) == 28\n      ')

    def test_empty_vs_deleted(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import collections\n      Foo = collections.namedtuple('Foo', 'x')\n      def f():\n        (x,) = Foo(10)  # x gets set to abstract.Empty here.\n        def g():\n          return x  # Should not raise a name-error\n    ")

    def test_closures_in_loop(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def make_fns(x):\n        fns = []\n        for i in range(x):\n          fns.append(lambda i=i: i)\n        return fns\n      fns = make_fns(3)\n      for f in fns:\n        print(f())\n      assert (fns[0](), fns[1](), fns[2]()) == (0, 1, 2)\n      ')

    def test_closure(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import ctypes\n      f = 0\n      def e():\n        global f\n        s = 0\n        f = (lambda: ctypes.foo(s))  # ctypes.foo doesn't exist\n        return f()\n      e()\n    ", report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      import ctypes\n      from typing import Any\n      def e() -> Any: ...\n      def f() -> Any: ...\n    ')

    def test_recursion(self):
        if False:
            return 10
        self.Check('\n      def f(x):\n        def g(y):\n          f({x: y})\n    ')

    def test_unbound_closure_variable(self):
        if False:
            return 10
        self.CheckWithErrors('\n      def foo():\n        def bar():\n          return tuple(xs)  # name-error\n        xs = bar()\n      foo()\n    ')

    def test_attribute_error(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f(x: int):\n        def g():\n          return x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*int'})

    def test_name_error(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      def f(x):\n        try:\n          return [g() for y in x]  # name-error\n        except:\n          return []\n        def g():\n          pass\n    ')

class ClosuresTestPy3(test_base.BaseTest):
    """Tests for closures in Python 3."""

    def test_if_split_delete_deref(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(a: int):\n        x = "hello"\n        def g():\n          nonlocal x\n          x = 42\n        if a:\n          g()\n        else:\n          return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional\n      def f(a: int) -> Optional[str]: ...\n    ')

    def test_closures_delete_deref(self):
        if False:
            while True:
                i = 10
        err = self.CheckWithErrors('\n      def f():\n        x = "hello"\n        def g():\n          nonlocal x  # force x to be stored in a closure cell\n          x = 10\n        del x\n        return x  # name-error[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Variable x', 'deleted', 'line 6']})

    def test_nonlocal(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        x = "hello"\n        def g():\n          nonlocal x\n          x = 10\n        g()\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_nonlocal_delete_deref(self):
        if False:
            for i in range(10):
                print('nop')
        err = self.CheckWithErrors('\n      def f():\n        x = True\n        def g():\n          nonlocal x\n          del x\n        g()\n        return x  # name-error[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Variable x', 'deleted', 'line 5']})

    def test_reuse_after_delete_deref(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f():\n        x = True\n        def g():\n          nonlocal x\n          del x\n        g()\n        x = 42\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_closure_annotations(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f():\n        a = 1\n        def g(x: int) -> int:\n          a  # makes sure g is a closure\n          return "hello"  # bad-return-type[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*str'})

    def test_filter_before_delete(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Optional\n      def f(x: Optional[str]):\n        if x is None:\n          raise TypeError()\n        def nested():\n          nonlocal x\n          print(x.upper())  # pytype: disable=name-error\n          del x\n        nested()\n        return x  # name-error\n    ')
if __name__ == '__main__':
    test_base.main()