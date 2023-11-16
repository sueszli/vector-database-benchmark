"""Basic tests."""
from pytype.tests import test_base

class TestBasic(test_base.BaseTest):
    """Basic tests."""

    def test_constant(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('17')

    def test_for_loop(self):
        if False:
            while True:
                i = 10
        self.Check('\n      out = ""\n      for i in range(5):\n        out = out + str(i)\n      print(out)\n      ')

    def test_inplace_operators(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Check, '\n      x, y = 2, 3\n      x **= y\n      assert x == 8 and y == 3\n      x *= y\n      assert x == 24 and y == 3\n      x //= y\n      assert x == 8 and y == 3\n      x %= y\n      assert x == 2 and y == 3\n      x += y\n      assert x == 5 and y == 3\n      x -= y\n      assert x == 2 and y == 3\n      x <<= y\n      assert x == 16 and y == 3\n      x >>= y\n      assert x == 2 and y == 3\n\n      x = 0x8F\n      x &= 0xA5\n      assert x == 0x85\n      x |= 0x10\n      assert x == 0x95\n      x ^= 0x33\n      assert x == 0xA6\n      ')

    def test_inplace_division(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Check, '\n      x, y = 24, 3\n      x /= y\n      assert x == 8 and y == 3\n      assert isinstance(x, int)\n      x /= y\n      assert x == 2 and y == 3\n      assert isinstance(x, int)\n      ')

    def test_slice(self):
        if False:
            return 10
        ty = self.Infer('\n      s = "hello, world"\n      def f1():\n        return s[3:8]\n      def f2():\n        return s[:8]\n      def f3():\n        return s[3:]\n      def f4():\n        return s[:]\n      def f5():\n        return s[::-1]\n      def f6():\n        return s[3:8:2]\n      ', show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n    s = ...  # type: str\n    def f1() -> str: ...\n    def f2() -> str: ...\n    def f3() -> str: ...\n    def f4() -> str: ...\n    def f5() -> str: ...\n    def f6() -> str: ...\n    ')

    def test_slice_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      l = list(range(10))\n      l[3:8] = ["x"]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      l[:8] = ["x"]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      l[3:] = ["x"]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      l[:] = ["x"]\n      print(l)\n      ')

    def test_slice_deletion(self):
        if False:
            print('Hello World!')
        self.Check('\n      l = list(range(10))\n      del l[3:8]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      del l[:8]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      del l[3:]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      del l[:]\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      del l[::2]\n      print(l)\n      ')

    def test_building_stuff(self):
        if False:
            while True:
                i = 10
        self.Check('\n      print((1+1, 2+2, 3+3))\n      ')
        self.Check('\n      print([1+1, 2+2, 3+3])\n      ')
        self.Check('\n      print({1:1+1, 2:2+2, 3:3+3})\n      ')

    def test_subscripting(self):
        if False:
            print('Hello World!')
        self.Check('\n      l = list(range(10))\n      print("%s %s %s" % (l[0], l[3], l[9]))\n      ')
        self.Check('\n      l = list(range(10))\n      l[5] = 17\n      print(l)\n      ')
        self.Check('\n      l = list(range(10))\n      del l[5]\n      print(l)\n      ')

    def test_generator_expression(self):
        if False:
            print('Hello World!')
        self.Check('\n      x = "-".join(str(z) for z in range(5))\n      assert x == "0-1-2-3-4"\n      ')

    def test_generator_expression2(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from textwrap import fill\n      x = set(['test_str'])\n      width = 70\n      indent = 4\n      blanks = ' ' * indent\n      res = fill(' '.join(str(elt) for elt in sorted(x)), width,\n            initial_indent=blanks, subsequent_indent=blanks)\n      print(res)\n      ")

    def test_list_comprehension(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = [z*z for z in range(5)]\n      assert x == [0, 1, 4, 9, 16]\n      ')

    def test_dict_comprehension(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = {z:z*z for z in range(5)}\n      assert x == {0:0, 1:1, 2:4, 3:9, 4:16}\n      ')

    def test_set_comprehension(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = {z*z for z in range(5)}\n      assert x == {0, 1, 4, 9, 16}\n      ')

    def test_list_slice(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      [1, 2, 3][1:2]\n      ')

    def test_strange_sequence_ops(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      x = [1,2]\n      x += [3,4]\n      x *= 2\n\n      assert x == [1, 2, 3, 4, 1, 2, 3, 4]\n\n      x = [1, 2, 3]\n      y = x\n      x[1:2] *= 2\n      y[1:2] += [1]\n\n      assert x == [1, 2, 1, 2, 3]\n      assert x is y\n      ')

    def test_unary_operators(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = 8\n      print(-x, ~x, not x)\n      ')

    def test_attributes(self):
        if False:
            print('Hello World!')
        self.Check('\n      l = lambda: 1   # Just to have an object...\n      l.foo = 17\n      print(hasattr(l, "foo"), l.foo)\n      del l.foo\n      print(hasattr(l, "foo"))\n      ')

    def test_attribute_inplace_ops(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      l = lambda: 1   # Just to have an object...\n      l.foo = 17\n      l.foo -= 3\n      print(l.foo)\n      ')

    def test_deleting_names(self):
        if False:
            return 10
        (_, err) = self.InferWithErrors('\n      g = 17\n      assert g == 17\n      del g\n      g  # name-error[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Variable g', 'deleted', 'line 3']})

    def test_deleting_local_names(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      def f():\n        l = 23\n        assert l == 23\n        del l\n        l  # name-error\n      f()\n    ')

    def test_import(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import math\n      print(math.pi, math.e)\n      from math import sqrt\n      print(sqrt(2))\n      from math import *\n      print(sin(2))\n      ')

    def test_classes(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Thing:\n        def __init__(self, x):\n          self.x = x\n        def meth(self, y):\n          return self.x * y\n      thing1 = Thing(2)\n      thing2 = Thing(3)\n      print(thing1.x, thing2.x)\n      print(thing1.meth(4), thing2.meth(5))\n      ')

    def test_class_mros(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A: pass\n      class B(A): pass\n      class C(A): pass\n      class D(B, C): pass\n      class E(C, B): pass\n      print([c.__name__ for c in D.__mro__])\n      print([c.__name__ for c in E.__mro__])\n      ')

    def test_class_mro_method_calls(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      class A:\n        def f(self): return 'A'\n      class B(A): pass\n      class C(A):\n        def f(self): return 'C'\n      class D(B, C): pass\n      print(D().f())\n      ")

    def test_calling_methods_wrong(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Thing:\n        def __init__(self, x):\n          self.x = x\n        def meth(self, y):\n          return self.x * y\n      thing1 = Thing(2)\n      print(Thing.meth(14))  # missing-parameter[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'self'})

    def test_calling_subclass_methods(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Thing:\n        def foo(self):\n          return 17\n\n      class SubThing(Thing):\n        pass\n\n      st = SubThing()\n      print(st.foo())\n      ')

    def test_other_class_methods(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Thing:\n        def foo(self):\n          return 17\n\n      class SubThing:\n        def bar(self):\n          return 9\n\n      st = SubThing()\n      print(st.foo())  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo.*SubThing'})

    def test_attribute_access(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Thing:\n        z = 17\n        def __init__(self):\n          self.x = 23\n      t = Thing()\n      print(Thing.z)\n      print(t.z)\n      print(t.x)\n      ')

    def test_attribute_access_error(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      class Thing:\n        z = 17\n        def __init__(self):\n          self.x = 23\n      t = Thing()\n      print(t.xyzzy)  # attribute-error[e]\n      ')
        self.assertErrorRegexes(errors, {'e': 'xyzzy.*Thing'})

    def test_staticmethods(self):
        if False:
            return 10
        self.Check('\n      class Thing:\n        @staticmethod\n        def smeth(x):\n          print(x)\n        @classmethod\n        def cmeth(cls, x):\n          print(x)\n\n      Thing.smeth(1492)\n      Thing.cmeth(1776)\n      ')

    def test_unbound_methods(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Thing:\n        def meth(self, x):\n          print(x)\n      m = Thing.meth\n      m(Thing(), 1815)\n      ')

    def test_callback(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def lcase(s):\n        return s.lower()\n      l = ["xyz", "ABC"]\n      l.sort(key=lcase)\n      print(l)\n      assert l == ["ABC", "xyz"]\n      ')

    def test_unpacking(self):
        if False:
            while True:
                i = 10
        self.Check('\n      a, b, c = (1, 2, 3)\n      assert a == 1\n      assert b == 2\n      assert c == 3\n      ')

    def test_jump_if_true_or_pop(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def f(a, b):\n        return a or b\n      assert f(17, 0) == 17\n      assert f(0, 23) == 23\n      assert f(0, "") == ""\n      ')

    def test_jump_if_false_or_pop(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(a, b):\n        return not(a and b)\n      assert f(17, 0) is True\n      assert f(0, 23) is True\n      assert f(0, "") is True\n      assert f(17, 23) is False\n      ')

    def test_pop_jump_if_true(self):
        if False:
            print('Hello World!')
        self.Check("\n      def f(a):\n        if not a:\n          return 'foo'\n        else:\n          return 'bar'\n      assert f(0) == 'foo'\n      assert f(1) == 'bar'\n      ")

    def test_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def verbose(func):\n        def _wrapper(*args, **kwargs):\n          return func(*args, **kwargs)\n        return _wrapper\n\n      @verbose\n      def add(x, y):\n        return x+y\n\n      add(7, 3)\n      ')

    def test_multiple_classes(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class A:\n        def __init__(self, a, b, c):\n          self.sum = a + b + c\n\n      class B:\n        def __init__(self, x):\n          self.x = x\n\n      a = A(1, 2, 3)\n      b = B(7)\n      print(a.sum)\n      print(b.x)\n      ')

    def test_global(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      foobar = False\n      def baz():\n        global foobar\n        foobar = True\n      baz()\n      assert(foobar)\n      ')

    def test_delete_global(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      a = 3\n      def f():\n        global a\n        del a\n      f()\n      x = a  # name-error\n      ')

    def test_string(self):
        if False:
            return 10
        self.Check("v = '\\xff'")

    def test_string2(self):
        if False:
            return 10
        self.Check("v = '\\uD800'")

    def test_del_after_listcomp(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def foo(x):\n        num = 1\n        nums = [num for _ in range(2)]\n        del num\n    ')

class TestLoops(test_base.BaseTest):
    """Loop tests."""

    def test_for(self):
        if False:
            while True:
                i = 10
        self.Check('\n      for i in range(10):\n        print(i)\n      print("done")\n      ')

    def test_break(self):
        if False:
            print('Hello World!')
        self.Check('\n      for i in range(10):\n        print(i)\n        if i == 7:\n          break\n      print("done")\n      ')

    def test_continue(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      for i in range(10):\n        if i % 3 == 0:\n          continue\n        print(i)\n      print("done")\n      ')

    def test_continue_in_try_except(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      for i in range(10):\n        try:\n          if i % 3 == 0:\n            continue\n          print(i)\n        except ValueError:\n          pass\n      print("done")\n      ')

    def test_continue_in_try_finally(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      for i in range(10):\n        try:\n          if i % 3 == 0:\n            continue\n          print(i)\n        finally:\n          print(".")\n      print("done")\n      ')

class TestComparisons(test_base.BaseTest):
    """Comparison tests."""

    def test_in(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      assert "x" in "xyz"\n      assert "x" not in "abc"\n      assert "x" in ("x", "y", "z")\n      assert "x" not in ("a", "b", "c")\n      ')

    def test_less(self):
        if False:
            print('Hello World!')
        self.Check('\n      assert 1 < 3\n      assert 1 <= 2 and 1 <= 1\n      assert "a" < "b"\n      assert "a" <= "b" and "a" <= "a"\n      ')

    def test_greater(self):
        if False:
            while True:
                i = 10
        self.Check('\n      assert 3 > 1\n      assert 3 >= 1 and 3 >= 3\n      assert "z" > "a"\n      assert "z" >= "a" and "z" >= "z"\n      ')

class TestSlices(test_base.BaseTest):

    def test_slice_with_step(self):
        if False:
            while True:
                i = 10
        self.Check('\n      [0][1:-2:2]\n      ')

    def test_slice_on_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      __any_object__[1:-2:2]\n      ')
if __name__ == '__main__':
    test_base.main()