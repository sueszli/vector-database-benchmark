"""Tests for methods."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class MethodsTest(test_base.BaseTest):
    """Tests for methods."""

    def test_flow_and_replacement_sanity(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        if x:\n          x = 42\n          y = x\n          x = 1\n        return x + 4\n      f(4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_multiple_returns(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        if x:\n          return 1\n        else:\n          return 1.5\n      f(0)\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int | float: ...')

    def test_loops_sanity(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        x = 4\n        y = -10\n        for i in range(1000):\n          x = x + (i+y)\n          y = i\n        return x\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_add_int(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x + 1\n      f(3.2)\n      f(3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: float) -> float: ...\n      @overload\n      def f(x: int) -> int: ...\n    ')

    def test_conjugate(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x, y):\n        return x.conjugate()\n      f(int(), int())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: int) -> int: ...')

    def test_class_sanity(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.x = 1\n\n        def get_x(self):\n          return self.x\n\n        def set_x(self, x):\n          self.x = x\n      a = A()\n      y = a.x\n      x1 = a.get_x()\n      a.set_x(1.2)\n      x2 = a.get_x()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      a = ...  # type: A\n      x1 = ...  # type: int\n      x2 = ...  # type: float\n      y = ...  # type: int\n      class A:\n        x = ...  # type: float\n        def __init__(self) -> None : ...\n        def get_x(self) -> Union[float, int]: ...\n        def set_x(self, x: float) -> None: ...\n    ')

    def test_boolean_op(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y):\n        return 1 < x < 10\n        return 1 > x > 10\n      f(1, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: int) -> bool: ...')

    def test_is(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(a, b):\n        return a is b\n      f(1, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(a: int, b: int) -> bool: ...')

    def test_is_not(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(a, b):\n        return a is not b\n      f(1, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(a: int, b: int) -> bool: ...')

    def test_slice(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        a, b = x\n        return (a, b)\n      f((1, 2))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: tuple[int, int]) -> tuple[int, int]: ...\n    ')

    def test_convert(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return repr(x)\n      f(1)\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x) -> str: ...')

    def test_not(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return not x\n      f(1)\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x) -> bool: ...')

    def test_positive(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        return +x\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_negative(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        return -x\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_invert(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x):\n        return ~x\n      f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_inheritance(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Base:\n        def get_suffix(self):\n            return u""\n\n      class Leaf(Base):\n        def __init__(self):\n          pass\n\n      def test():\n        l1 = Leaf()\n        return l1.get_suffix()\n\n      if __name__ == "__main__":\n        test()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Base:\n        def get_suffix(self) -> str: ...\n      class Leaf(Base):\n        def __init__(self) -> None: ...\n      def test() -> str: ...\n    ')

    def test_property(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        @property\n        def my_property(self):\n          return 1\n        def foo(self):\n          return self.my_property\n\n      def test():\n        x = A()\n        return x.foo()\n\n      test()\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated\n      class A:\n        my_property: Annotated[int, 'property']\n        def foo(self) -> int: ...\n      def test() -> int: ...\n    ")

    def test_explicit_property(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class B:\n        def _my_getter(self):\n          return 1\n        def _my_setter(self):\n          pass\n        my_property = property(_my_getter, _my_setter)\n      def test():\n        b = B()\n        b.my_property = 3\n        return b.my_property\n      test()\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated\n      class B:\n        def _my_getter(self) -> int: ...\n        def _my_setter(self) -> None: ...\n        my_property: Annotated[int, 'property']\n      def test() -> int: ...\n    ")

    def test_inherited_property(self):
        if False:
            return 10
        self.Check('\n      class A:\n        @property\n        def bar(self):\n          return 42\n      class B(A):\n        def foo(self):\n          return super(B, self).bar + 42\n    ')

    def test_error_in_property(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      class Foo:\n        @property\n        def f(self):\n          return self.nonexistent  # attribute-error\n    ')

    def test_generators(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        yield 3\n      def g():\n        for x in f():\n          return x\n      g()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def f() -> Generator[int, Any, None]: ...\n      def g() -> int | None: ...\n    ')

    def test_list_generator(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        yield 3\n      def g():\n        for x in list(f()):\n          return x\n      g()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def f() -> Generator[int, Any, None]: ...\n      def g() -> int | None: ...\n    ')

    def test_recursion(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        if __random__:\n          return f()\n        else:\n          return 3\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f() -> Any: ...\n    ')

    def test_in_not_in(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        if __random__:\n          return x in [x]\n        else:\n          return x not in [x]\n      f(3)\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x) -> bool: ...')

    def test_complex_cfg(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def g(h):\n        return 2\n      def h():\n        return 1\n      def f(x):\n        if x:\n          while x:\n            pass\n          while x:\n            pass\n          assert x\n        return g(h())\n      if __name__ == "__main__":\n        f(0)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def g(h) -> int: ...\n      def h() -> int: ...\n      def f(x) -> int: ...\n    ')

    def test_branch_and_loop_cfg(self):
        if False:
            return 10
        ty = self.Infer('\n      def g():\n          pass\n      def f():\n          if True:\n            while True:\n              pass\n            return False\n          g()\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def g() -> None: ...\n      def f() -> Any: ...\n    ')

    def test_closure(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n       def f(x, y):\n         closure = lambda: x + y\n         return closure()\n       f(1, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: int) -> int: ...')

    def test_deep_closure(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n       def f():\n         x = 3\n         def g():\n           def h():\n             return x\n           return h\n         return g()()\n       f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_two_closures(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n       def f():\n         def g():\n           return 3\n         def h():\n           return g\n         return h()()\n       f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int: ...')

    def test_closure_binding_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n       def f(x):\n         y = 1\n         def g(z):\n           return x + y + z\n         return g(1)\n       f(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_closure_on_multi_type(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        if __random__:\n          x = 1\n        else:\n          x = 3.5\n        return (lambda: x)()\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f() -> int | float: ...')

    def test_call_kwargs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x, y=3):\n        return x + y\n      f(40, **{"y": 2})\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: int = ...) -> int: ...')

    def test_call_args(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return x\n      args = (3,)\n      f(*args)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: int) -> int: ...\n      args: tuple[int]\n    ')

    def test_call_args_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        return x\n      args = (3,)\n      kwargs = {}\n      f(*args, **kwargs)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: int) -> int: ...\n      args: tuple[int]\n      kwargs: dict[nothing, nothing]\n    ')

    def test_call_positional_as_keyword(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(named):\n        return named\n      f(named=3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(named: int) -> int: ...')

    def test_two_keywords(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x, y):\n        return x if x else y\n      f(x=3, y=4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: int) -> int: ...')

    def test_two_distinct_keyword_params(self):
        if False:
            for i in range(10):
                print('nop')
        f = '\n      def f(x, y):\n        return x if x else y\n    '
        ty = self.Infer(f + '\n      f(x=3, y="foo")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: str) -> int: ...')
        ty = self.Infer(f + '\n      f(y="foo", x=3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int, y: str) -> int: ...')

    def test_starstar(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(x):\n        return x\n      f(**{"x": 3})\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(x: int) -> int: ...')

    def test_starstar2(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def f(x):\n        return x\n      kwargs = {}\n      kwargs['x'] = 3\n      f(**kwargs)\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: int) -> int: ...\n      kwargs: dict[str, int]\n    ')

    def test_starstar3(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x):\n        return x\n      kwargs = dict(x=3)\n      f(**kwargs)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: int) -> int: ...\n      kwargs: dict[str, int]\n    ')

    def test_starargs_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(*args, **kwds):\n        return args\n      f(3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(*args, **kwds) -> tuple[int]: ...')

    def test_starargs_type2(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(nr, *args):\n        return args\n      f("foo", 4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(nr: str, *args) -> tuple[int]: ...')

    def test_starargs_deep(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(*args):\n        return args\n      def g(x, *args):\n        return args\n      def h(x, y, *args):\n        return args\n    ')
        self.assertTypesMatchPytd(ty, '\n    def f(*args) -> tuple: ...\n    def g(x, *args) -> tuple: ...\n    def h(x, y, *args) -> tuple: ...\n    ')

    def test_starargs_pass_through(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __init__(self, *args, **kwargs):\n          super(Foo, self).__init__(*args, **kwargs)\n    ')
        self.assertTypesMatchPytd(ty, '\n    class Foo:\n      def __init__(self, *args, **kwargs) -> NoneType: ...\n    ')

    def test_empty_starargs_type(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(nr, *args):\n        return args\n      f(3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, 'def f(nr: int, *args) -> tuple[()]: ...')

    def test_starstar_kwargs_type(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(*args, **kwargs):\n        return kwargs\n      f(foo=3, bar=4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(*args, **kwargs) -> dict[str, int]: ...\n    ')

    def test_starstar_kwargs_type2(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x, y, **kwargs):\n        return kwargs\n      f("foo", "bar", z=3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(x: str, y: str, **kwargs) -> dict[str, int]: ...\n    ')

    def test_empty_starstar_kwargs_type(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(nr, **kwargs):\n        return kwargs\n      f(3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(nr: int, **kwargs) -> dict[nothing, nothing]: ...\n    ')

    def test_starstar_deep(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __init__(self, **kwargs):\n          self.kwargs = kwargs\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    class Foo:\n      def __init__(self, **kwargs) -> NoneType: ...\n      kwargs = ...  # type: dict[str, Any]\n    ')

    def test_starstar_deep2(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(**kwargs):\n        return kwargs\n      def g(x, **kwargs):\n        return kwargs\n      def h(x, y, **kwargs):\n        return kwargs\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f(**kwargs) -> dict[str, Any]: ...\n      def g(x, **kwargs) -> dict[str, Any]: ...\n      def h(x, y, **kwargs) -> dict[str, Any]: ...\n    ')

    def test_builtin_starargs(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('myjson.pyi', '\n        from typing import Any\n        def loads(s: str, encoding: Any = ...) -> Any: ...\n      ')
            ty = self.Infer('\n        import myjson\n        def f(*args):\n          return myjson.loads(*args)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import myjson\n        from typing import Any\n        def f(*args) -> Any: ...\n      ')

    def test_builtin_starstarargs(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('myjson.pyi', '\n        from typing import Any\n        def loads(s: str, encoding: Any = ...) -> Any: ...\n      ')
            ty = self.Infer('\n        import myjson\n        def f(**args):\n          return myjson.loads(**args)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import myjson\n        from typing import Any\n        def f(**args) -> Any: ...\n      ')

    def test_builtin_keyword(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('myjson.pyi', '\n        from typing import Any\n        def loads(s: str, encoding: Any = ...) -> Any: ...\n      ')
            ty = self.Infer('\n        import myjson\n        def f():\n          return myjson.loads(s="{}")\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import myjson\n        from typing import Any\n\n        def f() -> Any: ...\n      ')

    def test_none_or_function(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      def g():\n        return 3\n\n      def f():\n        if __random__:\n          x = None\n        else:\n          x = g\n\n        if __random__:\n          return x()  # not-callable\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional\n      def g() -> int: ...\n      def f() -> Optional[int]: ...\n    ')

    def test_define_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        @classmethod\n        def myclassmethod(*args):\n          return 3\n      def f():\n        a = A()\n        return a.myclassmethod\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable\n      class A:\n        @classmethod\n        def myclassmethod(*args) -> int: ...\n      def f() -> Callable: ...\n    ')

    def test_classmethod_smoke(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class A:\n        @classmethod\n        def mystaticmethod(x, y):\n          return x + y\n    ')

    def test_invalid_classmethod(self):
        if False:
            i = 10
            return i + 15
        (ty, err) = self.InferWithErrors('\n      def f(x):\n        return 42\n      class A:\n        @classmethod\n        @f\n        def myclassmethod(*args):  # not-callable[e]\n          return 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f(x) -> int: ...\n      class A:\n        myclassmethod: Any\n    ')
        self.assertErrorSequences(err, {'e': ['int', 'not callable', '@classmethod applied', 'not a function']})

    def test_staticmethod_smoke(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        @staticmethod\n        def mystaticmethod(x, y):\n          return x + y\n    ')

    def test_classmethod(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        @classmethod\n        def myclassmethod(cls):\n          return 3\n      def f():\n        return A().myclassmethod()\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class A:\n        @classmethod\n        def myclassmethod(cls: Type[A]) -> int: ...\n      def f() -> int: ...\n    ')

    def test_inherited_classmethod(self):
        if False:
            return 10
        self.Check('\n      class A:\n        @classmethod\n        def myclassmethod(cls):\n          return 3\n      class B(A):\n        @classmethod\n        def myclassmethod(cls):\n          return super(B, cls).myclassmethod()\n    ')

    def test_staticmethod(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        @staticmethod\n        def mystaticmethod(x, y):\n          return x + y\n      def f():\n        return A.mystaticmethod(1, 2)\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        @staticmethod\n        def mystaticmethod(x, y) -> Any: ...\n      def f() -> int: ...\n    ')

    def test_simple_staticmethod(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class MyClass:\n        @staticmethod\n        def static_method():\n          return None\n      MyClass().static_method()\n    ')

    def test_default_return_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(x=""):\n          x = list(x)\n      f()\n    ')
        self.assertTypesMatchPytd(ty, 'def f(x=...) -> None: ...')

    def test_lookup(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Cloneable:\n          def __init__(self):\n            pass\n\n          def clone(self):\n            return type(self)()\n      Cloneable().clone()\n    ')
        cls = ty.Lookup('Cloneable')
        method = cls.Lookup('clone')
        self.assertEqual(pytd_utils.Print(method), 'def clone(self: _TCloneable) -> _TCloneable: ...')

    @test_base.skip("pytype thinks 'clone' returns a TypeVar(bound=Cloneable)")
    def test_simple_clone(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Cloneable:\n        def clone(self):\n          return Cloneable()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Cloneable:\n        def clone(self) -> Cloneable: ...\n    ')

    def test_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class MyStaticMethodDecorator:\n        def __init__(self, func):\n          self.__func__ = func\n        def __get__(self, obj, cls):\n          return self.__func__\n\n      class A:\n        @MyStaticMethodDecorator\n        def mystaticmethod(x, y):\n          return x + y\n\n      def f():\n        return A.mystaticmethod(1, 2)\n\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class MyStaticMethodDecorator:\n        __func__: Any\n        def __init__(self, func) -> None: ...\n        def __get__(self, obj, cls) -> Any: ...\n      class A:\n        mystaticmethod: Any\n      def f() -> int: ...\n    ')

    def test_unknown_decorator(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      @__any_object__\n      def f():\n        return 3j\n      f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      f: Any\n    ')

    def test_func_name(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      def f():\n        pass\n      f.func_name = 3.1415\n      def g():\n        return f.func_name\n      g()\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> None: ...\n      def g() -> float: ...\n    ')

    def test_register(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, _) = self.InferWithErrors("\n      class Foo:\n        pass\n      def f():\n        lookup = {}\n        lookup[''] = Foo\n        return lookup.get('')()  # not-callable\n    ")
        self.assertTypesMatchPytd(ty, '\n      class Foo: ...\n      def f() -> Foo: ...\n    ')

    def test_copy_method(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        def mymethod(self, x, y):\n          return 3\n      myfunction = Foo.mymethod\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def mymethod(self, x, y) -> int: ...\n      def myfunction(self: Foo, x, y) -> int: ...\n    ')

    def test_assign_method(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        pass\n      def myfunction(self, x, y):\n        return 3\n      Foo.mymethod = myfunction\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def mymethod(self, x, y) -> int: ...\n      def myfunction(self: Foo, x, y) -> int: ...\n    ')

    def test_function_attr(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import os\n      def f():\n        pass\n      class Foo:\n        def method(self):\n          pass\n      foo = Foo()\n      f.x = 3\n      Foo.method.x = "bar"\n      foo.method.x = 3j  # overwrites previous line\n      os.chmod.x = 3.14\n      a = f.x\n      b = Foo.method.x\n      c = foo.method.x\n      d = os.chmod.x\n    ')
        self.assertTypesMatchPytd(ty, '\n    import os\n    def f() -> NoneType: ...\n    class Foo:\n      def method(self) -> NoneType: ...\n    foo = ...  # type: Foo\n    a = ...  # type: int\n    b = ...  # type: complex\n    c = ...  # type: complex\n    d = ...  # type: float\n    ')

    def test_json(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import json\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n    import json\n    ')

    def test_new(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      x = str.__new__(str)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: str\n    ')

    def test_override_new(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo(str):\n        def __new__(cls, string):\n          return str.__new__(cls, string)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type, TypeVar\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      class Foo(str):\n        def __new__(cls: Type[_TFoo], string) -> _TFoo: ...\n    ')

    def test_inherit_new(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo(str): pass\n      foo = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo(str): ...\n      foo = ...  # type: Foo\n    ')

    def test_attribute_in_new(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __new__(cls, name):\n          self = super(Foo, cls).__new__(cls)\n          self.name = name\n          return self\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Type, TypeVar\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      class Foo:\n        name = ...  # type: Any\n        def __new__(cls: Type[_TFoo], name) -> _TFoo: ...\n    ')

    def test_attributes_in_new_and_init(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __new__(cls):\n          self = super(Foo, cls).__new__(cls)\n          self.name = "Foo"\n          return self\n        def __init__(self):\n          self.nickname = 400\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type, TypeVar\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      class Foo:\n        name = ...  # type: str\n        nickname = ...  # type: int\n        def __new__(cls: Type[_TFoo]) -> _TFoo: ...\n        def __init__(self) -> None : ...\n    ')

    def test_variable_product_complexity_limit(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class A:\n        def __new__(cls, w, x, y, z):\n          pass\n      class B(A):\n        pass\n      class C(A):\n        pass\n      class D(A):\n        pass\n      options = [\n          (1, 2, 3, 4),\n          (5, 6, 7, 8),\n          (9, 10, 11, 12),\n          (13, 14, 15, 16),\n          (17, 18, 19, 20),\n      ]\n      for w, x, y, z in options:\n        A(w, x, y, z)\n        B(w, x, y, z)\n        C(w, x, y, z)\n        D(w, x, y, z)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Tuple\n      class A:\n        def __new__(cls, w, x, y, z) -> None: ...\n      class B(A): ...\n      class C(A): ...\n      class D(A): ...\n      options = ...  # type: List[Tuple[int, int, int, int]]\n      w = ...  # type: int\n      x = ...  # type: int\n      y = ...  # type: int\n      z = ...  # type: int\n    ')

    def test_return_self(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __enter__(self):\n          return self\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      class Foo:\n        def __enter__(self: _TFoo) -> _TFoo: ...\n    ')

    def test_attribute_in_inherited_new(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        def __new__(cls, name):\n          self = super(Foo, cls).__new__(cls)\n          self.name = name\n          return self\n      class Bar(Foo):\n        def __new__(cls):\n          return super(Bar, cls).__new__(cls, "")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Type, TypeVar\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      _TBar = TypeVar("_TBar", bound=Bar)\n      class Foo:\n        name = ...  # type: Any\n        def __new__(cls: Type[_TFoo], name) -> _TFoo: ...\n      class Bar(Foo):\n        name = ...  # type: str\n        def __new__(cls: Type[_TBar]) -> _TBar: ...\n    ')

    def test_pyi_classmethod_and_staticmethod(self):
        if False:
            print('Hello World!')
        with self.DepTree([('t.pyi', '\n      class A:\n        @classmethod\n        def foo(): ...\n        @staticmethod\n        def bar(): ...\n    ')]):
            self.Check('\n        import t\n        a = t.A.foo.__name__\n        b = t.A.bar.__name__\n        assert_type(a, str)\n        assert_type(b, str)\n      ')
if __name__ == '__main__':
    test_base.main()