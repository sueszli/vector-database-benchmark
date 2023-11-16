"""Test functions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestGenerators(test_base.BaseTest):
    """Tests for generators."""

    def test_first(self):
        if False:
            return 10
        self.Check('\n      def two():\n        yield 1\n        yield 2\n      for i in two():\n        print(i)\n      ')

    def test_partial_generator(self):
        if False:
            print('Hello World!')
        self.Check('\n      from functools import partial\n\n      def f(a,b):\n        num = a+b\n        while num:\n          yield num\n          num -= 1\n\n      f2 = partial(f, 2)\n      three = f2(1)\n      assert list(three) == [3,2,1]\n      ')

    def test_unsolvable(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      assert list(three) == [3,2,1]\n      ')

    def test_yield_multiple_values(self):
        if False:
            while True:
                i = 10
        self.assertNoCrash(self.Check, '\n      def triples():\n        yield 1, 2, 3\n        yield 4, 5, 6\n\n      for a, b, c in triples():\n        print(a, b, c)\n      ')

    def test_generator_reuse(self):
        if False:
            return 10
        self.Check('\n      g = (x*x for x in range(5))\n      print(list(g))\n      print(list(g))\n      ')

    def test_generator_from_generator2(self):
        if False:
            return 10
        self.Check('\n      g = (x*x for x in range(3))\n      print(list(g))\n\n      g = (x*x for x in range(5))\n      g = (y+1 for y in g)\n      print(list(g))\n      ')

    def test_generator_from_generator(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      class Thing:\n        RESOURCES = (\'abc\', \'def\')\n        def get_abc(self):\n          return "ABC"\n        def get_def(self):\n          return "DEF"\n        def resource_info(self):\n          for name in self.RESOURCES:\n            get_name = \'get_\' + name\n            yield name, getattr(self, get_name)\n\n        def boom(self):\n          #d = list((name, get()) for name, get in self.resource_info())\n          d = [(name, get()) for name, get in self.resource_info()]\n          return d\n\n      print(Thing().boom())\n      ')

class PreciseReturnTest(test_base.BaseTest):
    """Tests for --precise-return."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.options.tweak(precise_return=True)

    def test_pytd_return(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors("\n      x = 'hello'.startswith(0)  # wrong-arg-types[e]\n    ")
        self.assertTypesMatchPytd(ty, 'x: bool')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

    def test_param_return(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        def f(x: T) -> T: ...\n      ')
            (ty, _) = self.InferWithErrors('\n        import foo\n        x = foo.f()  # missing-parameter\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        x: Any\n      ')

    def test_binop(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors("x = 'oops' + 0  # unsupported-operands")
        self.assertTypesMatchPytd(ty, 'x: str')

    def test_inplace_op(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      x = []\n      x += 0  # unsupported-operands\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      x: List[nothing]\n    ')

class TestFunctions(test_base.BaseTest):
    """Tests for functions."""

    def test_functions(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def fn(a, b=17, c="Hello", d=[]):\n        d.append(99)\n        print(a, b, c, d)\n      fn(1)\n      fn(2, 3)\n      fn(3, c="Bye")\n      fn(4, d=["What?"])\n      fn(5, "b", "c")\n      ')

    def test_function_locals(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def f():\n        x = "Spite"\n        print(x)\n      def g():\n        x = "Malice"\n        print(x)\n      x = "Humility"\n      f()\n      print(x)\n      g()\n      print(x)\n      ')

    def test_recursion(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def fact(n):\n        if n <= 1:\n          return 1\n        else:\n          return n * fact(n-1)\n      f6 = fact(6)\n      print(f6)\n      assert f6 == 720\n      ')

    def test_calling_functions_with_args_kwargs(self):
        if False:
            print('Hello World!')
        self.Check('\n      def fn(a, b=17, c="Hello", d=[]):\n        d.append(99)\n        print(a, b, c, d)\n      fn(6, *[77, 88])\n      fn(**{\'c\': 23, \'a\': 7})\n      fn(6, *[77], **{\'c\': 23, \'d\': [123]})\n      ')

    def test_calling_functions_with_generator_args(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        def next(self):\n          raise StopIteration()\n        def __iter__(self):\n          return A()\n      def f(*args):\n        pass\n      f(*A())\n    ')

    def test_defining_functions_with_args_kwargs(self):
        if False:
            print('Hello World!')
        self.Check('\n      def fn(*args):\n        print("args is %r" % (args,))\n      fn(1, 2)\n      ')
        self.Check('\n      def fn(**kwargs):\n        print("kwargs is %r" % (kwargs,))\n      fn(red=True, blue=False)\n      ')
        self.Check('\n      def fn(*args, **kwargs):\n        print("args is %r" % (args,))\n        print("kwargs is %r" % (kwargs,))\n      fn(1, 2, red=True, blue=False)\n      ')
        self.Check('\n      def fn(x, y, *args, **kwargs):\n        print("x is %r, y is %r" % (x, y))\n        print("args is %r" % (args,))\n        print("kwargs is %r" % (kwargs,))\n      fn(\'a\', \'b\', 1, 2, red=True, blue=False)\n      ')

    def test_defining_functions_with_empty_args_kwargs(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def fn(*args):\n        print("args is %r" % (args,))\n      fn()\n      ')
        self.Check('\n      def fn(**kwargs):\n        print("kwargs is %r" % (kwargs,))\n      fn()\n      ')
        self.Check('\n      def fn(*args, **kwargs):\n        print("args is %r, kwargs is %r" % (args, kwargs))\n      fn()\n      ')

    def test_partial(self):
        if False:
            return 10
        self.Check('\n      from functools import partial\n\n      def f(a,b):\n        return a-b\n\n      f7 = partial(f, 7)\n      four = f7(3)\n      assert four == 4\n      ')

    def test_partial_with_kwargs(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from functools import partial\n\n      def f(a,b,c=0,d=0):\n        return (a,b,c,d)\n\n      f7 = partial(f, b=7, c=1)\n      them = f7(10)\n      assert them == (10,7,1,0)\n      ')

    def test_wraps(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('myfunctools.pyi', '\n        from typing import Any, Callable, Sequence\n        from typing import Any\n        _AnyCallable = Callable[..., Any]\n        def wraps(wrapped: _AnyCallable, assigned: Sequence[str] = ..., updated: Sequence[str] = ...) -> Callable[[_AnyCallable], _AnyCallable]: ...\n      ')
            self.Check("\n        from myfunctools import wraps\n        def my_decorator(f):\n          dec = wraps(f)\n          def wrapper(*args, **kwds):\n            print('Calling decorated function')\n            return f(*args, **kwds)\n          wrapper = dec(wrapper)\n          return wrapper\n\n        @my_decorator\n        def example():\n          '''Docstring'''\n          return 17\n\n        assert example() == 17\n        ", pythonpath=[d.path])

    def test_pass_through_args(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(a, b):\n        return a * b\n      def g(*args, **kwargs):\n        return f(*args, **kwargs)\n      g(1, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(a: int, b: int) -> int: ...\n      def g(*args, **kwargs) -> int: ...\n    ')

    def test_pass_through_kwargs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f(a, b):\n        return a * b\n      def g(*args, **kwargs):\n        return f(*args, **kwargs)\n      g(a=1, b=2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def f(a: int, b: int) -> int: ...\n      def g(*args, **kwargs) -> int: ...\n    ')

    def test_pass_through_named_args_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      def f(a: int, b: str):\n        pass\n      def g(*args, **kwargs):\n        return f(*args, a='a', **kwargs)  # wrong-arg-types\n    ")

    def test_pass_through_partial_named_args_and_kwargs(self):
        if False:
            print('Hello World!')
        self.Check("\n      class Foo:\n        def __init__(self, name, labels):\n          pass\n\n      def g(name, bar, **kwargs):\n        Foo(name=name, **kwargs)\n\n      def f(name, x, **args):\n        g(name=name, bar=x, **args)\n\n      f('a', 10, labels=None)\n    ")

    def test_list_comprehension(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(elements):\n        return "%s" % ",".join(t for t in elements)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(elements) -> str: ...\n    ')

    def test_named_arg_unsolvable_max_depth(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        return max(foo=repr(__any_object__))  # wrong-keyword-args[e]\n    ', deep=True, maximum_depth=1)
        self.assertErrorRegexes(errors, {'e': 'foo.*max'})

    def test_multiple_signatures_with_type_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: int) -> List[T]: ...\n        def f(x: List[T], y: str) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x, y):\n          return foo.f(x, y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x, y) -> list: ...\n      ')

    def test_multiple_signatures_with_multiple_type_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, Tuple, TypeVar\n        T = TypeVar("T")\n        def f(arg1: int) -> List[T]: ...\n        def f(arg2: str) -> Tuple[T, T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x) -> Any: ...\n      ')

    def test_unknown_single_signature(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: int) -> List[T]: ...\n        def f(x: List[T], y: str) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(y):\n          return foo.f("", y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import List\n        def f(y) -> List[str]: ...\n    ')

    def test_unknown_with_solved_type_parameter(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T, y: T) -> List[T]: ...\n        def f(x: List[T], y: T) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x, "")\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Union\n        def f(x) -> list: ...\n      ')

    def test_unknown_with_extra_information(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List, TypeVar\n        T = TypeVar("T")\n        def f(x: T) -> List[T]: ...\n        def f(x: List[T]) -> List[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)[0].isnumeric()\n        def g(x):\n          return foo.f(x) + [""]\n        def h(x):\n          ret = foo.f(x)\n          x + ""\n          return ret\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, List, MutableSequence\n        def f(x) -> Any: ...\n        def g(x) -> list: ...\n        def h(x) -> list: ...\n      ')

    def test_type_parameter_in_return(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class MyPattern(Generic[T]):\n          def match(self, string: T) -> MyMatch[T]: ...\n        class MyMatch(Generic[T]):\n          pass\n        def compile() -> MyPattern[T]: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.compile().match("")\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: foo.MyMatch[str]\n      ')

    def test_multiple_signatures(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: str) -> float: ...\n        def f(x: int, y: bool) -> int: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f(0, True)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: int\n      ')

    def test_multiple_signatures_with_unknown(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(arg1: str) -> float: ...\n        def f(arg2: int) -> bool: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x) -> Any: ...\n      ')

    def test_multiple_signatures_with_optional_arg(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: str) -> int: ...\n        def f(*args) -> float: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x) -> Any: ...\n      ')

    def test_multiple_signatures_with_kwarg(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(*, y: int) -> bool: ...\n        def f(y: str) -> float: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(y=x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        def f(x) -> Any: ...\n      ')

    def test_isinstance(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(isinstance=isinstance):\n        pass\n      def g():\n        f()\n      def h():\n        return isinstance\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable, Tuple, Union\n      def f(isinstance = ...) -> None: ...\n      def g() -> None: ...\n      def h() -> Callable[[Any, Union[Tuple[Union[Tuple[type, ...], type], ...], type]], bool]: ...\n    ')

    def test_wrong_keyword(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        pass\n      f("", y=42)  # wrong-keyword-args[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'y'})

    def test_staticmethod_class(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      v1, = (object.__new__,)\n      v2 = type(object.__new__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, Type\n      v1 = ...  # type: Callable\n      v2 = ...  # type: Type[Callable]\n    ')

    def test_function_class(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f() -> None: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f(): pass\n        v1 = (foo.f,)\n        v2 = type(foo.f)\n        w1 = (f,)\n        w2 = type(f)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Callable, Tuple\n        def f() -> None: ...\n        v1 = ...  # type: Tuple[Callable[[], None]]\n        v2 = Callable\n        w1 = ...  # type: Tuple[Callable[[], Any]]\n        w2 = Callable\n      ')

    def test_type_parameter_visibility(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple, TypeVar, Union\n        T = TypeVar("T")\n        def f(x: T) -> Tuple[Union[T, str], int]: ...\n      ')
            ty = self.Infer('\n        import foo\n        v1, v2 = foo.f(42j)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        v1 = ...  # type: Union[str, complex]\n        v2 = ...  # type: int\n      ')

    def test_pytd_function_in_class(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def bar(): ...\n      ')
            self.Check('\n        import foo\n        class A:\n          bar = foo.bar\n          def f(self):\n           self.bar()\n      ', pythonpath=[d.path])

    def test_interpreter_function_in_class(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      class A:\n        bar = lambda x: x\n        def f(self):\n          self.bar(42)  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '1.*2'})

    def test_lambda(self):
        if False:
            return 10
        self.CheckWithErrors('\n      def f():\n        a = lambda: 1 + ""  # unsupported-operands\n    ')

    def test_nested_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f(c):\n        return lambda c: f(c)\n    ')

    def test_nested_lambda2(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(d):\n        return lambda c: f(c)\n    ')

    def test_nested_lambda3(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(t):\n        lambda u=[t,1]: f(u)\n      ')

    def test_set_defaults(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      X = collections.namedtuple("X", "a b c d")\n      X.__new__.__defaults__ = (3, 4)\n      a = X(1, 2)\n      b = X(1, 2, 3)\n      c = X(1, 2, 3, 4)\n      ')

    def test_set_defaults_non_new(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def b(x: int, y: int, z: int): ...\n        ')
            ty = self.Infer("\n        import a\n        a.b.__defaults__ = ('3',)\n        a.b(1, 2)\n        c = a.b\n        ", deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        def c(x: int, y: int, z: int = ...): ...\n        ')

    def test_bad_defaults(self):
        if False:
            return 10
        self.InferWithErrors('\n      import collections\n      X = collections.namedtuple("X", "a b c")\n      X.__new__.__defaults__ = (1)  # bad-function-defaults\n      ')

    def test_multiple_valid_defaults(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections\n      X = collections.namedtuple("X", "a b c")\n      X.__new__.__defaults__ = (1,) if __random__ else (1,2)\n      X(0)  # should not cause an error\n      ')

    def test_set_defaults_to_expression(self):
        if False:
            return 10
        self.Check('\n      import collections\n      X = collections.namedtuple("X", "a b c")\n      X.__new__.__defaults__ = (None,) * len(X._fields)\n      ')

    def test_set_defaults_non_tuple_instance(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      import collections\n      X = collections.namedtuple("X", "a b c")\n      X.__new__.__defaults__ = (lambda x: x)(0)  # bad-function-defaults\n    ')

    def test_set_builtin_defaults(self):
        if False:
            return 10
        self.assertNoCrash(self.Check, '\n      import os\n      os.chdir.__defaults__ = ("/",)\n      os.chdir()\n      ')

    def test_interpreter_function_defaults(self):
        if False:
            return 10
        self.Check('\n      def test(a, b, c = 4):\n        return a + b + c\n      x = test(1, 2)\n      test.__defaults__ = (3, 4)\n      y = test(1, 2)\n      y = test(1)\n      test.__defaults__ = (2, 3, 4)\n      z = test()\n      z = test(1)\n      z = test(1, 2)\n      z = test(1, 2, 3)\n      ')
        self.InferWithErrors('\n      def test(a, b, c):\n        return a + b + c\n      x = test(1, 2)  # missing-parameter\n      test.__defaults__ = (3,)\n      x = test(1, 2)\n      x = test(1)  # missing-parameter\n      ')

    def test_interpreter_function_defaults_on_class(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      class Foo:\n        def __init__(self, a, b, c):\n          self.a = a\n          self.b = b\n          self.c = c\n      a = Foo()  # missing-parameter\n      Foo.__init__.__defaults__ = (1, 2)\n      b = Foo(0)\n      c = Foo()  # missing-parameter\n      ')

    def test_split_on_kwargs(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def make_foo(**kwargs):\n        varargs = kwargs.pop("varargs", None)\n        if kwargs:\n          raise TypeError()\n        return varargs\n      Foo = make_foo(varargs=True)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Optional\n      def make_foo(**kwargs) -> Any: ...\n      Foo = ...  # type: bool\n    ')

    def test_pyi_starargs(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: str, *args) -> None: ...\n      ')
            self.CheckWithErrors('\n        import foo\n        foo.f(True, False)  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_starargs_matching_pyi_posargs(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: int, y: int, z: int) -> None: ...\n      ')
            self.CheckWithErrors('\n        import foo\n        def g(x, *args):\n          foo.f(x, *args)\n          foo.f(x, 1, *args)\n          foo.f(x, 1)  # missing-parameter\n      ', pythonpath=[d.path])

    def test_starargs_forwarding(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: int) -> None: ...\n      ')
            self.Check('\n        import foo\n        def f(x, y, *args):\n          for i in args:\n            foo.f(i)\n        def g(*args):\n          f(1, 2, *args)\n      ', pythonpath=[d.path])

    def test_infer_bound_pytd_func(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import struct\n      if __random__:\n        int2byte = struct.Struct(">B").pack\n      else:\n        int2byte = chr\n    ')
        self.assertTypesMatchPytd(ty, '\n      import struct\n      from typing import overload\n      @overload\n      def int2byte(*v) -> bytes: ...\n      @overload\n      def int2byte(i: int) -> str: ...\n    ')

    def test_preserve_return_union(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Union\n        def f(x: int) -> Union[int, str]: ...\n        def f(x: float) -> Union[int, str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.f(__any_object__)\n      ', pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import Union\n      v = ...  # type: Union[int, str]\n    ')

    def test_call_with_varargs_and_kwargs(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def foo(an_arg):\n        pass\n      def bar(an_arg, *args, **kwargs):\n        foo(an_arg, *args, **kwargs)\n    ')

    def test_functools_partial(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import functools\n      def f(a, b):\n        pass\n      partial_f = functools.partial(f, 0)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import functools\n      def f(a, b) -> None: ...\n      partial_f: functools.partial\n    ')

    def test_functools_partial_kw(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import functools\n      def f(a, b=None):\n        pass\n      partial_f = functools.partial(f, 0)\n      partial_f(0)\n    ')

    def test_functools_partial_class(self):
        if False:
            return 10
        self.Check('\n      import functools\n      class X:\n        def __init__(self, a, b):\n          pass\n      PartialX = functools.partial(X, 0)\n      PartialX(0)\n    ')

    def test_functools_partial_class_kw(self):
        if False:
            return 10
        self.Check('\n      import functools\n      class X:\n        def __init__(self, a, b=None):\n          pass\n      PartialX = functools.partial(X, 0)\n      PartialX(0)\n    ')

    def test_functools_partial_bad_call(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      import functools\n      functools.partial()  # missing-parameter\n      functools.partial(42)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Callable.*int'})

    def test_bad_comprehensions(self):
        if False:
            return 10
        self.CheckWithErrors('\n      [name_error1 for x in ()]  # name-error\n      {name_error2 for x in ()}  # name-error\n      (name_error3 for x in ())  # name-error\n      lambda x: name_error4  # name-error\n    ')

    def test_new_function(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import types\n      def new_function(code, globals):\n        return types.FunctionType(code, globals)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import types\n      from typing import Callable\n      def new_function(code, globals) -> Callable: ...\n    ')

    def test_function_globals(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        def g():\n          pass\n        return g.__globals__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      def f() -> Dict[str, Any]: ...\n    ')

    def test_hashable(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Hashable\n      def f(x):\n        # type: (Hashable) -> None\n        pass\n      def g():\n        pass\n      f(g)\n    ')
if __name__ == '__main__':
    test_base.main()