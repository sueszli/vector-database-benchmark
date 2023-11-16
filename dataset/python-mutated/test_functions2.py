"""Test functions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class PreciseReturnTest(test_base.BaseTest):
    """Tests for --precise-return."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.options.tweak(precise_return=True)

    def test_interpreter_return(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      def f(x: str) -> str:\n        return x\n      x = f(0)  # wrong-arg-types[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(x: str) -> str: ...\n      x: str\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

    def test_interpreter_unknown_return(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      def f(x: str):\n        return x\n      x = f(0)  # wrong-arg-types[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f(x: str) -> str: ...\n      x: Any\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

    def test_interpreter_overload(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      from typing import overload\n      @overload\n      def f(x: str) -> str: ...\n      def f(x):\n        return x\n      x = f(0)  # wrong-arg-types[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import overload\n      @overload\n      def f(x: str) -> str: ...\n      x: str\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

class TestCheckDefaults(test_base.BaseTest):
    """Tests for checking parameter defaults against annotations."""

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      def f(x: int = ''):  # annotation-type-mismatch[e]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_typevar(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors("\n      from typing import TypeVar\n      T = TypeVar('T')\n      def f(x: T = 0, y: T = ''):  # annotation-type-mismatch[e]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_instance_method(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      class Foo:\n        def f(self, x: int = ''):  # annotation-type-mismatch[e]\n          pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_kwonly_arg(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      def f(*, x: int = ''):  # annotation-type-mismatch[e]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_multiple_errors(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      def f(x: int = '', y: str = 0):  # annotation-type-mismatch[e1]  # annotation-type-mismatch[e2]\n        pass\n    ")
        self.assertErrorRegexes(errors, {'e1': 'Annotation: int.*Assignment: str', 'e2': 'Annotation: str.*Assignment: int'})

    def test_ellipsis(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      def f(x: int = ...):  # annotation-type-mismatch\n        return x\n    ')

    def test_overload_ellipsis(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import overload\n\n      @overload\n      def f(x: int = ...): ...\n      @overload\n      def f(x: str = ...): ...\n\n      def f(x):\n        return x\n    ')

class TestFunctions(test_base.BaseTest):
    """Tests for functions."""

    def test_object_to_callable(self):
        if False:
            print('Hello World!')
        self.Check('\n      class MyClass:\n        def method(self):\n          return\n\n      def takes_object(o: object):\n        return\n\n      takes_object(MyClass().method)\n    ')

    def test_function_to_callable(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f():\n        def g1(x: int, y: bool) -> str:\n          return "hello world"\n        def g2() -> int:\n          return 42\n        return g1, g2\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, Tuple\n      def f() -> Tuple[Callable[[int, bool], str], Callable[[], int]]: ...\n    ')

    def test_function_to_callable_return_only(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        def g1(x=None) -> int:\n          return 42\n        def g2(*args) -> str:\n          return "hello world"\n        return g1, g2\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, Tuple\n      def f() -> Tuple[Callable[..., int], Callable[..., str]]: ...\n    ')

    def test_fake_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n\n      class Foo:\n        def __init__(self, x: int):\n          self.y = __any_object__\n\n      foo = Foo("foo")  # pytype: disable=wrong-arg-types\n      foo.y  # if __init__ fails, this line throws an error\n      ')

    def test_argument_name_conflict(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors('\n      from typing import Dict\n      def f(x: Dict[str, int]):\n        x[""] = ""  # container-type-mismatch\n        return x\n      def g(x: Dict[str, int]):\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      def f(x: Dict[str, int]) -> Dict[str, Union[str, int]]: ...\n      def g(x: Dict[str, int]) -> Dict[str, int]: ...\n    ')

    def test_argument_type_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, _) = self.InferWithErrors('\n      from typing import Dict\n      def f(x: Dict[str, int], y: Dict[str, int]):\n        x[""] = ""  # container-type-mismatch\n        return x, y\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Tuple, Union\n      def f(\n        x: Dict[str, int], y: Dict[str, int]\n      ) -> Tuple[Dict[str, Union[str, int]], Dict[str, int]]: ...\n    ')

    def test_typecheck_varargs(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      def f(*args: int) -> int:\n        return args[0]\n      f(*['value'])  # wrong-arg-types\n      f(1, 'hello', 'world')  # wrong-arg-types\n      ")

    def test_typecheck_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      def f(**kwargs: int) -> int:\n        return len(kwargs.values())\n      f(**{'arg': 'value'})  # wrong-arg-types\n      f(arg='value', arg2=3)  # wrong-arg-types\n    ")

    def test_pass_func_to_complex_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Optional\n      def f(x1, x2: Optional[str], x3, x4, x5, x6, x7, x8, x9, xA, xB):\n        pass\n      def g(x2: Optional[str] = None, x3: Optional[str] = None,\n            x4: Optional[str] = None, x5: Optional[str] = None,\n            x6: Optional[str] = None, x7: Optional[str] = None,\n            x8: Optional[str] = None, x9: Optional[str] = None,\n            xA: Optional[str] = None, xB: Optional[str] = None):\n        f(lambda: None, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB)\n    ')

    def test_type_param_args(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Any, Type, TypeVar\n      T = TypeVar('T')\n      def cast(typ: Type[T], val: Any) -> T:\n        return val\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Type, TypeVar\n\n      T = TypeVar('T')\n\n      def cast(typ: Type[T], val) -> T: ...\n    ")

    def test_varargs(self):
        if False:
            print('Hello World!')
        self.Check("\n      def foo(x: str, y: bytes, *z: int):\n        pass\n      foo('abc', b'def', 123)\n      foo('abc', b'def', 123, 456, 789)\n      foo('abc', b'def', *[123, 456, 789])\n      foo('abc', *[b'def', 123, 456, 789])\n      foo(*['abc', b'def', 123, 456, 789])\n      def bar(*y: int):\n        foo('abc', b'def', *y)\n    ")

    def text_varargs_errors(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      def foo(x: str, *y: int):\n        pass\n      foo(*[1, 2, 3])  # wrong-arg-types[e1]\n      def bar(*z: int):\n        foo(*z)  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'str.*int'})

    def test_varargs_in_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: int, *args): ...\n      ')
            self.Check('\n        import foo\n        def g(*args):\n          foo.f(42, *args)\n      ', pythonpath=[d.path])

    def test_varargs_in_pyi_error(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: int, *args): ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        def g(*args):\n          foo.f("", *args)  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'int.*str'})

    def test_function_type(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import types\n      def f(x: types.FunctionType):\n        pass\n      f(lambda: None)\n    ')

    def test_bad_function_match(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      def f():\n        pass\n      def g(x: [][0]):\n        pass\n      g(f)  # wrong-arg-types\n    ')

    def test_noreturn(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Any, Callable, NoReturn\n\n      def f(x: int) -> NoReturn:\n        raise NotImplementedError()\n\n      def g(x: Callable[[int], Any]):\n        pass\n\n      g(f)\n    ')

    def test_starargs_list(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import List\n      def f() -> List[int]:\n        return __any_object__\n      def g(x, y, z):\n        pass\n      def h(x):\n        return g(x, *f())\n    ')

    def test_namedargs_split(self):
        if False:
            print('Hello World!')
        self.Check("\n      def f(x):\n        pass\n      def g(y):\n        pass\n      def h():\n        kws = {}\n        if __random__:\n          kws['x'] = 0\n          f(**kws)\n        else:\n          kws['y'] = 0\n          g(**kws)\n    ")

    def test_namedargs_split_pyi(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      def f(x): ...\n      def g(y): ...\n    ')]):
            self.Check("\n        import foo\n        def h():\n          kws = {}\n          if __random__:\n            kws['x'] = 0\n            foo.f(**kws)\n          else:\n            kws['y'] = 0\n            foo.g(**kws)\n      ")

    def test_filter_none(self):
        if False:
            return 10
        self.Check("\n      import copy\n      from typing import Dict, Optional, Union\n      X = {'a': 1}\n      def f(x: Optional[Dict[str, bytes]] = None):\n        y = x or X\n        z = copy.copy(y)\n        assert_type(z, Union[Dict[str, int], Dict[str, bytes]])\n    ")

class TestFunctionsPython3Feature(test_base.BaseTest):
    """Tests for functions."""

    def test_make_function(self):
        if False:
            for i in range(10):
                print('nop')
        src = '\n      def uses_annotations(x: int) -> int:\n        i, j = 3, 4\n        return i\n\n      def uses_pos_defaults(x, y=1):\n        i, j = 3, 4\n        return __any_object__\n\n      def uses_kw_defaults(x, *myargs, y=1):\n        i, j = 3, 4\n        return __any_object__\n\n      def uses_kwargs(x, **mykwargs):\n        i, j = 3, 4\n        return __any_object__\n    '
        output = '\n      from typing import Any\n      def uses_annotations(x: int) -> int: ...\n      def uses_pos_defaults(x, y=...) -> Any: ...\n      def uses_kw_defaults(x, *myargs, y=...) -> Any: ...\n      def uses_kwargs(x, **mykwargs) -> Any: ...\n    '
        self.assertTypesMatchPytd(self.Infer(src, deep=False), output)
        self.assertTypesMatchPytd(self.Infer(src, deep=True), output)

    def test_make_function2(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(x, *myargs, y):\n        return __any_object__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f(x, *myargs, y) -> Any: ...\n    ')

    def test_make_function3(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(a = 2, *args, b:int = 1, **kwargs):\n        x = 0\n        def g(i:int = 3) -> int:\n          print(x)\n        return g\n\n      y = f(2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n\n      def f(a: int = ..., *args, b: int = ..., **kwargs) -> Callable[Any, int]: ...\n      def y(i: int = ...) -> int: ...\n    ')

    def test_make_function_deep(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(a = 2, *args, b:int = 1, **kwargs):\n        x = 0\n        def g(i:int = 3) -> int:\n          return x + i\n        return g\n\n      y = f(2)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n\n      def f(a = ..., *args, b: int = ..., **kwargs) -> Callable[Any, int]: ...\n      def y(i: int = ...) -> int: ...\n    ')

    def test_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def foo(a, b, c, d=0, e=0, f=0, g=0, *myargs,\n              u, v, x, y=0, z=0, **mykwargs):\n        return 3\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(a, b, c, d=..., e=..., f=..., g=..., *myargs,\n              u, v, x, y=..., z=..., **mykwargs): ...\n    ')

    def test_defaults_and_annotations(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo(a, b, c:int, d=0, e=0, f=0, g=0, *myargs,\n              u:str, v, x:float=0, y=0, z=0, **mykwargs):\n        return 3\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def foo(a, b, c:int, d=..., e=..., f=..., g=..., *myargs,\n              u:str, v, x:float=..., y=..., z=..., **mykwargs): ...\n    ')

    def test_namedtuple_defaults(self):
        if False:
            return 10
        self.Check('\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        field: int\n      Foo.__new__.__defaults__ = ((),) * len(Foo._fields)\n   ')

    def test_matching_functions(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        return 3\n\n      class Foo:\n        def match_method(self):\n          return map(self.method, [])\n        def match_function(self):\n          return map(f, [])\n        def match_pytd_function(self):\n          return map(map, [])\n        def match_bound_pytd_function(self):\n          return map({}.keys, [])\n        def method(self):\n          pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator\n      def f() -> int: ...\n      class Foo:\n        def match_method(self) -> Iterator[nothing]: ...\n        def match_function(self) -> Iterator[nothing]: ...\n        def match_pytd_function(self) -> Iterator[nothing]: ...\n        def match_bound_pytd_function(self) -> Iterator[nothing]: ...\n        def method(self) -> NoneType: ...\n    ')

    def test_build_with_unpack(self):
        if False:
            return 10
        ty = self.Infer("\n      a = [1,2,3,4]\n      b = [1,2,3,4]\n      c = {'1':2, '3':4}\n      d = {'5':6, '7':8}\n      e = {'9':10, 'B':12}\n      # Test merging two dicts into an args dict for k\n      x = {'a': 1, 'c': 2}\n      y = {'b': 1, 'd': 2}\n\n      def f(**kwargs):\n        print(kwargs)\n\n      def g(*args):\n        print(args)\n\n      def h(*args, **kwargs):\n        print(args, kwargs)\n\n      def j(a=1, b=2, c=3):\n        print(a, b,c)\n\n      def k(a, b, c, d, **kwargs):\n        print(a, b, c, d, kwargs)\n\n      p = [*a, *b]  # BUILD_LIST_UNPACK\n      q = {*a, *b}  # BUILD_SET_UNPACK\n      r = (*a, *b)  # BUILD_TUPLE_UNPACK\n      s = {**c, **d}  # BUILD_MAP_UNPACK\n      f(**c, **d, **e)  # BUILD_MAP_UNPACK_WITH_CALL\n      g(*a, *b)  # BUILD_TUPLE_UNPACK_WITH_CALL\n      h(*a, *b, **c, **d)\n      j(**{'a': 1, 'b': 2})  # BUILD_CONST_KEY_MAP\n      k(**x, **y, **e)  # BUILD_MAP_UNPACK_WITH_CALL\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Set, Tuple\n\n      a = ...  # type: List[int]\n      b = ...  # type: List[int]\n      c = ...  # type: Dict[str, int]\n      d = ...  # type: Dict[str, int]\n      e = ...  # type: Dict[str, int]\n      p = ...  # type: List[int]\n      q = ...  # type: Set[int]\n      r = ...  # type: Tuple[int, int, int, int, int, int, int, int]\n      s = ...  # type: Dict[str, int]\n      x = ...  # type: Dict[str, int]\n      y = ...  # type: Dict[str, int]\n\n      def f(**kwargs) -> None: ...\n      def g(*args) -> None: ...\n      def h(*args, **kwargs) -> None: ...\n      def j(a = ..., b = ..., c = ...) -> None: ...\n      def k(a, b, c, d, **kwargs) -> None: ...\n    ')

    def test_unpack_str(self):
        if False:
            return 10
        ty = self.Infer('\n      s1 = "abc"\n      s2 = "def"\n      tup = (*s1, *s2)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      s1 = ...  # type: str\n      s2 = ...  # type: str\n      tup = ...  # type: Tuple[str, str, str, str, str, str]\n    ')

    def test_unpack_nonliteral(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      def f(x, **kwargs):\n        return kwargs['y']\n      def g(**kwargs):\n        return f(x=10, **kwargs)\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n\n      def f(x, **kwargs) -> Any: ...\n      def g(**kwargs) -> Any: ...\n    ')

    def test_unpack_multiple_bindings(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      if __random__:\n        x = {'a': 1, 'c': 2}\n      else:\n        x = {'a': '1', 'c': '2'}\n      if __random__:\n        y = {'b': 1, 'd': 2}\n      else:\n        y = {'b': b'1', 'd': b'2'}\n      z = {**x, **y}\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, TypeVar, Union\n\n      x = ...  # type: Dict[str, Union[str, int]]\n      y = ...  # type: Dict[str, Union[bytes, int]]\n      z = ...  # type: Dict[str, Union[bytes, int, str]]\n    ')

    def test_kwonly(self):
        if False:
            return 10
        self.Check('\n      from typing import Optional\n      def foo(x: int, *, z: Optional[int] = None) -> None:\n        pass\n\n      foo(1, z=5)\n    ')

    def test_varargs_with_kwonly(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def foo(x: int, *args: int, z: int) -> None:\n        pass\n\n      foo(1, 2, z=5)\n    ')

    def test_varargs_with_missing_kwonly(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def foo(x: int, *args: int, z: int) -> None:\n        pass\n\n      foo(1, 2, 5)  # missing-parameter[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\bz\\b'})

    def test_multiple_varargs_packs(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Tuple\n      def foo1(*x: int):\n        pass\n      def foo2(x: str, y: bytes, *z: int):\n        pass\n      foo1(*[1, 2, 3], *[4, 5, 6])\n      foo2('abc', b'def', *[1, 2, 3], *[4, 5, 6])\n      def bar(y: Tuple[int], *z: int):\n        foo1(*y, *z)\n        foo2('abc', b'def', *y, *z)\n    ")

    def text_multiple_varargs_packs_errors(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def foo(x: str, *y: int):\n        pass\n      foo(*[1, 2, 3], *[4, 5, 6])  # wrong-arg-types[e1]\n      def bar(*z: int):\n        foo(*z, *z)  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'str.*int'})

    def test_kwonly_to_callable(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, *, y):\n        pass\n      class Foo:\n        def __init__(self):\n          self.f = f\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable\n      def f(x, *, y) -> None: ...\n      class Foo:\n        f: Callable\n        def __init__(self) -> None: ...\n    ')

    def test_positional_only_parameter(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors('\n      def f(x, /, y):\n        pass\n      f(0, 1)  # ok\n      f(0, y=1)  # ok\n      f(x=0, y=1)  # wrong-keyword-args[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(x, /, y) -> None: ...\n    ')
        self.assertErrorSequences(errors, {'e': ['Invalid keyword argument x', 'Expected: (x, /, y)']})

    def test_positional_only_parameter_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x, /, y) -> None: ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.f(0, 1)  # ok\n        foo.f(0, y=1)  # ok\n        foo.f(x=0, y=1)  # wrong-keyword-args[e]\n      ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['Invalid keyword argument x', 'Expected: (x, /, y)']})

    def test_positional_and_keyword_arguments(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x, /, **kwargs) -> None: ...\n      ')
            self.Check('\n        import foo\n        def f(x, /, **kwargs):\n          pass\n        foo.f(1, x=1)\n        f(1, x=1)\n      ', pythonpath=[d.path])

    def test_posonly_starstararg_clash(self):
        if False:
            print('Hello World!')
        self.Check("\n      def f(arg: int, /, **kwargs: str):\n        pass\n      f(1, arg='text')\n    ")

    def test_pyi_posonly_starstararg_clash(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(arg: int, /, **kwargs: str) -> None: ...\n      ')
            self.Check("\n        import foo\n        foo.f(1, arg='text')\n      ", pythonpath=[d.path])

class DisableTest(test_base.BaseTest):
    """Tests for error disabling."""

    def test_invalid_parameter_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f(\n        x: 0 = 0\n      ):  # pytype: disable=invalid-annotation\n        pass\n    ')

    def test_invalid_return_annotation(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f() -> (\n        list[\n            3.14]):  # pytype: disable=invalid-annotation\n        return []\n      def g(\n      ) -> list[\n          3.14\n      ]:  # pytype: disable=invalid-annotation\n        return []\n    ')

    def test_invalid_subscripted_parameter_annotation(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(\n        x: list[3.14]  # pytype: disable=invalid-annotation\n      ):\n        pass\n    ')

    def test_bad_yield_annotation(self):
        if False:
            return 10
        self.Check('\n      def f(\n          x: int) -> int:  # pytype: disable=bad-yield-annotation\n        yield x\n    ')
if __name__ == '__main__':
    test_base.main()