"""Tests for typing.py."""
from pytype.pytd import pep484
from pytype.tests import test_base
from pytype.tests import test_utils

class TypingTest(test_base.BaseTest):
    """Tests for typing.py."""
    _TEMPLATE = '\n    import collections\n    import typing\n    def f(s: %(annotation)s):%(disables)s\n      return s\n    f(%(arg)s)\n  '

    def _test_match(self, arg, annotation, disables=''):
        if False:
            i = 10
            return i + 15
        self.Check(self._TEMPLATE % locals())

    def _test_no_match(self, arg, annotation, disables=''):
        if False:
            return 10
        code = (self._TEMPLATE % locals()).rstrip() + '  # wrong-arg-types'
        self.InferWithErrors(code)

    def test_list_match(self):
        if False:
            i = 10
            return i + 15
        self._test_match('[1, 2, 3]', 'typing.List')
        self._test_match('[1, 2, 3]', 'typing.List[int]')
        self._test_match('[1, 2, 3.1]', 'typing.List[typing.Union[int, float]]')
        self._test_no_match('[1.1, 2.1, 3.1]', 'typing.List[int]')

    def test_sequence_match(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_match('[1, 2, 3]', 'typing.Sequence')
        self._test_match('[1, 2, 3]', 'typing.Sequence[int]')
        self._test_match('(1, 2, 3.1)', 'typing.Sequence[typing.Union[int, float]]')
        self._test_no_match('[1.1, 2.1, 3.1]', 'typing.Sequence[int]')

    def test_generator(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Generator\n      def f() -> Generator[int, None, None]:\n        for i in range(3):\n          yield i\n    ')

    def test_type(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      from typing import Type\n      class Foo:\n        x = 1\n      def f1(foo: Type[Foo]):\n        return foo.x\n      def f2(foo: Type[Foo]):\n        return foo.y  # attribute-error[e]\n      def f3(foo: Type[Foo]):\n        return foo.mro()\n      def f4(foo: Type[Foo]):\n        return foo()\n      v1 = f1(Foo)\n      v2 = f2(Foo)\n      v3 = f3(Foo)\n      v4 = f4(Foo)\n    ')
        self.assertErrorRegexes(errors, {'e': 'y.*Foo'})
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Type\n      class Foo:\n        x = ...  # type: int\n      def f1(foo: Type[Foo]) -> int: ...\n      def f2(foo: Type[Foo]) -> Any: ...\n      def f3(foo: Type[Foo]) -> list: ...\n      def f4(foo: Type[Foo]) -> Foo: ...\n      v1 = ...  # type: int\n      v2 = ...  # type: Any\n      v3 = ...  # type: list\n      v4 = ...  # type: Foo\n    ')

    def test_type_union(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      from typing import Type, Union\n      class Foo:\n        bar = ...  # type: int\n      def f1(x: Type[Union[int, Foo]]):\n        # Currently not an error, since attributes on Unions are retrieved\n        # differently.  See get_attribute() in attribute.py.\n        x.bar\n      def f2(x: Union[Type[int], Type[Foo]]):\n        x.bar  # attribute-error[e]\n        f1(x)\n      def f3(x: Type[Union[int, Foo]]):\n        f1(x)\n        f2(x)\n    ')
        self.assertErrorRegexes(errors, {'e': 'bar.*int'})

    def test_use_type_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import List\n        MyType = List[str]\n      ')
            self.Check('\n        import foo\n        def f(x: foo.MyType):\n          pass\n        f([""])\n      ', pythonpath=[d.path])

    def test_callable(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Callable\n        def f() -> Callable: ...\n      ')
            self.Check('\n        from typing import Callable\n        import foo\n        def f() -> Callable:\n          return foo.f()\n        def g() -> Callable:\n          return int\n      ', pythonpath=[d.path])

    def test_callable_parameters(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import Any, Callable\n\n      # The below are all valid.\n      def f1(x: Callable[[int, str], bool]): ...\n      def f2(x: Callable[..., bool]): ...\n      def f3(x: Callable[[], bool]): ...\n\n      def g1(x: Callable[int, bool]): ...  # _ARGS not a list  # invalid-annotation[e1]\n      lst = [int] if __random__ else [str]\n      def g2(x: Callable[lst, bool]): ...  # _ARGS ambiguous  # invalid-annotation[e2]  # invalid-annotation[e3]\n      # bad: _RET ambiguous\n      def g3(x: Callable[[], bool if __random__ else str]): ...  # invalid-annotation[e4]\n      # bad: _ARGS[0] ambiguous\n      def g4(x: Callable[[int if __random__ else str], bool]): ...  # invalid-annotation[e5]\n      lst = None  # type: list[int]\n      def g5(x: Callable[lst, bool]): ...  # _ARGS not a constant  # invalid-annotation[e6]\n      def g6(x: Callable[[42], bool]): ...  # _ARGS[0] not a type  # invalid-annotation[e7]\n      def g7(x: Callable[[], bool, int]): ...  # Too many params  # invalid-annotation[e8]\n      def g8(x: Callable[Any, bool]): ...  # Any is not allowed  # invalid-annotation[e9]\n      def g9(x: Callable[[]]) -> None: ...  # invalid-annotation[e10]\n    ')
        self.assertTypesMatchPytd(ty, '\n       from typing import Any, Callable, List, Type\n\n       lst = ...  # type: List[int]\n\n       def f1(x: Callable[[int, str], bool]) -> None: ...\n       def f2(x: Callable[Any, bool]) -> None: ...\n       def f3(x: Callable[[], bool]) -> None: ...\n       def g1(x: Callable[Any, bool]) -> None: ...\n       def g2(x: Callable[Any, bool]) -> None: ...\n       def g3(x: Callable[[], Any]) -> None: ...\n       def g4(x: Callable[[Any], bool]) -> None: ...\n       def g5(x: Callable[Any, bool]) -> None: ...\n       def g6(x: Callable[[Any], bool]) -> None: ...\n       def g7(x: Callable[[], bool]) -> None: ...\n       def g8(x: Callable[Any, bool]) -> None: ...\n       def g9(x: Callable[[], Any]) -> None: ...\n    ')
        self.assertErrorRegexes(errors, {'e1': "'int'.*must be a list of argument types or ellipsis", 'e2': '\\[int\\] or \\[str\\].*Must be constant', 'e3': "'Any'.*must be a list of argument types or ellipsis", 'e4': 'bool or str.*Must be constant', 'e5': 'int or str.*Must be constant', 'e6': 'instance of List\\[int\\].*Must be constant', 'e7': 'instance of int', 'e8': 'Callable.*expected 2.*got 3', 'e9': "'Any'.*must be a list of argument types or ellipsis", 'e10': 'Callable\\[_ARGS, _RET].*2.*1'})

    def test_callable_bad_args(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import Callable\n      lst1 = [str]\n      lst1[0] = int\n      def g1(x: Callable[lst1, bool]): ...  # invalid-annotation[e1]\n      lst2 = [str]\n      while __random__:\n        lst2.append(int)\n      def g2(x: Callable[lst2, bool]): ...  # invalid-annotation[e2]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, List, Type, Union\n      lst1 = ...  # type: List[Type[Union[int, str]]]\n      lst2 = ...  # type: List[Type[Union[int, str]]]\n      def g1(x: Callable[..., bool]) -> None: ...\n      def g2(x: Callable[..., bool]) -> None: ...\n    ')
        self.assertErrorRegexes(errors, {'e1': 'instance of List\\[Type\\[Union\\[int, str\\]\\]\\].*Must be constant', 'e2': 'instance of List\\[Type\\[Union\\[int, str\\]\\]\\].*Must be constant'})

    def test_generics(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Dict\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class CustomDict(Dict[K, V]): ...\n      ')
            self.Check('\n        import typing\n        import foo\n        def f(x: typing.Callable[..., int]): pass\n        def f(x: typing.Iterator[int]): pass\n        def f(x: typing.Iterable[int]): pass\n        def f(x: typing.Container[int]): pass\n        def f(x: typing.Sequence[int]): pass\n        def f(x: typing.Tuple[int, str]): pass\n        def f(x: typing.MutableSequence[int]): pass\n        def f(x: typing.List[int]): pass\n        def f(x: typing.Deque[int]): pass\n        def f(x: typing.IO[str]): pass\n        def f(x: typing.Collection[str]): pass\n        def f(x: typing.Mapping[int, str]): pass\n        def f(x: typing.MutableMapping[int, str]): pass\n        def f(x: typing.Dict[int, str]): pass\n        def f(x: typing.AbstractSet[int]): pass\n        def f(x: typing.FrozenSet[int]): pass\n        def f(x: typing.MutableSet[int]): pass\n        def f(x: typing.Set[int]): pass\n        def f(x: typing.Reversible[int]): pass\n        def f(x: typing.SupportsAbs[int]): pass\n        def f(x: typing.Optional[int]): pass\n        def f(x: typing.Generator[int, None, None]): pass\n        def f(x: typing.Type[int]): pass\n        def f(x: typing.Pattern[str]): pass\n        def f(x: typing.Match[str]): pass\n        def f(x: foo.CustomDict[int, str]): pass\n      ', pythonpath=[d.path])

    def test_generator_iterator_match(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Iterator\n      def f(x: Iterator[int]):\n        pass\n      f(x for x in [42])\n    ')

    def test_name_conflict(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import typing\n      def f() -> typing.Any:\n        pass\n      class Any:\n        pass\n      def g() -> Any:\n        pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import typing\n      def f() -> typing.Any: ...\n      def g() -> Any: ...\n      class Any:\n          pass\n    ')

    def test_callable_call(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      from typing import Callable\n      f = ...  # type: Callable[[int], str]\n      v1 = f()  # wrong-arg-count[e1]\n      v2 = f(True)  # ok\n      v3 = f(42.0)  # wrong-arg-types[e2]\n      v4 = f(1, 2)  # wrong-arg-count[e3]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      f = ...  # type: Callable[[int], str]\n      v1 = ...  # type: Any\n      v2 = ...  # type: str\n      v3 = ...  # type: Any\n      v4 = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e1': '1.*0', 'e2': 'int.*float', 'e3': '1.*2'})

    def test_callable_call_with_type_parameters(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      from typing import Callable, TypeVar\n      T = TypeVar("T")\n      def f(g: Callable[[T, T], T], y, z):\n        return g(y, z)  # wrong-arg-types[e]\n      v1 = f(__any_object__, 42, 3.14)  # ok\n      v2 = f(__any_object__, 42, "hello world")\n    ', deep=True)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable, TypeVar, Union\n      T = TypeVar("T")\n      def f(g: Callable[[T, T], T], y, z): ...\n      v1 = ...  # type: Union[int, float]\n      v2 = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*str'})

    def test_callable_call_with_return_only(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Callable\n      f = ...  # type: Callable[..., int]\n      v = f()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable\n      f = ...  # type: Callable[..., int]\n      v = ...  # type: int\n    ')

    def test_callable_call_with_varargs_and_kwargs(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import Callable\n      f = ...  # type: Callable[[], int]\n      f(x=3)  # wrong-keyword-args[e1]\n      f(*(42,))  # wrong-arg-count[e2]\n      f(**{"x": "hello", "y": "world"})  # wrong-keyword-args[e3]\n      f(*(42,), **{"hello": "world"})  # wrong-keyword-args[e4]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'x', 'e2': '0.*1', 'e3': 'x, y', 'e4': 'hello'})

    def test_callable_attribute(self):
        if False:
            return 10
        self.Check('\n      from typing import Any, Callable\n      def foo(fn: Callable[[Any], Any]):\n        fn.foo # pytype: disable=attribute-error\n    ')

    def test_items_view(self):
        if False:
            return 10
        self.Check('\n      from typing import ItemsView\n      def f(x: ItemsView[str, int]): ...\n    ')

    def test_new_type(self):
        if False:
            return 10
        ty = self.Infer("\n      from typing import NewType\n      MyInt = NewType('MyInt', int)\n      class A:\n        pass\n      MyA = NewType('MyA', A)\n      MySpecialA = NewType('MySpecialA', MyA)\n      MyStr1 = NewType(*('MyStr1', str))\n      MyStr2 = NewType(**{'tp':str, 'name':'MyStr2'})\n      MyAnyType = NewType('MyAnyType', tp=str if __random__ else int)\n      MyFunnyNameType = NewType('Foo' if __random__ else 'Bar', tp=str)\n      def func1(i: MyInt) -> MyInt:\n        return i\n      def func2(i: MyInt) -> int:\n        return i\n      def func3(a: MyA) -> MyA:\n        return a\n      def func4(a: MyA) -> A:\n        return a\n      def func5(a: MySpecialA) -> MySpecialA:\n        return a\n      def func6(a: MySpecialA) -> MyA:\n        return a\n      def func7(a: MySpecialA) -> A:\n        return a\n      v = 123\n      func1(MyInt(v))\n      func2(MyInt(v))\n      my_a = MyA(A())\n      func3(my_a)\n      func4(my_a)\n      my_special_a = MySpecialA(my_a)\n      func5(my_special_a)\n      func6(my_special_a)\n      func7(my_special_a)\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        pass\n      class MyInt(int):\n        def __init__(self, val: int): ...\n      class MyA(A):\n        def __init__(self, val: A): ...\n      class MySpecialA(MyA):\n        def __init__(self, val: MyA): ...\n      class MyStr1(str):\n        def __init__(self, val: str): ...\n      class MyStr2(str):\n        def __init__(self, val: str): ...\n      MyAnyType = ... # type: Any\n      class MyFunnyNameType(str):\n        def __init__(self, val:str): ...\n      def func1(i: MyInt) -> MyInt: ...\n      def func2(i: MyInt) -> int: ...\n      def func3(a: MyA) -> MyA: ...\n      def func4(a: MyA) -> A: ...\n      def func5(a: MySpecialA) -> MySpecialA: ...\n      def func6(a: MySpecialA) -> MyA: ...\n      def func7(a: MySpecialA) -> A: ...\n      v = ...  # type: int\n      my_a = ...  # type: MyA\n      my_special_a = ...  # type: MySpecialA\n    ')

    def test_new_type_error(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors("\n      from typing import NewType\n      MyInt = NewType('MyInt', int)\n      MyStr = NewType('MyStr', str)\n      def func1(i: MyInt) -> MyInt:\n        return i\n      def func2(i: int) -> MyInt:\n        return i  # bad-return-type[e1]\n      def func3(s: MyStr) -> MyStr:\n        return s\n      func1(123)  # wrong-arg-types[e2]\n      func3(MyStr(123))  # wrong-arg-types[e3]\n    ")
        self.assertErrorRegexes(errors, {'e1': 'Expected: MyInt\\nActually returned: int', 'e2': '.*Expected: \\(i: MyInt\\)\\nActually passed: \\(i: int\\)', 'e3': '.*Expected:.*val: str\\)\\nActually passed:.*val: int\\)'})

    def test_new_type_not_abstract(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Mapping, NewType\n      X = NewType('X', Mapping)\n      def f() -> X:\n        return X({})\n    ")

    def test_maybe_return(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def f() -> int:\n        if __random__:\n          return 42\n        else:\n          raise ValueError()\n    ')

    def test_no_return_against_str(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f() -> str:\n        raise ValueError()\n      def g():\n        return f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> str: ...\n      def g() -> str: ...\n    ')

    def test_called_no_return_against_str(self):
        if False:
            return 10
        self.Check('\n      def f():\n        raise ValueError()\n      def g() -> str:\n        return f()\n    ')

    def test_union_ellipsis(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Union\n      MyUnion = Union[int, ...]  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Ellipsis.*index 1.*Union'})

    def test_list_ellipsis(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import List\n      MyList = List[int, ...]  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Ellipsis.*index 1.*List'})

    def test_multiple_ellipses(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import Union\n      MyUnion = Union[..., int, ..., str, ...]  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Ellipsis.*indices 0, 2, 4.*Union'})

    def test_bad_tuple_ellipsis(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import Tuple\n      MyTuple1 = Tuple[..., ...]  # invalid-annotation[e1]\n      MyTuple2 = Tuple[...]  # invalid-annotation[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Ellipsis.*index 0.*Tuple', 'e2': 'Ellipsis.*index 0.*Tuple'})

    def test_bad_callable_ellipsis(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Callable\n      MyCallable1 = Callable[..., ...]  # invalid-annotation[e1]\n      MyCallable2 = Callable[[int], ...]  # invalid-annotation[e2]\n      MyCallable3 = Callable[[...], int]  # invalid-annotation[e3]\n      MyCallable4 = Callable[[int], int, int]  # invalid-annotation[e4]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Ellipsis.*index 1.*Callable', 'e2': 'Ellipsis.*index 1.*Callable', 'e3': 'Ellipsis.*index 0.*list', 'e4': 'Callable\\[_ARGS, _RET].*2.*3'})

    def test_optional_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Optional\n\n      def func1(x: Optional[int]):\n        pass\n\n      def func2(x: Optional):  # invalid-annotation[e1]\n        pass\n\n      def func3(x: Optional[int, float, str]):  # invalid-annotation[e2]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Not a type', 'e2': 'typing\\.Optional can only contain one type parameter'})

    def test_noreturn_possible_return(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import NoReturn\n      def func(x) -> NoReturn:\n        if x > 1:\n          raise ValueError()  # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Expected: Never', 'Actually returned: None']})

    def test_noreturn(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import Any, List, NoReturn\n\n      def func0() -> NoReturn:\n        raise ValueError()\n\n      def func1() -> List[NoReturn]:\n        return [None]  # bad-return-type[e1]\n\n      def func2(x: NoReturn):\n        pass\n      func2(0)  # wrong-arg-types[e2]\n\n      def func3(x: List[NoReturn]):\n        pass\n      func3([0])  # wrong-arg-types[e3]\n\n      def func4():\n        x: List[NoReturn] = []\n        x.append(0)  # container-type-mismatch[e4]\n    ')
        self.assertErrorSequences(errors, {'e1': ['Expected: List[nothing]', 'Actually returned: List[None]'], 'e2': ['Expected: (x: Never)', 'Actually passed: (x: int)'], 'e3': ['Expected: (x: List[nothing])', 'Actually passed: (x: List[int])'], 'e4': ['Allowed', '_T: Never', 'New', '_T: int']})

    def test_noreturn_pyi(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n      from typing import NoReturn\n      def f(x: NoReturn): ...\n    ')]):
            errors = self.CheckWithErrors('\n        import foo\n        foo.f(0)  # wrong-arg-types[e]\n      ')
            self.assertErrorSequences(errors, {'e': ['Expected: (x: empty)', 'Actually passed: (x: int)']})

    def test_noreturn_in_tuple(self):
        if False:
            return 10
        self.Check('\n      from typing import NoReturn\n      def _returns(annotations) -> bool:\n        return annotations["return"] not in (None, NoReturn)\n    ')

    def test_SupportsComplex(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import SupportsComplex\n      def foo(x: SupportsComplex):\n        pass\n      foo(1j)\n    ')

    def test_mutable_set_sub(self):
        if False:
            return 10
        self.Check('\n      from typing import MutableSet\n      def f(x: MutableSet) -> MutableSet:\n        return x - {0}\n    ')

    def test_union_of_classes(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Type, Union\n\n      class Foo:\n        def __getitem__(self, x) -> int:\n          return 0\n      class Bar:\n        def __getitem__(self, x) -> str:\n          return ''\n\n      def f(x: Union[Type[Foo], Type[Bar]]):\n        return x.__getitem__\n      def g(x: Type[Union[Foo, Bar]]):\n        return x.__getitem__\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable, Type, Union\n\n      class Foo:\n        def __getitem__(self, x) -> int: ...\n      class Bar:\n        def __getitem__(self, x) -> str: ...\n\n      def f(x: Type[Union[Foo, Bar]]) -> Callable[[Any, Any], Union[int, str]]: ...\n      def g(x: Type[Union[Foo, Bar]]) -> Callable: ...\n    ')

    def test_bytestring(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import ByteString, Union\n      def f(x: Union[bytes, bytearray, memoryview]):\n        pass\n      x = None  # type: ByteString\n      f(x)\n    ')

    def test_forwardref(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing import ForwardRef\n      X = ForwardRef("Y")  # not-callable\n    ')

class CounterTest(test_base.BaseTest):
    """Tests for typing.Counter."""

    def test_counter_generic(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors('\n      import collections\n      import typing\n      def freqs(s: str) -> typing.Counter[str]:\n        return collections.Counter(s)\n      x = freqs("")\n      y = freqs("")\n      z = collections.Counter()  # type: typing.Counter[int]\n      x - y\n      x + y\n      x | y\n      x & y\n      x - z  # unsupported-operands\n      x.most_common(1, 2, 3)  # wrong-arg-count\n      a = x.most_common()\n      b = x.most_common(1)\n      c = x.elements()\n      d = z.elements()\n      e = x.copy()\n      f = x | z\n    ')
        self.assertTypesMatchPytd(ty, '\n      import collections\n      import typing\n      from typing import Counter, Iterable, List, Tuple, Union\n\n      a: List[Tuple[str, int]]\n      b: List[Tuple[str, int]]\n      c: Iterable[str]\n      d: Iterable[int]\n      e: Counter[str]\n      f: Counter[Union[int, str]]\n\n      x: Counter[str]\n      y: Counter[str]\n      z: Counter[int]\n\n      def freqs(s: str) -> Counter[str]: ...\n    ')

class TypingTestPython3Feature(test_base.BaseTest):
    """Typing tests (Python 3)."""

    def test_namedtuple_item(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        class Ret(NamedTuple):\n          x: int\n          y: str\n        def f() -> Ret: ...\n      ')
            ty = self.Infer('\n        import foo\n        w = foo.f()[-1]\n        x = foo.f()[0]\n        y = foo.f()[1]\n        z = foo.f()[2]  # out of bounds, fall back to the combined element type\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        w: str\n        x: int\n        y: str\n        z: Union[int, str]\n      ')

    def test_import_all(self):
        if False:
            return 10
        python = ['from typing import *  # pytype: disable=not-supported-yet'] + pep484.ALL_TYPING_NAMES
        ty = self.Infer('\n'.join(python), deep=False)
        self.assertTypesMatchPytd(ty, '')

    def test_callable_func_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Any, Callable\n      def foo(fn: Callable[[Any], Any]) -> str:\n        return fn.__qualname__\n    ')

    def test_classvar(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import ClassVar\n      class A:\n        x: ClassVar[int] = 5\n      print(A.x + 3)  # make sure using a ClassVar[int] as an int works\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import ClassVar\n      class A:\n        x: ClassVar[int]\n    ')

    def test_uninitialized_classvar(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import ClassVar\n      class A:\n        x: ClassVar[int]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import ClassVar\n      class A:\n        x: ClassVar[int]\n    ')

    def test_pyi_classvar_of_union(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import ClassVar, Optional\n        class Foo:\n          x: ClassVar[Optional[str]]\n      ')
            self.Check('\n        import foo\n        from typing import Optional\n        def f(x: Optional[str]):\n          pass\n        f(foo.Foo.x)\n      ', pythonpath=[d.path])

    def test_ordered_dict(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n      from typing import OrderedDict\n      def f(x: OrderedDict[str, int]): ...\n      f(collections.OrderedDict(a=0))\n      def g(x: collections.OrderedDict[str, int]): ...\n      g(OrderedDict(a=0))\n    ')

    def test_instantiate_ordered_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import OrderedDict\n      OrderedDict()\n    ')

    def test_typed_dict(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing_extensions import TypedDict\n        X = TypedDict('X', {'a': int})\n      ")
            self.CheckWithErrors("\n        import foo\n        from typing import Dict\n\n        def f1(x: Dict[str, int]):\n          pass\n        def f2(x: Dict[int, str]):\n          pass\n        def f3(x: foo.X):\n          pass\n\n        x = None  # type: foo.X\n\n        f1(x)  # okay\n        f2(x)  # wrong-arg-types\n        f3({'a': 0})  # okay\n        f3({0: 'a'})  # wrong-arg-types\n      ", pythonpath=[d.path])

class LiteralTest(test_base.BaseTest):
    """Tests for typing.Literal in source code."""

    def test_basic(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import Literal\n      x1: Literal["hello"]\n      x2: Literal[b"hello"]\n      x3: Literal[u"hello"]\n      x4: Literal[0]\n      x5: Literal[True]\n      x6: Literal[None]\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Literal\n      x1: Literal['hello']\n      x2: Literal[b'hello']\n      x3: Literal['hello']\n      x4: Literal[0]\n      x5: Literal[True]\n      x6: None\n    ")

    def test_basic_enum(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import enum\n      from typing_extensions import Literal\n      class Color(enum.Enum):\n        RED = "RED"\n      x: Literal[Color.RED]\n    ')
        self.assertTypesMatchPytd(ty, '\n      import enum\n      from typing import Literal\n      x: Literal[Color.RED]\n      class Color(enum.Enum):\n        RED: str\n    ')

    @test_base.skip('Pytype loads N.A and treats it as a literal.')
    def test_not_an_enum(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing_extensions import Literal\n      class N:\n        A = 1\n      x: Literal[N.A]  # bad-annotation\n    ')

    def test_missing_enum_member(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      import enum\n      from typing_extensions import Literal\n      class M(enum.Enum):\n        A = 1\n      x: Literal[M.B]  # attribute-error\n    ')

    def test_union(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing_extensions import Literal\n      def f(x: Literal["x", "y"]):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Literal, Union\n      def f(x: Literal['x', 'y']) -> None: ...\n    ")

    def test_unnest(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing_extensions import Literal\n      X = Literal["X"]\n      def f(x: Literal[X, Literal[None], Literal[Literal["Y"]]]):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Literal, Optional, Union\n      X = Literal['X']\n      def f(x: Optional[Literal['X', 'Y']]) -> None: ...\n    ")

    def test_invalid(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import Literal\n      x1: Literal[0, ...]  # invalid-annotation[e1]\n      x2: Literal[str, 4.2]  # invalid-annotation[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': "Bad parameter '...' at index 1", 'e2': "Bad parameter 'str' at index 0\\n\\s*Bad parameter 'float' at index 1"})

    def test_variable(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing_extensions import Literal\n      x: Literal[0] = 0\n      y: Literal[0] = 1  # annotation-type-mismatch[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: Literal\\[0\\].*Assignment: Literal\\[1\\]'})

    def test_parameter(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing_extensions import Literal\n      def f(x: Literal[True]):\n        pass\n      f(True)\n      f(False)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*Literal\\[True\\].*Actual.*Literal\\[False\\]'})

    def test_union_parameter(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing_extensions import Literal\n      def f(x: Literal["x", "z"]):\n        pass\n      f("x")\n      f("y")  # wrong-arg-types[e]\n      f("z")\n    ')
        self.assertErrorRegexes(errors, {'e': "Expected.*Literal\\['x', 'z'\\].*Actual.*Literal\\['y'\\]"})

    def test_mixed_union(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import enum\n      from typing_extensions import Literal\n\n      class M(enum.Enum):\n        A = 1\n\n      def use(x: Literal["hello", M.A]) -> None: ...\n\n      use(None)  # wrong-arg-types\n  ')

    def test_return(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing_extensions import Literal\n      def f() -> Literal["hello"]:\n        if __random__:\n          return "hello"\n        else:\n          return "goodbye"  # bad-return-type[e]\n    ')
        self.assertErrorRegexes(errors, {'e': "Expected.*Literal\\['hello'\\].*Actual.*Literal\\['goodbye'\\]"})

    def test_match_non_literal(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing_extensions import Literal\n      x: Literal["x"]\n      def f(x: str):\n        pass\n      def g(x: int):\n        pass\n      f(x)\n      g(x)  # wrong-arg-types\n    ')

    def test_match_enum(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n    from typing_extensions import Literal\n    import enum\n\n    class M(enum.Enum):\n      A = 1\n      B = 2\n\n    x: Literal[M.A]\n\n    def f(x: Literal[M.A]) -> None:\n      pass\n\n    f(M.A)\n    f(x)\n    f(M.B)  # wrong-arg-types\n    ')

    def test_iterate(self):
        if False:
            print('Hello World!')
        self.options.tweak(strict_parameter_checks=False)
        self.Check('\n      from typing_extensions import Literal\n      def f(x: Literal["x", "y"]):\n        pass\n      for x in ["x", "y"]:\n        f(x)\n    ')

    def test_overloads(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Optional, overload\n      from typing_extensions import Literal\n\n      @overload\n      def f(x: Literal[False]) -> str: ...\n\n      @overload\n      def f(x: Literal[True]) -> Optional[str]: ...\n\n      def f(x) -> Optional[str]:\n        if x:\n          return None\n        return ""\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Literal, Optional, overload\n      @overload\n      def f(x: Literal[False]) -> str: ...\n      @overload\n      def f(x: Literal[True]) -> Optional[str]: ...\n    ')

    def test_list_of_literals(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      import dataclasses\n      from typing import List\n      from typing_extensions import Literal\n\n      Strings = Literal['hello', 'world']\n\n      @dataclasses.dataclass\n      class A:\n        x: List[Strings]\n\n      A(x=['hello', 'world'])\n      A(x=['oops'])  # wrong-arg-types\n    ")

    def test_list_of_list_of_literals(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      import dataclasses\n      from typing import List\n      from typing_extensions import Literal\n\n      Strings = Literal['hello', 'world']\n\n      @dataclasses.dataclass\n      class A:\n        x: List[List[Strings]]\n\n      A(x=[['hello', 'world']])\n      A(x=[['oops']])  # wrong-arg-types\n    ")

    def test_lots_of_literals(self):
        if False:
            return 10
        ty = self.Infer("\n      from typing import Literal\n      X: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 'A']\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Literal\n      X: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 'A']\n    ")

class TypeAliasTest(test_base.BaseTest):
    """Tests for typing.TypeAlias."""

    def test_basic(self):
        if False:
            return 10
        for suffix in ('', '_extensions'):
            typing_module = f'typing{suffix}'
            with self.subTest(typing_module=typing_module):
                ty = self.Infer(f'\n          from {typing_module} import TypeAlias\n          X: TypeAlias = int\n        ')
                self.assertTypesMatchPytd(ty, '\n          from typing import Type\n          X: Type[int]\n        ')

    def test_bad_alias(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import TypeAlias\n      X: TypeAlias = 0  # invalid-annotation\n    ')

    def test_pyi(self):
        if False:
            while True:
                i = 10
        for suffix in ('', '_extensions'):
            typing_module = f'typing{suffix}'
            with self.subTest(typing_module=typing_module):
                with test_utils.Tempdir() as d:
                    d.create_file('foo.pyi', f'\n            from {typing_module} import TypeAlias\n            X: TypeAlias = int\n          ')
                    self.Check('\n            import foo\n            assert_type(foo.X, "Type[int]")\n          ', pythonpath=[d.path])

    def test_forward_ref(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import TypeAlias\n      X: TypeAlias = "int"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      X: Type[int]\n    ')
if __name__ == '__main__':
    test_base.main()