"""Tests for the analysis phase matcher (match_var_against_type)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class MatchTest(test_base.BaseTest):
    """Tests for matching types."""

    def test_no_argument_pytd_function_against_callable(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def bar() -> bool: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        from typing import Callable\n        import foo\n\n        def f(x: Callable[[], int]): ...\n        def g(x: Callable[[], str]): ...\n\n        f(foo.bar)  # ok\n        g(foo.bar)  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '\\(x: Callable\\[\\[\\], str\\]\\).*\\(x: Callable\\[\\[\\], bool\\]\\)'})

    def test_pytd_function_against_callable_with_type_parameters(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f1(x: int) -> int: ...\n        def f2(x: int) -> bool: ...\n        def f3(x: int) -> str: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        from typing import Callable, TypeVar\n        import foo\n\n        T_plain = TypeVar("T_plain")\n        T_constrained = TypeVar("T_constrained", int, bool)\n        def f1(x: Callable[[T_plain], T_plain]): ...\n        def f2(x: Callable[[T_constrained], T_constrained]): ...\n\n        f1(foo.f1)  # ok\n        f1(foo.f2)  # ok\n        f1(foo.f3)  # wrong-arg-types[e1]\n        f2(foo.f1)  # ok\n        f2(foo.f2)  # wrong-arg-types[e2]\n        f2(foo.f3)  # wrong-arg-types[e3]\n      ', pythonpath=[d.path])
            expected = 'Callable\\[\\[Union\\[bool, int\\]\\], Union\\[bool, int\\]\\]'
            self.assertErrorRegexes(errors, {'e1': 'Expected.*Callable\\[\\[str\\], str\\].*Actual.*Callable\\[\\[int\\], str\\]', 'e2': 'Expected.*Callable\\[\\[bool\\], bool\\].*Actual.*Callable\\[\\[int\\], bool\\]', 'e3': 'Expected.*' + expected + '.*Actual.*Callable\\[\\[int\\], str\\]'})

    def test_interpreter_function_against_callable(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import Callable\n      def f(x: Callable[[bool], int]): ...\n      def g1(x: int) -> bool:\n        return __any_object__\n      def g2(x: str) -> int:\n        return __any_object__\n      f(g1)  # ok\n      f(g2)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*Callable\\[\\[bool\\], int\\].*Actual.*Callable\\[\\[str\\], int\\]'})

    def test_bound_interpreter_function_against_callable(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      from typing import Callable\n\n      class A:\n        def f(self, x: int) -> bool:\n          return __any_object__\n      unbound = A.f\n      bound = A().f\n\n      def f1(x: Callable[[bool], int]): ...\n      def f2(x: Callable[[A, bool], int]): ...\n      def f3(x: Callable[[bool], str]): ...\n\n      f1(bound)  # ok\n      f2(bound)  # wrong-arg-types[e1]\n      f3(bound)  # wrong-arg-types[e2]\n      f1(unbound)  # wrong-arg-types[e3]\n      f2(unbound)  # ok\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Expected.*Callable\\[\\[A, bool\\], int\\].*Actual.*Callable\\[\\[int\\], bool\\]', 'e2': 'Expected.*Callable\\[\\[bool\\], str\\].*Actual.*Callable\\[\\[int\\], bool\\]', 'e3': 'Expected.*Callable\\[\\[bool\\], int\\].*Actual.*Callable\\[\\[Any, int\\], bool\\]'})

    def test_callable_parameters(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any, Callable, List, TypeVar\n        T = TypeVar("T")\n        def f1(x: Callable[..., T]) -> List[T]: ...\n        def f2(x: Callable[[T], Any]) -> List[T]: ...\n      ')
            ty = self.Infer('\n        from typing import Any, Callable\n        import foo\n\n        def g1(): pass\n        def g2() -> int: pass\n        v1 = foo.f1(g1)\n        v2 = foo.f1(g2)\n\n        def g3(x): pass\n        def g4(x: int): pass\n        w1 = foo.f2(g3)\n        w2 = foo.f2(g4)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any, List\n        import foo\n        def g1() -> Any: ...\n        def g2() -> int: ...\n        def g3(x) -> Any: ...\n        def g4(x: int) -> Any: ...\n\n        v1 = ...  # type: list\n        v2 = ...  # type: List[int]\n        w1 = ...  # type: list\n        w2 = ...  # type: List[int]\n      ')

    def test_variable_length_function_against_callable(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import Any, Callable\n      def f(x: Callable[[int], Any]): pass\n      def g1(x: int=0): pass\n      def g2(x: str=""): pass\n      f(g1)  # ok\n      f(g2)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*Callable\\[\\[int\\], Any\\].*Actual.*Callable\\[\\[str\\], Any\\]'})

    def test_callable_instance_against_callable_with_type_parameters(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import Callable, TypeVar\n      T = TypeVar("T")\n      def f(x: Callable[[T], T]): ...\n      def g() -> Callable[[int], str]: return __any_object__\n      f(g())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*Callable\\[\\[str\\], str\\].*Actual.*Callable\\[\\[int\\], str\\]'})

    def test_function_with_type_parameter_return_against_callable(self):
        if False:
            print('Hello World!')
        self.InferWithErrors('\n      from typing import Callable, AnyStr, TypeVar\n      T = TypeVar("T")\n      def f(x: Callable[..., AnyStr]): ...\n      def g1(x: AnyStr) -> AnyStr: return x\n      def g2(x: T) -> T: return x\n\n      f(g1)  # ok\n      f(g2)  # wrong-arg-types\n    ')

    def test_union_in_type_parameter(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Callable, Iterator, List, TypeVar\n        T = TypeVar("T")\n        def decorate(func: Callable[..., Iterator[T]]) -> List[T]: ...\n      ')
            ty = self.Infer('\n        from typing import Generator, Optional\n        import foo\n        @foo.decorate\n        def f() -> Generator[Optional[str], None, None]:\n          yield "hello world"\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import List, Optional\n        import foo\n        f = ...  # type: List[Optional[str]]\n      ')

    def test_anystr(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import AnyStr, Dict, Tuple\n      class Foo:\n        def bar(self, x: Dict[Tuple[AnyStr], AnyStr]): ...\n    ')

    def test_formal_type(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      from typing import AnyStr, List, NamedTuple\n      def f(x: str):\n        pass\n      f(AnyStr)  # wrong-arg-types\n      def g(x: List[str]):\n        pass\n      g([AnyStr])  # wrong-arg-types\n      H = NamedTuple("H", [(\'a\', AnyStr)])  # invalid-annotation\n    ')

    def test_typevar_with_bound(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import Callable, TypeVar\n      T1 = TypeVar("T1", bound=int)\n      T2 = TypeVar("T2")\n      def f(x: T1) -> T1:\n        return __any_object__\n      def g(x: Callable[[T2], T2]) -> None:\n        pass\n      g(f)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected.*T2.*Actual.*T1'})

    def test_callable_base_class(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any, Callable, Union, Type\n        def f() -> Union[Callable[[], Any], Type[Exception]]: ...\n        def g() -> Union[Type[Exception], Callable[[], Any]]: ...\n      ')
            self.Check('\n        from typing import Union\n        import foo\n        class Foo(foo.f()):\n          pass\n        class Bar(foo.g()):\n          pass\n        def f(x: Foo, y: Bar) -> Union[Bar, Foo]:\n          return x or y\n        f(Foo(), Bar())\n      ', pythonpath=[d.path])

    def test_anystr_against_callable(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Any, AnyStr, Callable, TypeVar\n      T = TypeVar('T')\n      def f(x: AnyStr) -> AnyStr:\n        return x\n      def g(f: Callable[[T], Any], x: T):\n        pass\n      g(f, 'hello')\n    ")

    def test_anystr_against_bounded_callable(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Any, AnyStr, Callable, TypeVar\n      IntVar = TypeVar('IntVar', bound=int)\n      def f(x: AnyStr) -> AnyStr:\n        return x\n      def g(f: Callable[[IntVar], Any], x: IntVar):\n        pass\n      g(f, 0)  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Callable\\[\\[IntVar\\], Any\\].*Callable\\[\\[AnyStr\\], AnyStr\\]'})

    def test_anystr_against_multiple_param_callable(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      from typing import Any, AnyStr, Callable, TypeVar\n      T = TypeVar('T')\n      def f(x: AnyStr) -> AnyStr:\n        return x\n      def g(f: Callable[[T], T]):\n        pass\n      g(f)  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Callable\\[\\[T\\], T\\].*Callable\\[\\[AnyStr\\], AnyStr\\]'})

    @test_utils.skipOnWin32('Fails on windows for unknown reasons')
    def test_filter_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import collections\n      import six\n      from typing import Dict\n\n      def f() -> Dict[str, bytes]:\n        d = collections.defaultdict(list)\n        for _ in range(10):\n          subdict = {}  # type: Dict[str, str]\n          k = subdict.get('k')\n          if not k:\n            continue\n          d[k].append(b'')\n        return {k: b', '.join(v) for k, v in six.iteritems(d)}\n    ")

    def test_cast_away_optional(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import Optional, TypeVar\n      T = TypeVar('T')\n      def f(x: Optional[T]) -> T:\n        assert x is not None\n        return x\n      def g(x: Optional[str]):\n        return f(x)\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Optional, TypeVar\n      T = TypeVar('T')\n      def f(x: Optional[T]) -> T: ...\n      def g(x: Optional[str]) -> str: ...\n    ")

    def test_mapping_attributes(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Mapping\n\n      def f(x: Mapping): ...\n\n      class BadMap:\n        def __getitem__(self, *a) -> str:\n         return \'\'\n        def __iter__(self): ...\n        def __len__(self) -> int:\n          return 1\n\n      f(BadMap())  # wrong-arg-types\n\n      class GoodMap:\n        def keys(self): ...\n        def values(self): ...\n        def items(self): ...\n        def __getitem__(self, *a) -> str:\n         return \'\'\n        def get(self, k): ...\n        def __iter__(self): ...\n        def __contains__(self): ...\n        def __eq__(self, other) -> bool:\n          return True\n        def __ne__(self, other): ...\n        def __len__(self) -> int:\n          return 1\n\n      f(GoodMap())\n\n      class GoodMapChild(GoodMap):\n        pass\n\n      f(GoodMapChild())\n\n      f(range(10))  # wrong-arg-types\n      f("abc")  # wrong-arg-types\n    ')

    def test_inherited_mapping_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Mapping\n      class GoodMap(Mapping):\n        def __getitem__(self, *a) -> str:\n         return ''\n        def __iter__(self): ...\n        def __len__(self) -> int:\n          return 1\n      def f(x: Mapping): ...\n      f(GoodMap())\n    ")

    def test_mutablemapping_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Mapping\n      from collections import MutableMapping\n      def f(x: Mapping): ...\n      f(MutableMapping())\n    ')

    def test_typevar_and_list_of_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import List, TypeVar, Union\n      T = TypeVar('T')\n      KeepFilter = Union[T, List[T]]\n      def filter_values(values: List[T], to_keep: KeepFilter[T]) -> List[T]:\n        return [v for v in values if _should_keep(v, to_keep)]\n      def _should_keep(value: T, to_keep: KeepFilter[T]) -> bool:\n        if isinstance(to_keep, list):\n          return value in to_keep\n        else:\n          return value == to_keep\n      filter_values([.1, .2], .2)\n    ")

    def test_typevar_union(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import TypeVar, Union\n      T1 = TypeVar('T1')\n      T2 = TypeVar('T2')\n      def f(x: T1, y: Union[T1, T2]) -> T2:\n        return __any_object__\n      assert_type(f(0, None), None)\n    ")

    def test_append_tuple(self):
        if False:
            return 10
        self.Check("\n      from typing import List, Tuple\n      x: List[Tuple[str, int]]\n      x.append(('', 0))\n    ")

class MatchTestPy3(test_base.BaseTest):
    """Tests for matching types."""

    def test_callable(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import tokenize\n      def f():\n        pass\n      x = tokenize.generate_tokens(f)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Generator\n      import tokenize\n      def f() -> NoneType: ...\n      x = ...  # type: Generator[tokenize.TokenInfo, None, None]\n    ')

    def test_callable_against_generic(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import TypeVar, Callable, Generic, Iterable, Iterator\n        A = TypeVar("A")\n        N = TypeVar("N")\n        class Foo(Generic[A]):\n          def __init__(self, c: Callable[[], N]):\n            self = Foo[N]\n        x = ...  # type: Iterator[int]\n      ')
            self.Check('\n        import foo\n        foo.Foo(foo.x.__next__)\n      ', pythonpath=[d.path])

    def test_empty(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      a = []\n      b = ["%d" % i for i in a]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      a = ...  # type: List[nothing]\n      b = ...  # type: List[str]\n    ')

    def test_bound_against_callable(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import io\n      import tokenize\n      x = tokenize.generate_tokens(io.StringIO("").readline)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Generator\n      import io\n      import tokenize\n      x = ...  # type: Generator[tokenize.TokenInfo, None, None]\n    ')

class NonIterableStringsTest(test_base.BaseTest):
    """Tests for non-iterable string behavior."""

    def test_add_string(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = []\n      a += list("foo")\n      a += "bar"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      a = ...  # type: List[str]\n    ')

    def test_str_against_plain_iterable(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Iterable\n      def f (itr: Iterable):\n        return\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_str_against_iterable(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing import Iterable\n      def f(x: Iterable[str]):\n        return x\n      f("abcdef")  # wrong-arg-types\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_str_against_plain_sequence(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Sequence\n      def f (itr: Sequence):\n        return\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n    ')

    def test_str_against_sequence(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Sequence\n      def f(x: Sequence[str]):\n        return x\n      f("abcdef")  # wrong-arg-types\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_intended_iterable_str_against_sequence(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union, Sequence\n      def f(x: Union[str, Sequence[str]]):\n        return x\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_intended_iterable_str_against_iterable(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Union, Iterable\n      def f(x: Union[str, Iterable[str]]):\n        return x\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_str_against_union_sequence_str(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union, Sequence\n      def f(x: Union[Sequence[str], str]):\n        return x\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_str_against_union_iterable_str(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Union, Iterable\n      def f(x: Union[Iterable[str], str]):\n        return x\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n      f(("abc", "def", "ghi"))\n    ')

    def test_optional_str_against_iterable(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Iterable, Optional\n      def foo(x: Iterable[str]): ...\n\n      def bar(s: str):\n        foo(s)  # wrong-arg-types\n\n      def baz(os: Optional[str]):\n        foo(os)  # wrong-arg-types\n    ')

    def test_optional_str_against_plain_iterable(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import Iterable, Optional\n      def foo(x: Iterable): ...\n\n      def bar(s: str):\n        foo(s)\n\n      def baz(os: Optional[str]):\n        foo(os)  # TODO(b/63407497): should be wrong-arg-types\n    ')

    def test_str_against_plain_collection(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Collection\n      def f(itr: Collection):\n        return\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n    ')

    def test_str_against_plain_container(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Container\n      def f(itr: Container):\n        return\n      f("abcdef")\n      f(["abc", "def", "ghi"])\n    ')

    def test_str_against_plain_mapping(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Mapping\n      def f(itr: Mapping):\n        return\n      f("abcdef")  # wrong-arg-types\n    ')

    def test_str_against_collection(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing import Collection\n      def f(x: Collection[str]):\n        return\n      f("abcdef")  # wrong-arg-types\n    ')

    def test_str_against_container(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Container\n      def f(x: Container[str]):\n        return\n      f("abcdef")  # wrong-arg-types\n    ')

    def test_str_against_mapping(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Mapping\n      def f(x: Mapping[int, str]):\n        return\n      f("abcdef")  # wrong-arg-types\n    ')

    def test_star_unpacking_strings(self):
        if False:
            while True:
                i = 10
        self.Check('\n      *a, b = "hello world"\n    ')

    def test_from_keys(self):
        if False:
            while True:
                i = 10
        self.Check('\n      d = dict.fromkeys(u"x")\n    ')

    def test_filter(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = filter(None, "")\n    ')

    def test_reduce(self):
        if False:
            return 10
        self.Check('\n      x = reduce(lambda x, y: 42, "abcdef")\n    ')

    def test_sorted(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = sorted(u"hello")\n    ')

    def test_iter(self):
        if False:
            print('Hello World!')
        self.Check('\n      x = iter("hello")\n    ')

    def test_zip(self):
        if False:
            return 10
        self.Check('\n      x = zip("abc", "def")\n    ')

    def test_tuple_init(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = tuple("abcdef")\n    ')

    def test_frozenset_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = frozenset("abcdef")\n    ')
if __name__ == '__main__':
    test_base.main()