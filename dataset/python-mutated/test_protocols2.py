"""Tests for matching against protocols.

Based on PEP 544 https://www.python.org/dev/peps/pep-0544/.
"""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class ProtocolTest(test_base.BaseTest):
    """Tests for protocol implementation."""

    def test_check_protocol(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import protocols\n      from typing import Sized\n      def f(x: protocols.Sized):\n        return None\n      def g(x: Sized):\n        return None\n      class Foo:\n        def __len__(self):\n          return 5\n      f([])\n      foo = Foo()\n      f(foo)\n      g([])\n      g(foo)\n    ')

    def test_check_protocol_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      import protocols\n\n      def f(x: protocols.SupportsAbs):\n        return x.__abs__()\n      f(["foo"])  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\(x: SupportsAbs\\).*\\(x: List\\[str\\]\\)'})

    def test_check_iterator_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors("\n      from typing import Iterator\n      def f(x: Iterator[int]):\n        return None\n      class Foo:\n        def next(self) -> str:\n          return ''\n        def __iter__(self):\n          return self\n      f(Foo())  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Iterator\\[int\\].*Foo'})

    def test_check_protocol_match_unknown(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Sized\n      def f(x: Sized):\n        pass\n      class Foo:\n        pass\n      def g(x):\n        foo = Foo()\n        foo.__class__ = x\n        f(foo)\n    ')

    def test_check_parameterized_protocol(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Iterator, Iterable\n\n      class Foo:\n        def __iter__(self) -> Iterator[int]:\n          return iter([])\n\n      def f(x: Iterable[int]):\n        pass\n\n      foo = Foo()\n      f(foo)\n      f(iter([3]))\n    ')

    def test_check_parameterized_protocol_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import Iterator, Iterable\n\n      class Foo:\n        def __iter__(self) -> Iterator[str]:\n          return iter([])\n\n      def f(x: Iterable[int]):\n        pass\n\n      foo = Foo()\n      f(foo)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\(x: Iterable\\[int\\]\\).*\\(x: Foo\\)'})

    def test_check_parameterized_protocol_multi_signature(self):
        if False:
            return 10
        self.Check('\n      from typing import Sequence, Union\n\n      class Foo:\n        def __len__(self):\n          return 0\n        def __getitem__(self, x: Union[int, slice]) -> Union[int, Sequence[int]]:\n          return 0\n\n      def f(x: Sequence[int]):\n        pass\n\n      foo = Foo()\n      f(foo)\n    ')

    def test_check_parameterized_protocol_error_multi_signature(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import Sequence, Union\n\n      class Foo:\n        def __len__(self):\n          return 0\n        def __getitem__(self, x: int) -> int:\n          return 0\n\n      def f(x: Sequence[int]):\n        pass\n\n      foo = Foo()\n      f(foo)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\(x: Sequence\\[int\\]\\).*\\(x: Foo\\)'})

    def test_construct_dict_with_protocol(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __iter__(self):\n          pass\n      def f(x: Foo):\n        return dict(x)\n    ')

    def test_method_on_superclass(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __iter__(self):\n          pass\n      class Bar(Foo):\n        pass\n      def f(x: Bar):\n        return iter(x)\n    ')

    def test_method_on_parameterized_superclass(self):
        if False:
            return 10
        self.Check('\n      from typing import List\n      class Bar(List[int]):\n        pass\n      def f(x: Bar):\n        return iter(x)\n    ')

    def test_any_superclass(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Bar(__any_object__):\n        pass\n      def f(x: Bar):\n        return iter(x)\n    ')

    def test_multiple_options(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Bar:\n        if __random__:\n          def __iter__(self): return 1\n        else:\n          def __iter__(self): return 2\n      def f(x: Bar):\n        return iter(x)\n    ')

    def test_iterable_getitem(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Iterable, Iterator, TypeVar\n      T = TypeVar("T")\n      class Bar:\n        def __getitem__(self, i: T) -> T:\n          if i > 10:\n            raise IndexError()\n          return i\n      T2 = TypeVar("T2")\n      def f(s: Iterable[T2]) -> Iterator[T2]:\n        return iter(s)\n      next(f(Bar()))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterable, Iterator, TypeVar\n      T = TypeVar("T")\n      class Bar:\n        def __getitem__(self, i: T) -> T: ...\n      T2 = TypeVar("T2")\n      def f(s: Iterable[T2]) -> Iterator[T2]: ...\n    ')

    def test_iterable_iter(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Iterable, Iterator, TypeVar\n      class Bar:\n        def __iter__(self) -> Iterator:\n          return iter([])\n      T = TypeVar("T")\n      def f(s: Iterable[T]) -> Iterator[T]:\n        return iter(s)\n      next(f(Bar()))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterable, Iterator, TypeVar\n      class Bar:\n        def __iter__(self) -> Iterator: ...\n      T = TypeVar("T")\n      def f(s: Iterable[T]) -> Iterator[T]: ...\n    ')

    def test_pyi_iterable_getitem(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        T = TypeVar("T")\n        class Foo:\n          def __getitem__(self, i: T) -> T: ...\n      ')
            self.Check('\n        from typing import Iterable, TypeVar\n        import foo\n        T = TypeVar("T")\n        def f(s: Iterable[T]) -> T: ...\n        f(foo.Foo())\n      ', pythonpath=[d.path])

    def test_pyi_iterable_iter(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        class Foo:\n          def __iter__(self) -> Any: ...\n      ')
            self.Check('\n        from typing import Iterable, TypeVar\n        import foo\n        T = TypeVar("T")\n        def f(s: Iterable[T]) -> T: ...\n        f(foo.Foo())\n      ', pythonpath=[d.path])

    def test_inherited_abstract_method_error(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import Iterator\n      class Foo:\n        def __iter__(self) -> Iterator[str]:\n          return __any_object__\n        def next(self):\n          return __any_object__\n      def f(x: Iterator[int]):\n        pass\n      f(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Iterator\\[int\\].*Foo'})

    def test_reversible(self):
        if False:
            return 10
        self.Check('\n      from typing import Reversible\n      class Foo:\n        def __reversed__(self):\n          pass\n      def f(x: Reversible):\n        pass\n      f(Foo())\n    ')

    def test_collection(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Collection\n      class Foo:\n        def __contains__(self, x):\n          pass\n        def __iter__(self):\n          pass\n        def __len__(self):\n          pass\n      def f(x: Collection):\n        pass\n      f(Foo())\n    ')

    def test_list_against_collection(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Collection\n      def f() -> Collection[str]:\n        return [""]\n    ')

    def test_hashable(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Hashable\n      class Foo:\n        def __hash__(self):\n          pass\n      def f(x: Hashable):\n        pass\n      f(Foo())\n    ')

    def test_list_hash(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Hashable\n      def f(x: Hashable):\n        pass\n      f([])  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Hashable.*List.*__hash__'})

    def test_hash_constant(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import Hashable\n      class Foo:\n        __hash__ = None\n      def f(x: Hashable):\n        pass\n      f(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Hashable.*Foo.*__hash__'})

    def test_hash_type(self):
        if False:
            return 10
        self.Check('\n      from typing import Hashable, Type\n      def f(x: Hashable):\n        pass\n      def g(x: Type[int]):\n        return f(x)\n    ')

    def test_hash_module(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import subprocess\n      from typing import Hashable\n      def f(x: Hashable):\n        pass\n      f(subprocess)\n    ')

    def test_generic_callable(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class Foo(Generic[T]):\n          def __init__(self, x: T):\n            self = Foo[T]\n          def __call__(self) -> T: ...\n      ')
            errors = self.CheckWithErrors('\n        from typing import Any, Callable\n        import foo\n        def f() -> Callable:\n          return foo.Foo("")\n        def g() -> Callable[[], str]:\n          return foo.Foo("")\n        def h() -> Callable[[Any], str]:\n          return foo.Foo("")  # bad-return-type[e1]\n        def i() -> Callable[[], int]:\n          return foo.Foo("")  # bad-return-type[e2]\n      ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e1': ['def <callable>(self, _0) -> str: ...', 'def __call__(self: foo.Foo[T]) -> T: ...'], 'e2': ['def <callable>(self) -> int: ...', 'def __call__(self: foo.Foo[T]) -> T: ...']})

    def test_staticmethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import Any, Callable, Protocol\n\n      class MyProtocol(Protocol):\n        @staticmethod\n        def __call__(a, b) -> int:\n          return 0\n\n      def f() -> MyProtocol:\n        return __any_object__\n\n      def g1(x: Callable[[Any, Any], int]):\n        pass\n      def g2(x: Callable[[Any], int]):\n        pass\n      def g3(x: Callable[[Any, Any, Any], int]):\n        pass\n      def g4(x: Callable[[Any, Any], str]):\n        pass\n\n      g1(f())  # ok\n      g2(f())  # wrong-arg-types  # too few Callable args\n      g3(f())  # wrong-arg-types  # too many Callable args\n      g3(f())  # wrong-arg-types  # wrong Callable return\n    ')

    def test_protocol_caching(self):
        if False:
            return 10
        self.Check('\n      import collections\n      from typing import Text\n\n      class _PortInterface:\n\n        def __init__(self):\n          self._flattened_ports = collections.OrderedDict()\n\n        def PortBundle(self, prefix: Text, bundle):\n          for name, port in bundle.ports.items():\n            full_name = prefix + "_" + name\n            self._flattened_ports[full_name] = port\n\n        def _GetPortsWithDirection(self):\n          return collections.OrderedDict(\n              (name, port) for name, port in self._flattened_ports.items())\n    ')

    def test_custom_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self):\n          pass\n      class MyAppendable:\n        def append(self):\n          pass\n      def f(x: Appendable):\n        pass\n      f([])\n      f(MyAppendable())\n    ')

    def test_custom_protocol_error(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self):\n          pass\n      class NotAppendable:\n        pass\n      def f(x: Appendable):\n        pass\n      f(42)  # wrong-arg-types[e1]\n      f(NotAppendable())  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Appendable.*int.*append', 'e2': 'Appendable.*NotAppendable.*append'})

    def test_reingest_custom_protocol(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self) -> None:\n          pass\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            self.Check('\n        import foo\n        class MyAppendable:\n          def append(self):\n            pass\n        def f(x: foo.Appendable):\n          pass\n        f([])\n        f(MyAppendable())\n      ', pythonpath=[d.path])

    def test_reingest_custom_protocol_error(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self) -> None:\n          pass\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            errors = self.CheckWithErrors('\n        import foo\n        class NotAppendable:\n          pass\n        def f(x: foo.Appendable):\n          pass\n        f(42)  # wrong-arg-types[e1]\n        f(NotAppendable())  # wrong-arg-types[e2]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'Appendable.*int.*append', 'e2': 'Appendable.*NotAppendable.*append'})

    def test_reingest_custom_protocol_inherit_method(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self):\n          pass\n      class Mutable(Appendable, Protocol):\n        def remove(self):\n          pass\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            errors = self.CheckWithErrors('\n        from foo import Mutable\n        class NotMutable:\n          def remove(self):\n            pass\n        def f(x: Mutable):\n          pass\n        f([])  # ok\n        f(NotMutable())  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'Mutable.*NotMutable.*append'})

    def test_reingest_custom_protocol_implement_method(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing_extensions import Protocol\n      class Appendable(Protocol):\n        def append(self):\n          pass\n      class Mixin:\n        def append(self):\n          pass\n      class Removable(Mixin, Appendable, Protocol):\n        def remove(self):\n          pass\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            self.Check('\n        from foo import Removable\n        def f(x: Removable):\n          pass\n        class MyRemovable:\n          def remove(self):\n            pass\n        f(MyRemovable())\n      ', pythonpath=[d.path])

    def test_ignore_method_body(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Protocol\n      class Countable(Protocol):\n        def count(self) -> int:\n          ...\n    ')

    def test_check_method_body(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import Protocol\n      class Countable(Protocol):\n        def count(self) -> int:\n          ...  # bad-return-type[e]\n      class MyCountable(Countable):\n        def count(self):\n          return super(MyCountable, self).count()\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*None.*line 7'})

    def test_callback_protocol(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing_extensions import Protocol\n      class Foo(Protocol):\n        def __call__(self) -> int:\n          return 0\n\n      def f1() -> int:\n        return 0\n      def f2(x) -> int:\n        return x\n      def f3() -> str:\n        return ''\n\n      def accepts_foo(f: Foo):\n        pass\n\n      accepts_foo(f1)\n      accepts_foo(f2)  # wrong-arg-types\n      accepts_foo(f3)  # wrong-arg-types\n    ")

    def test_callback_protocol_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Protocol\n        class Foo(Protocol):\n          def __call__(self, x: str) -> str: ...\n        def accepts_foo(f: Foo) -> None: ...\n      ')
            self.CheckWithErrors("\n        import foo\n        def f1(x: str) -> str:\n          return x\n        def f2() -> str:\n          return ''\n        def f3(x: int) -> str:\n          return str(x)\n\n        foo.accepts_foo(f1)\n        foo.accepts_foo(f2)  # wrong-arg-types\n        foo.accepts_foo(f3)  # wrong-arg-types\n      ", pythonpath=[d.path])

    def test_class_matches_callback_protocol(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing_extensions import Protocol\n      class Foo(Protocol):\n        def __call__(self) -> int:\n          return 0\n      def accepts_foo(f: Foo):\n        pass\n\n      accepts_foo(int)\n      accepts_foo(str)  # wrong-arg-types\n    ')

    def test_class_matches_callback_protocol_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Protocol\n        class Foo(Protocol):\n          def __call__(self) -> int: ...\n        def accepts_foo(f: Foo) -> None: ...\n      ')
            self.CheckWithErrors('\n        import foo\n        foo.accepts_foo(int)\n        foo.accepts_foo(str)  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_classmethod(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import Protocol\n      class Foo(Protocol):\n        @classmethod\n        def f(cls):\n          return cls()\n      class Bar:\n        @classmethod\n        def f(cls):\n          return cls()\n      class Baz:\n        def f(self):\n          return type(self)\n      class Qux:\n        pass\n      def f(x: Foo):\n        pass\n      f(Bar())\n      f(Baz())\n      f(Qux())  # wrong-arg-types\n    ')

    def test_abstractmethod(self):
        if False:
            return 10
        self.CheckWithErrors('\n      import abc\n      from typing import Protocol\n\n      class Foo(Protocol):\n        @abc.abstractmethod\n        def f(self) -> int:\n          pass\n\n      class Bar:\n        def f(self):\n          pass\n\n      class Baz:\n        pass\n\n      def f(x: Foo):\n        pass\n\n      f(Bar())\n      f(Baz())  # wrong-arg-types\n    ')

    def test_decorated_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable\n      from typing_extensions import Protocol\n      class Foo(Protocol):\n        def foo(self):\n          pass\n      def decorate(f: Callable) -> Callable:\n        return f\n      class Bar:\n        @decorate\n        def foo(self):\n          pass\n      def accept(foo: Foo):\n        pass\n      accept(Bar())\n    ')

    def test_len(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Generic, Protocol, TypeVar\n      T = TypeVar('T')\n      class SupportsLen(Generic[T], Protocol):\n        def __len__(self) -> int: ...\n      def f() -> SupportsLen[int]:\n        return [1, 2, 3]\n    ")

    def test_property(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Protocol\n      class Foo(Protocol):\n        @property\n        def name(self) -> str: ...\n        def f(self) -> int: ...\n    ')

    def test_has_dynamic_attributes(self):
        if False:
            return 10
        self.Check('\n      from typing import Protocol\n      class Foo(Protocol):\n        def f(self) -> int: ...\n      class Bar:\n        _HAS_DYNAMIC_ATTRIBUTES = True\n      def f(x: Foo):\n        pass\n      f(Bar())\n    ')

    def test_empty(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Protocol\n      class Foo(Protocol):\n        pass\n      class Bar:\n        pass\n      def f(foo: Foo):\n        pass\n      f(Bar())\n    ')

    def test_empty_and_generic(self):
        if False:
            while True:
                i = 10
        self.Check("\n      from typing import Protocol, TypeVar\n      T = TypeVar('T')\n      class Foo(Protocol[T]):\n        pass\n      class Bar:\n        pass\n      def f(foo: Foo[int]):\n        pass\n      f(Bar())\n    ")

    def test_deduplicate_error_message(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import Callable, Iterable, Optional, Union\n\n      DistanceFunctionsType = Iterable[Union[Callable[[str, str], float], str]]\n\n      def f(x: DistanceFunctionsType) -> DistanceFunctionsType:\n        return (x,)  # bad-return-type[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Actually returned[^\\n]*\\nAttributes[^\\n]*$'})

    def test_annotated_classmethod(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Protocol\n      class Foo(Protocol):\n        @classmethod\n        def f(cls) -> str: ...\n    ')

    def test_typing_extensions_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import SupportsIndex\n      def f(x: SupportsIndex):\n        pass\n      f(0)\n    ')

    def test_not_instantiable(self):
        if False:
            return 10
        self.CheckWithErrors('\n      import abc\n      from typing import Protocol\n\n      class MyProtocol(Protocol):\n        @abc.abstractmethod\n        def f(self): ...\n\n      class Child(MyProtocol):\n        pass\n\n      Child()  # not-instantiable\n    ')

    def test_substitute_typevar(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Protocol, TypeVar\n      _T = TypeVar('_T')\n      _T_int = TypeVar('_T_int', bound=int)\n      class MyProtocol(Protocol[_T]):\n        def __getitem__(self, __k: int) -> _T: ...\n      def f(x: MyProtocol[_T_int]) -> _T_int:\n        return x[0]\n      f([0])\n      f([])\n    ")

class ProtocolsTestPython3Feature(test_base.BaseTest):
    """Tests for protocol implementation on a target using a Python 3 feature."""

    def test_check_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Iterator\n      def f(x: Iterator):\n        return None\n      class Foo:\n        def __next__(self):\n          return None\n        def __iter__(self):\n          return None\n      foo = Foo()\n      f(foo)\n    ')

    def test_check_parameterized_iterator(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Iterator\n      def f(x: Iterator[int]):\n        return None\n      class Foo:\n        def __next__(self):\n          return 42\n        def __iter__(self):\n          return self\n      f(Foo())\n    ')

    def test_inherited_abstract_method(self):
        if False:
            return 10
        self.Check('\n      from typing import Iterator\n      class Foo:\n        def __iter__(self) -> Iterator[int]:\n          return __any_object__\n        def __next__(self):\n          return __any_object__\n      def f(x: Iterator[int]):\n        pass\n      f(Foo())\n    ')

    def test_check_supports_bytes_protocol(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import protocols\n      from typing import SupportsBytes\n      def f(x: protocols.SupportsBytes):\n        return None\n      def g(x: SupportsBytes):\n        return None\n      class Foo:\n        def __bytes__(self):\n          return b"foo"\n      foo = Foo()\n      f(foo)\n      g(foo)\n    ')

    def test_metaclass_abstractness(self):
        if False:
            print('Hello World!')
        self.Check('\n      import abc\n      from typing import Protocol\n      class Meta1(type(Protocol)):\n        pass\n      class Meta2(Protocol.__class__):\n        pass\n      class Foo(metaclass=Meta1):\n        @abc.abstractmethod\n        def foo(self):\n          pass\n      class Bar(metaclass=Meta2):\n        @abc.abstractmethod\n        def bar(self):\n          pass\n    ')

    def test_module(self):
        if False:
            print('Hello World!')
        foo_ty = self.Infer("\n      x: int\n      def f() -> str:\n        return 'hello world'\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            errors = self.CheckWithErrors('\n        import foo\n        from typing import Protocol\n        class ShouldMatch(Protocol):\n          x: int\n          def f(self) -> str: ...\n        class ExtraAttribute(Protocol):\n          x: int\n          y: str\n        class ExtraMethod(Protocol):\n          def f(self) -> str: ...\n          def g(self) -> int: ...\n        class WrongType(Protocol):\n          x: str\n        def should_match(x: ShouldMatch):\n          pass\n        def extra_attribute(x: ExtraAttribute):\n          pass\n        def extra_method(x: ExtraMethod):\n          pass\n        def wrong_type(x: WrongType):\n          pass\n        should_match(foo)\n        extra_attribute(foo)  # wrong-arg-types[e1]\n        extra_method(foo)  # wrong-arg-types[e2]\n        wrong_type(foo)  # wrong-arg-types[e3]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'not implemented on module: y', 'e2': 'not implemented on module: g', 'e3': 'x.*expected str, got int'})

class ProtocolAttributesTest(test_base.BaseTest):
    """Tests for non-method protocol attributes."""

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Protocol\n      class Foo(Protocol):\n        x: int\n      class Bar:\n        x: int\n      class Baz:\n        x: str\n      def f(foo: Foo):\n        pass\n      f(Bar())\n      f(Baz())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'x.*expected int, got str'})

    def test_missing(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Protocol\n      class Foo(Protocol):\n        x: int\n        y: str\n      class Bar:\n        y = ''\n      def f(foo: Foo):\n        pass\n      f(Bar())  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Foo.*Bar.*x'})

    def test_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Protocol\n        class Foo(Protocol):\n          x: int\n      ')
            self.CheckWithErrors("\n        import foo\n        class Bar:\n          x = 0\n        class Baz:\n          x = '1'\n        def f(x: foo.Foo):\n          pass\n        f(Bar())\n        f(Baz())  # wrong-arg-types\n      ", pythonpath=[d.path])

    def test_pyi_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          x: int\n      ')
            self.CheckWithErrors('\n        import foo\n        from typing import Protocol\n        class Bar(Protocol):\n          x: int\n        class Baz(Protocol):\n          x: str\n        class Foo2(foo.Foo):\n          pass\n        def f(bar: Bar):\n          pass\n        def g(baz: Baz):\n          pass\n        f(Foo2())\n        g(Foo2())  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_instance_attribute(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors("\n      from typing import Protocol\n      class Foo(Protocol):\n        x: int\n      class Bar:\n        def __init__(self):\n          self.x = 0\n      class Baz:\n        def __init__(self):\n          self.x = ''\n      def f(foo: Foo):\n        pass\n      f(Bar())\n      f(Baz())  # wrong-arg-types\n    ")

    def test_property(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      from typing import Protocol\n      class Foo(Protocol):\n        @property\n        def x(self) -> int: ...\n      class Bar:\n        @property\n        def x(self):\n          return 0\n      class Baz:\n        @property\n        def x(self):\n          return ''\n      def f(foo: Foo):\n        pass\n      f(Bar())\n      f(Baz())  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'x.*expected int, got str'})

    def test_property_in_pyi_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        foo_ty = self.Infer('\n      from typing import Protocol\n      class Foo(Protocol):\n        @property\n        def x(self) -> int: ...\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors("\n        import foo\n        class Bar:\n          @property\n          def x(self):\n            return 0\n        class Baz:\n          @property\n          def x(self):\n            return ''\n        def f(x: foo.Foo):\n          pass\n        f(Bar())\n        f(Baz())  # wrong-arg-types\n      ", pythonpath=[d.path])

    def test_inherit_property(self):
        if False:
            for i in range(10):
                print('nop')
        foo_ty = self.Infer('\n      class Foo:\n        @property\n        def x(self):\n          return 0\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors('\n        import foo\n        from typing import Protocol\n        class Protocol1(Protocol):\n          @property\n          def x(self) -> int: ...\n        class Protocol2(Protocol):\n          @property\n          def x(self) -> str: ...\n        class Bar(foo.Foo):\n          pass\n        def f1(x: Protocol1):\n          pass\n        def f2(x: Protocol2):\n          pass\n        f1(Bar())\n        f2(Bar())  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_optional(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      from typing import Optional, Protocol\n      class Foo(Protocol):\n        x: Optional[int]\n      class Bar:\n        x = 0\n      class Baz:\n        x = ''\n      def f(x: Foo):\n        pass\n      f(Bar())\n      f(Baz())  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected Optional\\[int\\], got str'})

    def test_match_optional_to_optional(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Optional, Protocol\n      class Foo(Protocol):\n        x: Optional[int]\n      class Bar:\n        def __init__(self, x: Optional[int]):\n          self.x = x\n      def f(x: Foo):\n        pass\n      f(Bar(0))\n    ')

    def test_generic(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Generic, Protocol, Type, TypeVar\n\n      T = TypeVar('T')\n      class Foo(Protocol[T]):\n        x: T\n\n      T2 = TypeVar('T2', bound=Foo[int])\n      def f(cls: Type[T2]) -> T2:\n        return cls()\n\n      class Bar:\n        x = 0\n      class Baz:\n        x = ''\n\n      f(Bar)  # ok\n      f(Baz)  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected int, got str'})

    def test_generic_from_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Protocol, TypeVar\n        T = TypeVar('T')\n        class Foo(Protocol[T]):\n          x: T\n      ")
            errors = self.CheckWithErrors("\n        from typing import Type, TypeVar\n        import foo\n\n        T = TypeVar('T', bound=foo.Foo[int])\n        def f(cls: Type[T]) -> T:\n          return cls()\n\n        class Bar:\n          x = 0\n        class Baz:\n          x = ''\n\n        f(Bar)  # ok\n        f(Baz)  # wrong-arg-types[e]\n      ", pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'expected int, got str'})

    def test_generic_used_in_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('protocol.pyi', "\n        from typing import Dict, List, Protocol, TypeVar\n        T = TypeVar('T')\n        class Foo(Protocol[T]):\n          x: Dict[str, List[T]]\n    ")
            d.create_file('util.pyi', "\n        import protocol\n        from typing import Type, TypeVar\n        T = TypeVar('T', bound=protocol.Foo[int])\n        def f(x: Type[T]) -> T: ...\n      ")
            errors = self.CheckWithErrors('\n        from typing import Dict, List\n        import util\n        class Bar:\n          x: Dict[str, List[int]]\n        class Baz:\n          x: Dict[str, List[str]]\n        util.f(Bar)  # ok\n        util.f(Baz)  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'expected Dict\\[str, List\\[int\\]\\], got Dict\\[str, List\\[str\\]\\]'})

    def test_match_multi_attributes_against_dataclass_protocol(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      from typing import Dict, Protocol, TypeVar, Union\n      import dataclasses\n      T = TypeVar('T')\n      class Dataclass(Protocol[T]):\n        __dataclass_fields__: Dict[str, dataclasses.Field[T]]\n      def f(x: Dataclass[int]):\n        pass\n      @dataclasses.dataclass\n      class ShouldMatch:\n        x: int\n        y: int\n      @dataclasses.dataclass\n      class ShouldNotMatch:\n        x: int\n        y: str\n      f(ShouldMatch(0, 0))\n      f(ShouldNotMatch(0, ''))  # wrong-arg-types[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'expected Dict\\[str, dataclasses\\.Field\\[int\\]\\], got Dict\\[str, dataclasses\\.Field\\[Union\\[int, str\\]\\]\\]'})
if __name__ == '__main__':
    test_base.main()