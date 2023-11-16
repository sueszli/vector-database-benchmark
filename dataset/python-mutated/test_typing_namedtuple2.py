"""Tests for the typing.NamedTuple overlay."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class NamedTupleTest(test_base.BaseTest):
    """Tests for the typing.NamedTuple overlay."""

    def test_make(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n        import typing\n        A = typing.NamedTuple("A", [("b", str), ("c", str)])\n        a = A._make(["hello", "world"])\n        b = A._make(["hello", "world"], len=len)\n        c = A._make([1, 2])  # wrong-arg-types\n        d = A._make(A)  # wrong-arg-types\n        def f(e: A) -> None: pass\n        f(a)\n        ')

    def test_subclass(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n        import typing\n        A = typing.NamedTuple("A", [("b", str), ("c", int)])\n        class B(A):\n          def __new__(cls, b: str, c: int=1):\n            return super(B, cls).__new__(cls, b, c)\n        x = B("hello", 2)\n        y = B("world")\n        def take_a(a: A) -> None: pass\n        def take_b(b: B) -> None: pass\n        take_a(x)\n        take_b(x)\n        take_b(y)\n        take_b(A("", 0))  # wrong-arg-types\n        B()  # missing-parameter\n        # _make and _replace should return instances of the subclass.\n        take_b(B._make(["hello", 0]))\n        take_b(y._replace(b="world"))\n        ')

    def test_callable_attribute(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Callable, NamedTuple\n      X = NamedTuple("X", [("f", Callable)])\n      def foo(x: X):\n        return x.f\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(ty.Lookup('foo')), 'def foo(x: X) -> Callable: ...')

    def test_bare_union_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      from typing import NamedTuple, Union\n      X = NamedTuple("X", [("x", Union)])  # invalid-annotation[e]\n      def foo(x: X):\n        return x.x\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(ty.Lookup('foo')), 'def foo(x: X) -> Any: ...')
        self.assertErrorRegexes(errors, {'e': 'Union.*x'})

    def test_reingest_functional_form(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', "\n      from typing import NamedTuple\n      Foo = NamedTuple('Foo', [('name', str)])\n      Bar = NamedTuple('Bar', [('name', str)])\n      Baz = NamedTuple('Baz', [('foos', list[Foo]), ('bars', list[Bar])])\n    ")]):
            self.Check("\n        import foo\n        foo.Baz([foo.Foo('')], [foo.Bar('')])\n      ")

    def test_kwargs(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.py', '\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        def replace(self, *args, **kwargs):\n          pass\n    ')]):
            self.Check('\n        import foo\n        foo.Foo().replace(x=0)\n      ')

    def test_property(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        x: int\n        @property\n        def y(self):\n          return __any_object__\n        @property\n        def z(self) -> int:\n          return self.x + 1\n    ')]):
            self.Check('\n        import foo\n        nt = foo.Foo(0)\n        assert_type(nt.y, "Any")\n        assert_type(nt.z, "int")\n      ')

    def test_pyi_error(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        x: int\n    ')]):
            errors = self.CheckWithErrors('\n        import foo\n        foo.Foo()  # missing-parameter[e]\n      ')
            self.assertErrorSequences(errors, {'e': 'function foo.Foo.__new__'})

    def test_star_import(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', '\n      from typing import NamedTuple\n      class Foo(NamedTuple): ...\n      def f(x: Foo): ...\n    '), ('bar.py', '\n      from foo import *\n    ')]):
            self.Check('\n        import foo\n        import bar\n        foo.f(bar.Foo())\n      ')

    def test_callback_protocol_as_field(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import NamedTuple, Protocol\n      class Foo(Protocol):\n        def __call__(self, x):\n          return x\n      class Bar(NamedTuple):\n        x: Foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, NamedTuple, Protocol\n      class Foo(Protocol):\n        def __call__(self, x) -> Any: ...\n      class Bar(NamedTuple):\n        x: Foo\n    ')

    def test_custom_new_with_subclasses(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import NamedTuple\n      class Foo(NamedTuple('Foo', [('x', str)])):\n        def __new__(cls, x: str = ''):\n          return super().__new__(cls, x)\n      class Bar(Foo):\n        pass\n      class Baz(Foo):\n        pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import NamedTuple, Type, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      class Foo(NamedTuple):\n        x: str\n        def __new__(cls: Type[_TFoo], x: str = ...) -> _TFoo: ...\n      class Bar(Foo): ...\n      class Baz(Foo): ...\n    ")

    def test_defaults(self):
        if False:
            return 10
        with self.DepTree([('foo.py', '\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: int\n        b: str = ...\n    ')]):
            self.CheckWithErrors("\n        import foo\n        foo.X()  # missing-parameter\n        foo.X(0)\n        foo.X(0, '1')\n        foo.X(0, '1', 'oops')  # wrong-arg-count\n      ")

    def test_override_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', "\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: int\n        b: str\n      X.__new__.__defaults__ = ('',)\n    ")]):
            self.CheckWithErrors("\n        import foo\n        foo.X()  # missing-parameter\n        foo.X(0)\n        foo.X(0, '1')\n        foo.X(0, '1', 'oops')  # wrong-arg-count\n      ")

class NamedTupleTestPy3(test_base.BaseTest):
    """Tests for the typing.NamedTuple overlay in Python 3."""

    def test_basic_namedtuple(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import typing\n      X = typing.NamedTuple("X", [("a", int), ("b", str)])\n      x = X(1, "hello")\n      a = x.a\n      b = x.b\n      ')
        self.assertTypesMatchPytd(ty, '\n        import typing\n        from typing import NamedTuple\n        a: int\n        b: str\n        x: X\n        class X(NamedTuple):\n          a: int\n          b: str\n        ')

    def test_union_attribute(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import NamedTuple, Union\n      X = NamedTuple("X", [("x", Union[bytes, str])])\n      def foo(x: X):\n        return x.x\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(ty.Lookup('foo')), 'def foo(x: X) -> Union[bytes, str]: ...')

    def test_bad_call(self):
        if False:
            while True:
                i = 10
        (_, errorlog) = self.InferWithErrors("\n        from typing import NamedTuple\n        E2 = NamedTuple('Employee2', [('name', str), ('id', int)],  # invalid-namedtuple-arg[e1]  # wrong-keyword-args[e2]\n                        birth=str, gender=bool)\n    ")
        self.assertErrorRegexes(errorlog, {'e1': 'Either list of fields or keywords.*', 'e2': '.*(birth, gender).*NamedTuple'})

    def test_bad_attribute(self):
        if False:
            print('Hello World!')
        (_, errorlog) = self.InferWithErrors('\n        from typing import NamedTuple\n\n        class SubCls(NamedTuple):  # not-writable[e]\n          def __init__(self):\n            pass\n    ')
        self.assertErrorRegexes(errorlog, {'e': ".*'__init__'.*[SubCls]"})

    def test_bad_arg_count(self):
        if False:
            while True:
                i = 10
        (_, errorlog) = self.InferWithErrors('\n        from typing import NamedTuple\n\n        class SubCls(NamedTuple):\n          a: int\n          b: int\n\n        cls1 = SubCls(5)  # missing-parameter[e]\n    ')
        self.assertErrorRegexes(errorlog, {'e': "Missing.*'b'.*__new__"})

    def test_bad_arg_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n        from typing import NamedTuple\n\n        class SubCls(NamedTuple):  # invalid-namedtuple-arg\n          _a: int\n          b: int\n\n        cls1 = SubCls(5)\n    ')

    def test_namedtuple_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import NamedTuple\n\n      class SubNamedTuple(NamedTuple):\n        a: int\n        b: str ="123"\n        c: int = 123\n\n        def __repr__(self) -> str:\n          return "__repr__"\n\n        def func():\n          pass\n\n      tuple1 = SubNamedTuple(5)\n      tuple2 = SubNamedTuple(5, "123")\n      tuple3 = SubNamedTuple(5, "123", 123)\n\n      E1 = NamedTuple(\'Employee1\', name=str, id=int)\n      E2 = NamedTuple(\'Employee2\', [(\'name\', str), (\'id\', int)])\n      ')

    def test_baseclass(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import NamedTuple\n\n      class baseClass:\n        x=5\n        y=6\n\n      class SubNamedTuple(baseClass, NamedTuple):\n        a: int\n      ')
        self.assertTypesMatchPytd(ty, '\n        from typing import NamedTuple\n\n        class SubNamedTuple(baseClass, NamedTuple):\n            a: int\n\n        class baseClass:\n            x = ...  # type: int\n            y = ...  # type: int\n        ')

    def test_fields(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: str\n        b: int\n\n      v = X("answer", 42)\n      a = v.a  # type: str\n      b = v.b  # type: int\n      ')

    def test_field_wrong_type(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: str\n        b: int\n\n      v = X("answer", 42)\n      a_int = v.a  # type: int  # annotation-type-mismatch\n      ')

    def test_unpacking(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        class X(NamedTuple):\n          a: str\n          b: int\n      ')
            (ty, unused_errorlog) = self.InferWithErrors('\n        import foo\n        v = None  # type: foo.X\n        a, b = v\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        v = ...  # type: foo.X\n        a = ...  # type: str\n        b = ...  # type: int\n      ')

    def test_bad_unpacking(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        class X(NamedTuple):\n          a: str\n          b: int\n      ')
            self.CheckWithErrors('\n        import foo\n        v = None  # type: foo.X\n        _, _, too_many = v  # bad-unpacking\n        too_few, = v  # bad-unpacking\n        a: float\n        b: str\n        a, b = v  # annotation-type-mismatch # annotation-type-mismatch\n      ', deep=False, pythonpath=[d.path])

    def test_is_tuple_type_and_superclasses(self):
        if False:
            while True:
                i = 10
        'Test that a NamedTuple (function syntax) behaves like a tuple.'
        self.Check('\n      from typing import MutableSequence, NamedTuple, Sequence, Tuple, Union\n      class X(NamedTuple):\n        a: int\n        b: str\n\n      a = X(1, "2")\n      a_tuple = a  # type: tuple\n      a_typing_tuple = a  # type: Tuple[int, str]\n      a_typing_tuple_elipses = a  # type: Tuple[Union[int, str], ...]\n      a_sequence = a  # type: Sequence[Union[int, str]]\n      a_iter = iter(a)  # type: tupleiterator[Union[int, str]]\n\n      a_first = a[0]  # type: int\n      a_second = a[1]  # type: str\n      a_first_next = next(iter(a))  # We don\'t know the type through the iter() function\n    ')

    def test_is_not_incorrect_types(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import MutableSequence, NamedTuple, Sequence, Tuple, Union\n      class X(NamedTuple):\n        a: int\n        b: str\n\n      x = X(1, "2")\n\n      x_wrong_tuple_types = x  # type: Tuple[str, str]  # annotation-type-mismatch\n      x_not_a_list = x  # type: list  # annotation-type-mismatch\n      x_not_a_mutable_seq = x  # type: MutableSequence[Union[int, str]]  # annotation-type-mismatch\n      x_first_wrong_element_type = x[0]  # type: str  # annotation-type-mismatch\n    ')

    def test_meets_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import NamedTuple, Protocol\n      class X(NamedTuple):\n        a: int\n        b: str\n\n      class IntAndStrHolderVars(Protocol):\n        a: int\n        b: str\n\n      class IntAndStrHolderProperty(Protocol):\n        @property\n        def a(self) -> int:\n          ...\n\n        @property\n        def b(self) -> str:\n          ...\n\n      a = X(1, "2")\n      a_vars_protocol: IntAndStrHolderVars = a\n      a_property_protocol: IntAndStrHolderProperty = a\n    ')

    def test_does_not_meet_mismatching_protocol(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing import NamedTuple, Protocol\n      class X(NamedTuple):\n        a: int\n        b: str\n\n      class DualStrHolder(Protocol):\n        a: str\n        b: str\n\n      class IntAndStrHolderVars_Alt(Protocol):\n        the_number: int\n        the_string: str\n\n      class IntStrIntHolder(Protocol):\n        a: int\n        b: str\n        c: int\n\n      a = X(1, "2")\n      a_wrong_types: DualStrHolder = a  # annotation-type-mismatch\n      a_wrong_names: IntAndStrHolderVars_Alt = a  # annotation-type-mismatch\n      a_too_many: IntStrIntHolder = a  # annotation-type-mismatch\n    ')

    def test_generated_members(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import NamedTuple\n      class X(NamedTuple):\n        a: int\n        b: str\n      ')
        self.assertTypesMatchPytd(ty, '\n        from typing import NamedTuple\n\n        class X(NamedTuple):\n            a: int\n            b: str\n        ')

    def test_namedtuple_with_defaults(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import NamedTuple\n\n      class SubNamedTuple(NamedTuple):\n        a: int\n        b: str ="123"\n        c: int = 123\n\n        def __repr__(self) -> str:\n          return "__repr__"\n\n        def func():\n          pass\n\n      X = SubNamedTuple(1, "aaa", 222)\n      a = X.a\n      b = X.b\n      c = X.c\n      f = X.func\n\n      Y = SubNamedTuple(1)\n      a2 = Y.a\n      b2 = Y.b\n      c2 = Y.c\n      ')
        self.assertTypesMatchPytd(ty, '\n        from typing import NamedTuple\n\n        X: SubNamedTuple\n        a: int\n        b: str\n        c: int\n\n        Y: SubNamedTuple\n        a2: int\n        b2: str\n        c2: int\n\n        class SubNamedTuple(NamedTuple):\n            a: int\n            b: str = ...\n            c: int = ...\n            def __repr__(self) -> str: ...\n            def func() -> None: ...\n\n        def f() -> None: ...\n        ')

    def test_bad_default(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        x: str = 0  # annotation-type-mismatch[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: str.*Assignment: int'})

    def test_nested_namedtuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNoCrash(self.Check, '\n      from typing import NamedTuple\n\n      def foo() -> None:\n        class A(NamedTuple):\n          x: int\n\n      def bar():\n        foo()\n    ')

    def test_generic_namedtuple(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import Callable, Generic, NamedTuple, TypeVar\n\n      def _int_identity(x: int) -> int:\n        return x\n\n      T = TypeVar(\'T\')\n\n      class Foo(NamedTuple, Generic[T]):\n        x: T\n        y: Callable[[T], T]\n      foo_int = Foo(x=0, y=_int_identity)\n      x_out = foo_int.x\n      y_out = foo_int.y\n      y_call_out = foo_int.y(2)\n      foo_str: Foo[str] = Foo(x="hi", y=__any_object__)\n    ')
        self.assertTypesMatchPytd(ty, '\n        from typing import Callable, Generic, NamedTuple, TypeVar\n\n\n        def _int_identity(x: int) -> int: ...\n\n        T = TypeVar("T")\n\n        foo_int = ...  # type: Foo[int]\n        x_out = ...  # type: int\n        y_out = ...  # type: Callable[[int], int]\n        y_call_out = ...  # type: int\n        foo_str = ...  # type: Foo[str]\n\n        class Foo(NamedTuple, Generic[T]):\n          x: T\n          y: Callable[[T], T]\n      ')

    def test_bad_typevar(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Generic, NamedTuple, TypeVar\n      T = TypeVar('T')\n      class Foo(NamedTuple):\n        x: T  # invalid-annotation\n    ")

    def test_generic_callable(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Callable, NamedTuple, TypeVar\n      T = TypeVar('T')\n      class Foo(NamedTuple):\n        f: Callable[[T], T]\n      assert_type(Foo(f=__any_object__).f(''), str)\n    ")

    def test_reingest(self):
        if False:
            while True:
                i = 10
        foo_ty = self.Infer("\n      from typing import Callable, Generic, NamedTuple, TypeVar\n      T = TypeVar('T')\n      class Foo(NamedTuple, Generic[T]):\n        x: T\n      class Bar(NamedTuple):\n        x: Callable[[T], T]\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.Check('\n        import foo\n        assert_type(foo.Foo(x=0).x, int)\n        assert_type(foo.Bar(x=__any_object__).x(0), int)\n      ', pythonpath=[d.path])

    def test_recursive_tuple(self):
        if False:
            print('Hello World!')
        'Regression test for a recursive tuple containing a namedtuple.'
        self.Check('\n      from typing import Any, NamedTuple\n\n      A = NamedTuple("A", [("x", Any), ("y", Any)])\n\n      def decorator(fn):\n        def wrapper(*args, **kwargs):\n          return fn(*args, **kwargs)\n        return wrapper\n\n      @decorator\n      def g(x, y):\n        nt = A(1, 2)\n        x = x, nt\n        y = y, nt\n        def h():\n          max(x, y)\n    ')

    def test_override_method(self):
        if False:
            i = 10
            return i + 15
        foo_pyi = '\n      from typing import NamedTuple\n      class Foo(NamedTuple):\n        a: float\n        b: str\n        def __repr__(self) -> str: ...\n    '
        with self.DepTree([('foo.pyi', foo_pyi)]):
            self.Check("\n        import foo\n        class Bar(foo.Foo):\n          def __repr__(self):\n            return super().__repr__()\n        x = Bar(1, '2')\n        y = x.__repr__()\n      ")

class NamedTupleFunctionSubclassTest(test_base.BaseTest):
    """Tests for subclassing an anonymous NamedTuple in a different module."""

    def test_class_method(self):
        if False:
            return 10
        foo_py = '\n      from typing import NamedTuple\n\n      class A(NamedTuple("A", [("x", int)])):\n        @classmethod\n        def make(cls, x):\n          return cls(x)\n    '
        with self.DepTree([('foo.py', foo_py)]):
            self.Check('\n        import foo\n        x = foo.A.make(10)\n      ')

    def test_class_constant(self):
        if False:
            i = 10
            return i + 15
        foo_py = '\n      from typing import NamedTuple\n\n      class A(NamedTuple("A", [("x", int)])):\n        pass\n\n      A.Foo = A(10)\n    '
        with self.DepTree([('foo.py', foo_py)]):
            self.Check('\n        import foo\n        x = foo.A.Foo\n        assert_type(x, foo.A)\n      ')
if __name__ == '__main__':
    test_base.main()