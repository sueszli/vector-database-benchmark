"""Tests for the typing.NamedTuple overlay."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class NamedTupleTest(test_base.BaseTest):
    """Tests for the typing.NamedTuple overlay."""

    def test_basic_calls(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import typing\n      Basic = typing.NamedTuple("Basic", [(\'a\', str)])\n      ex = Basic("hello world")\n      ea = ex.a\n      ey = Basic()  # missing-parameter\n      ez = Basic("a", "b")  # wrong-arg-count\n      ')

    def test_optional_field_type(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      import typing\n      X = typing.NamedTuple("X", [(\'a\', str), (\'b\', typing.Optional[int])])\n      xa = X(\'hello\', None)\n      xb = X(\'world\', 2)\n      xc = X(\'nope\', \'2\')  # wrong-arg-types\n      xd = X()  # missing-parameter\n      xe = X(1, "nope")  # wrong-arg-types\n      ')

    def test_class_field_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import typing\n      class Foo:\n        pass\n      Y = typing.NamedTuple("Y", [(\'a\', str), (\'b\', Foo)])\n      ya = Y(\'a\', Foo())\n      yb = Y(\'a\', 1)  # wrong-arg-types\n      yc = Y(Foo())  # missing-parameter\n      yd = Y(1)  # missing-parameter\n      ')

    def test_late_annotation(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      import typing\n      class Foo:\n        pass\n      X = typing.NamedTuple("X", [(\'a\', \'Foo\')]) # should be fine\n      Y = typing.NamedTuple("Y", [(\'a\', \'Bar\')]) # should fail  # name-error[e]\n      ')
        self.assertErrorRegexes(errors, {'e': 'Bar'})

    def test_nested_containers(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import typing\n      Z = typing.NamedTuple("Z", [(\'a\', typing.List[typing.Optional[int]])])\n      za = Z([1])\n      zb = Z([None, 2])\n      zc = Z(1)  # wrong-arg-types\n\n      import typing\n      A = typing.NamedTuple("A", [(\'a\', typing.Dict[int, str]), (\'b\', typing.Tuple[int, int])])\n      aa = A({1: \'1\'}, (1, 2))\n      ab = A({}, (1, 2))\n      ac = A(1, 2)  # wrong-arg-types\n      ')

    def test_pytd_field(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import typing\n      import datetime\n      B = typing.NamedTuple("B", [(\'a\', datetime.date)])\n      ba = B(datetime.date(1,2,3))\n      bb = B()  # missing-parameter\n      bc = B(1)  # wrong-arg-types\n      ')

    def test_bad_calls(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n        import typing\n        typing.NamedTuple("_", ["abc", "def", "ghi"])  # wrong-arg-types\n        # "def" is a keyword, so the call on the next line fails.\n        typing.NamedTuple("_", [("abc", int), ("def", int), ("ghi", int)])  # invalid-namedtuple-arg\n        typing.NamedTuple("1", [("a", int)])  # invalid-namedtuple-arg\n        typing.NamedTuple("_", [[int, "a"]])  # wrong-arg-types\n        ')

    def test_empty_args(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n        import typing\n        X = typing.NamedTuple("X", [])\n        ')

    def test_tuple_fields(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import NamedTuple\n      X = NamedTuple("X", (("a", str),))\n      X(a="")\n      X(a=42)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

    def test_list_field(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import NamedTuple\n      X = NamedTuple("X", [["a", str]])\n      X(a="")\n      X(a=42)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

    def test_str_fields_error(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import NamedTuple\n      X = NamedTuple("X", "a b")  # wrong-arg-types[e1]\n      Y = NamedTuple("Y", ["ab"])  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Tuple.*str', 'e2': 'Tuple.*str'})

    def test_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable, NamedTuple, TypeVar\n      T = TypeVar(\'T\')\n      X = NamedTuple("X", [("f", Callable[[T], T])])\n      assert_type(X(f=__any_object__).f(""), str)\n    ')

    def test_bad_typevar(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing import NamedTuple, TypeVar\n      T = TypeVar(\'T\')\n      X = NamedTuple("X", [("a", T)])  # invalid-annotation\n    ')

    def test_reingest(self):
        if False:
            i = 10
            return i + 15
        foo_ty = self.Infer('\n      from typing import Callable, NamedTuple, TypeVar\n      T = TypeVar(\'T\')\n      X = NamedTuple("X", [("a", Callable[[T], T])])\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.Check('\n        import foo\n        assert_type(foo.X(a=__any_object__).a(4.2), float)\n      ', pythonpath=[d.path])

    def test_fields(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import NamedTuple\n      X = NamedTuple("X", [(\'a\', str), (\'b\', int)])\n\n      v = X("answer", 42)\n      a = v.a  # type: str\n      b = v.b  # type: int\n      ')

    def test_field_wrong_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n        from typing import NamedTuple\n        X = NamedTuple("X", [(\'a\', str), (\'b\', int)])\n\n        v = X("answer", 42)\n        a_int = v.a  # type: int  # annotation-type-mismatch\n      ')

    def test_unpacking(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        X = NamedTuple("X", [(\'a\', str), (\'b\', int)])\n      ')
            ty = self.Infer('\n        import foo\n        v = None  # type: foo.X\n        a, b = v\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Union\n        v = ...  # type: foo.namedtuple_X_0\n        a = ...  # type: str\n        b = ...  # type: int\n      ')

    def test_bad_unpacking(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        X = NamedTuple("X", [(\'a\', str), (\'b\', int)])\n      ')
            self.CheckWithErrors('\n        import foo\n        v = None  # type: foo.X\n        _, _, too_many = v  # bad-unpacking\n        too_few, = v  # bad-unpacking\n        a: float\n        b: str\n        a, b = v  # annotation-type-mismatch # annotation-type-mismatch\n      ', deep=False, pythonpath=[d.path])

    def test_is_tuple_type_and_superclasses(self):
        if False:
            while True:
                i = 10
        'Test that a NamedTuple (function syntax) behaves like a tuple.'
        self.Check('\n      from typing import MutableSequence, NamedTuple, Sequence, Tuple, Union\n      X = NamedTuple("X", [("a", int), ("b", str)])\n\n      a = X(1, "2")\n      a_tuple = a  # type: tuple\n      a_typing_tuple = a  # type: Tuple[int, str]\n      a_typing_tuple_elipses = a  # type: Tuple[Union[int, str], ...]\n      a_sequence = a  # type: Sequence[Union[int, str]]\n      a_iter = iter(a)  # type: tupleiterator[Union[int, str]]\n\n      a_first = a[0]  # type: int\n      a_second = a[1]  # type: str\n      a_first_next = next(iter(a))  # We don\'t know the type through the iter() function\n    ')

    def test_is_not_incorrect_types(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      from typing import MutableSequence, NamedTuple, Sequence, Tuple, Union\n      X = NamedTuple("X", [("a", int), ("b", str)])\n\n      x = X(1, "2")\n\n      x_wrong_tuple_types = x  # type: Tuple[str, str]  # annotation-type-mismatch\n      x_not_a_list = x  # type: list  # annotation-type-mismatch\n      x_not_a_mutable_seq = x  # type: MutableSequence[Union[int, str]]  # annotation-type-mismatch\n      x_first_wrong_element_type = x[0]  # type: str  # annotation-type-mismatch\n    ')

    def test_meets_protocol(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import NamedTuple, Protocol\n      X = NamedTuple("X", [("a", int), ("b", str)])\n\n      class IntAndStrHolderVars(Protocol):\n        a: int\n        b: str\n\n      class IntAndStrHolderProperty(Protocol):\n        @property\n        def a(self) -> int:\n          ...\n\n        @property\n        def b(self) -> str:\n          ...\n\n      a = X(1, "2")\n      a_vars_protocol: IntAndStrHolderVars = a\n      a_property_protocol: IntAndStrHolderProperty = a\n    ')

    def test_does_not_meet_mismatching_protocol(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing import NamedTuple, Protocol\n      X = NamedTuple("X", [("a", int), ("b", str)])\n\n      class DualStrHolder(Protocol):\n        a: str\n        b: str\n\n      class IntAndStrHolderVars_Alt(Protocol):\n        the_number: int\n        the_string: str\n\n      class IntStrIntHolder(Protocol):\n        a: int\n        b: str\n        c: int\n\n      a = X(1, "2")\n      a_wrong_types: DualStrHolder = a  # annotation-type-mismatch\n      a_wrong_names: IntAndStrHolderVars_Alt = a  # annotation-type-mismatch\n      a_too_many: IntStrIntHolder = a  # annotation-type-mismatch\n    ')

    def test_generated_members(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import NamedTuple\n      X = NamedTuple("X", [(\'a\', int), (\'b\', str)])')
        self.assertTypesMatchPytd(ty, '\n      from typing import NamedTuple\n      class X(NamedTuple):\n          a: int\n          b: str\n      ')
if __name__ == '__main__':
    test_base.main()