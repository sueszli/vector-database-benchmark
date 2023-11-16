"""Tests for type_match.py."""
import textwrap
from pytype.pyi import parser
from pytype.pytd import booleq
from pytype.pytd import escape
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import type_match
from pytype.pytd import visitors
from pytype.pytd.parse import parser_test_base
import unittest
_BUILTINS = '\n  class object: ...\n  class classobj: ...\n'

def pytd_src(text):
    if False:
        return 10
    text = textwrap.dedent(escape.preprocess_pytd(text))
    text = text.replace('`', '')
    return text

class TestTypeMatch(parser_test_base.ParserTest):
    """Test algorithms and datastructures of booleq.py."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        builtins = parser.parse_string(textwrap.dedent(_BUILTINS), name='builtins', options=self.options)
        typing = parser.parse_string('class Generic: ...', name='typing', options=self.options)
        self.mini_builtins = pytd_utils.Concat(builtins, typing)

    def LinkAgainstSimpleBuiltins(self, ast):
        if False:
            print('Hello World!')
        ast = ast.Visit(visitors.AdjustTypeParameters())
        ast = visitors.LookupClasses(ast, self.mini_builtins)
        return ast

    def assertMatch(self, m, t1, t2):
        if False:
            return 10
        eq = m.match_type_against_type(t1, t2, {})
        self.assertEqual(eq, booleq.TRUE)

    def assertNoMatch(self, m, t1, t2):
        if False:
            i = 10
            return i + 15
        eq = m.match_type_against_type(t1, t2, {})
        self.assertEqual(eq, booleq.FALSE)

    def test_anything(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({})
        self.assertMatch(m, pytd.AnythingType(), pytd.AnythingType())
        self.assertMatch(m, pytd.AnythingType(), pytd.NamedType('x'))
        self.assertMatch(m, pytd.NamedType('x'), pytd.AnythingType())

    def test_anything_as_top(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({}, any_also_is_bottom=False)
        self.assertMatch(m, pytd.AnythingType(), pytd.AnythingType())
        self.assertNoMatch(m, pytd.AnythingType(), pytd.NamedType('x'))
        self.assertMatch(m, pytd.NamedType('x'), pytd.AnythingType())

    def test_nothing_left(self):
        if False:
            while True:
                i = 10
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NothingType(), pytd.NamedType('A'), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_nothing_right(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NamedType('A'), pytd.NothingType(), {})
        self.assertEqual(eq, booleq.FALSE)

    def test_nothing_nothing(self):
        if False:
            print('Hello World!')
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NothingType(), pytd.NothingType(), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_nothing_anything(self):
        if False:
            for i in range(10):
                print('nop')
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NothingType(), pytd.AnythingType(), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_anything_nothing(self):
        if False:
            return 10
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.AnythingType(), pytd.NothingType(), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_anything_late(self):
        if False:
            while True:
                i = 10
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.AnythingType(), pytd.LateType('X'), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_late_anything(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.LateType('X'), pytd.AnythingType(), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_late_named(self):
        if False:
            while True:
                i = 10
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NamedType('X'), pytd.LateType('X'), {})
        self.assertEqual(eq, booleq.FALSE)

    def test_named_late(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.LateType('X'), pytd.NamedType('X'), {})
        self.assertEqual(eq, booleq.FALSE)

    def test_named(self):
        if False:
            for i in range(10):
                print('nop')
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.NamedType('A'), pytd.NamedType('A'), {})
        self.assertEqual(eq, booleq.TRUE)
        eq = m.match_type_against_type(pytd.NamedType('A'), pytd.NamedType('B'), {})
        self.assertNotEqual(eq, booleq.TRUE)

    def test_named_against_generic(self):
        if False:
            i = 10
            return i + 15
        m = type_match.TypeMatch({})
        eq = m.match_type_against_type(pytd.GenericType(pytd.NamedType('A'), ()), pytd.NamedType('A'), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_function(self):
        if False:
            while True:
                i = 10
        ast = parser.parse_string(textwrap.dedent('\n      def left(a: int) -> int: ...\n      def right(a: int) -> int: ...\n    '), options=self.options)
        m = type_match.TypeMatch()
        self.assertEqual(m.match(ast.Lookup('left'), ast.Lookup('right'), {}), booleq.TRUE)

    def test_return(self):
        if False:
            print('Hello World!')
        ast = parser.parse_string(textwrap.dedent('\n      def left(a: int) -> float: ...\n      def right(a: int) -> int: ...\n    '), options=self.options)
        m = type_match.TypeMatch()
        self.assertNotEqual(m.match(ast.Lookup('left'), ast.Lookup('right'), {}), booleq.TRUE)

    def test_optional(self):
        if False:
            return 10
        ast = parser.parse_string(textwrap.dedent('\n      def left(a: int) -> int: ...\n      def right(a: int, *args) -> int: ...\n    '), options=self.options)
        m = type_match.TypeMatch()
        self.assertEqual(m.match(ast.Lookup('left'), ast.Lookup('right'), {}), booleq.TRUE)

    def test_generic(self):
        if False:
            i = 10
            return i + 15
        ast = parser.parse_string(textwrap.dedent("\n      from typing import Any\n      T = TypeVar('T')\n      class A(typing.Generic[T], object):\n        pass\n      left = ...  # type: A[Any]\n      right = ...  # type: A[Any]\n    "), options=self.options)
        ast = self.LinkAgainstSimpleBuiltins(ast)
        m = type_match.TypeMatch()
        self.assertEqual(m.match_type_against_type(ast.Lookup('left').type, ast.Lookup('right').type, {}), booleq.TRUE)

    def test_class_match(self):
        if False:
            while True:
                i = 10
        ast = parser.parse_string(textwrap.dedent('\n      from typing import Any\n      class Left():\n        def method(self) -> Any: ...\n      class Right():\n        def method(self) -> Any: ...\n        def method2(self) -> Any: ...\n    '), options=self.options)
        ast = visitors.LookupClasses(ast, self.mini_builtins)
        m = type_match.TypeMatch()
        (left, right) = (ast.Lookup('Left'), ast.Lookup('Right'))
        self.assertEqual(m.match(left, right, {}), booleq.TRUE)
        self.assertNotEqual(m.match(right, left, {}), booleq.TRUE)

    def test_subclasses(self):
        if False:
            i = 10
            return i + 15
        ast = parser.parse_string(textwrap.dedent('\n      class A():\n        pass\n      class B(A):\n        pass\n      a = ...  # type: A\n      def left(a: B) -> B: ...\n      def right(a: A) -> A: ...\n    '), options=self.options)
        ast = visitors.LookupClasses(ast, self.mini_builtins)
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        (left, right) = (ast.Lookup('left'), ast.Lookup('right'))
        self.assertEqual(m.match(left, right, {}), booleq.TRUE)
        self.assertNotEqual(m.match(right, left, {}), booleq.TRUE)

    def _TestTypeParameters(self, reverse=False):
        if False:
            return 10
        ast = parser.parse_string(pytd_src("\n      from typing import Any, Generic\n      class `~unknown0`():\n        def next(self) -> Any: ...\n      T = TypeVar('T')\n      class A(Generic[T], object):\n        def next(self) -> Any: ...\n      class B():\n        pass\n      def left(x: `~unknown0`) -> Any: ...\n      def right(x: A[B]) -> Any: ...\n    "), options=self.options)
        ast = self.LinkAgainstSimpleBuiltins(ast)
        m = type_match.TypeMatch()
        (left, right) = (ast.Lookup('left'), ast.Lookup('right'))
        match = m.match(right, left, {}) if reverse else m.match(left, right, {})
        unknown0 = escape.unknown(0)
        self.assertEqual(match, booleq.And((booleq.Eq(unknown0, 'A'), booleq.Eq(f'{unknown0}.A.T', 'B'))))
        self.assertIn(f'{unknown0}.A.T', m.solver.variables)

    def test_unknown_against_generic(self):
        if False:
            for i in range(10):
                print('nop')
        self._TestTypeParameters()

    def test_generic_against_unknown(self):
        if False:
            while True:
                i = 10
        self._TestTypeParameters(reverse=True)

    def test_strict(self):
        if False:
            return 10
        ast = parser.parse_string(pytd_src("\n      import typing\n\n      T = TypeVar('T')\n      class list(typing.Generic[T], object):\n        pass\n      class A():\n        pass\n      class B(A):\n        pass\n      class `~unknown0`():\n        pass\n      a = ...  # type: A\n      def left() -> `~unknown0`: ...\n      def right() -> list[A]: ...\n    "), options=self.options)
        ast = self.LinkAgainstSimpleBuiltins(ast)
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        (left, right) = (ast.Lookup('left'), ast.Lookup('right'))
        unknown0 = escape.unknown(0)
        self.assertEqual(m.match(left, right, {}), booleq.And((booleq.Eq(unknown0, 'list'), booleq.Eq(f'{unknown0}.list.T', 'A'))))

    def test_base_class(self):
        if False:
            while True:
                i = 10
        ast = parser.parse_string(textwrap.dedent('\n      class Base():\n        def f(self, x:Base) -> Base: ...\n      class Foo(Base):\n        pass\n\n      class Match():\n        def f(self, x:Base) -> Base: ...\n    '), options=self.options)
        ast = self.LinkAgainstSimpleBuiltins(ast)
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        eq = m.match_Class_against_Class(ast.Lookup('Match'), ast.Lookup('Foo'), {})
        self.assertEqual(eq, booleq.TRUE)

    def test_homogeneous_tuple(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import Tuple\n      x1 = ...  # type: Tuple[bool, ...]\n      x2 = ...  # type: Tuple[int, ...]\n    ')
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        x1 = ast.Lookup('x1').type
        x2 = ast.Lookup('x2').type
        self.assertEqual(m.match_Generic_against_Generic(x1, x1, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x1, x2, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x2, x1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x2, x2, {}), booleq.TRUE)

    def test_heterogeneous_tuple(self):
        if False:
            print('Hello World!')
        ast = self.ParseWithBuiltins('\n      from typing import Tuple\n      x1 = ...  # type: Tuple[int]\n      x2 = ...  # type: Tuple[bool, str]\n      x3 = ...  # type: Tuple[int, str]\n    ')
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        x1 = ast.Lookup('x1').type
        x2 = ast.Lookup('x2').type
        x3 = ast.Lookup('x3').type
        self.assertEqual(m.match_Generic_against_Generic(x1, x1, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x1, x2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x1, x3, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x2, x1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x2, x2, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x2, x3, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x3, x1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x3, x2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(x3, x3, {}), booleq.TRUE)

    def test_tuple(self):
        if False:
            while True:
                i = 10
        ast = self.ParseWithBuiltins('\n      from typing import Tuple\n      x1 = ...  # type: Tuple[bool, ...]\n      x2 = ...  # type: Tuple[int, ...]\n      y1 = ...  # type: Tuple[bool, int]\n    ')
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        x1 = ast.Lookup('x1').type
        x2 = ast.Lookup('x2').type
        y1 = ast.Lookup('y1').type
        self.assertEqual(m.match_Generic_against_Generic(x1, y1, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(x2, y1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(y1, x1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(y1, x2, {}), booleq.TRUE)

    def test_unknown_against_tuple(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins(pytd_src('\n      from typing import Tuple\n      class `~unknown0`():\n        pass\n      x = ...  # type: Tuple[int, str]\n    '))
        unknown0 = escape.unknown(0)
        unk = ast.Lookup(unknown0)
        tup = ast.Lookup('x').type
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        match = m.match_Unknown_against_Generic(unk, tup, {})
        self.assertCountEqual(sorted(match.extract_equalities()), [(unknown0, 'builtins.tuple'), (f'{unknown0}.builtins.tuple._T', 'int'), (f'{unknown0}.builtins.tuple._T', 'str')])

    def test_function_against_tuple_subclass(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import Tuple\n      class A(Tuple[int, str]): ...\n      def f(x): ...\n    ')
        a = ast.Lookup('A')
        f = ast.Lookup('f')
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        self.assertEqual(m.match_Function_against_Class(f, a, {}, {}), booleq.FALSE)

    def test_callable_no_arguments(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import Callable\n      v1 = ...  # type: Callable[..., int]\n      v2 = ...  # type: Callable[..., bool]\n    ')
        v1 = ast.Lookup('v1').type
        v2 = ast.Lookup('v2').type
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        self.assertEqual(m.match_Generic_against_Generic(v1, v2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v2, v1, {}), booleq.TRUE)

    def test_callable_with_arguments(self):
        if False:
            while True:
                i = 10
        ast = self.ParseWithBuiltins('\n      from typing import Callable\n      v1 = ...  # type: Callable[[int], int]\n      v2 = ...  # type: Callable[[bool], int]\n      v3 = ...  # type: Callable[[int], bool]\n      v4 = ...  # type: Callable[[int, str], int]\n      v5 = ...  # type: Callable[[bool, str], int]\n      v6 = ...  # type: Callable[[], int]\n    ')
        v1 = ast.Lookup('v1').type
        v2 = ast.Lookup('v2').type
        v3 = ast.Lookup('v3').type
        v4 = ast.Lookup('v4').type
        v5 = ast.Lookup('v5').type
        v6 = ast.Lookup('v6').type
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        self.assertEqual(m.match_Generic_against_Generic(v1, v2, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v2, v1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v1, v4, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v4, v1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v4, v5, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v5, v4, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v1, v6, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v6, v1, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v6, v6, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v1, v3, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v3, v1, {}), booleq.TRUE)

    def test_callable(self):
        if False:
            for i in range(10):
                print('nop')
        ast = self.ParseWithBuiltins('\n      from typing import Callable\n      v1 = ...  # type: Callable[..., int]\n      v2 = ...  # type: Callable[..., bool]\n      v3 = ...  # type: Callable[[int, str], int]\n      v4 = ...  # type: Callable[[int, str], bool]\n      v5 = ...  # type: Callable[[], int]\n      v6 = ...  # type: Callable[[], bool]\n    ')
        v1 = ast.Lookup('v1').type
        v2 = ast.Lookup('v2').type
        v3 = ast.Lookup('v3').type
        v4 = ast.Lookup('v4').type
        v5 = ast.Lookup('v5').type
        v6 = ast.Lookup('v6').type
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        self.assertEqual(m.match_Generic_against_Generic(v1, v4, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v4, v1, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v2, v3, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v3, v2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v2, v5, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v5, v2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v1, v6, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v6, v1, {}), booleq.TRUE)

    def test_callable_and_type(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import Callable, Type\n      v1 = ...  # type: Callable[..., int]\n      v2 = ...  # type: Callable[..., bool]\n      v3 = ...  # type: Callable[[], int]\n      v4 = ...  # type: Callable[[], bool]\n      v5 = ...  # type: Type[int]\n      v6 = ...  # type: Type[bool]\n    ')
        v1 = ast.Lookup('v1').type
        v2 = ast.Lookup('v2').type
        v3 = ast.Lookup('v3').type
        v4 = ast.Lookup('v4').type
        v5 = ast.Lookup('v5').type
        v6 = ast.Lookup('v6').type
        m = type_match.TypeMatch(type_match.get_all_subclasses([ast]))
        self.assertEqual(m.match_Generic_against_Generic(v1, v6, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v6, v1, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v2, v5, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v5, v2, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v3, v6, {}), booleq.FALSE)
        self.assertEqual(m.match_Generic_against_Generic(v6, v3, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v4, v5, {}), booleq.TRUE)
        self.assertEqual(m.match_Generic_against_Generic(v5, v4, {}), booleq.FALSE)
if __name__ == '__main__':
    unittest.main()