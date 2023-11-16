import textwrap
from pytype.pyi import parser
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.pytd.parse import parser_test_base
import unittest

class TestUtils(parser_test_base.ParserTest):
    """Test pytype.pytd.pytd_utils."""

    def test_unpack_union(self):
        if False:
            i = 10
            return i + 15
        'Test for UnpackUnion.'
        ast = self.Parse('\n      from typing import Union\n      c1 = ...  # type: Union[int, float]\n      c2 = ...  # type: int\n      c3 = ...  # type: list[Union[int, float]]')
        c1 = ast.Lookup('c1').type
        c2 = ast.Lookup('c2').type
        c3 = ast.Lookup('c3').type
        self.assertCountEqual(pytd_utils.UnpackUnion(c1), c1.type_list)
        self.assertCountEqual(pytd_utils.UnpackUnion(c2), [c2])
        self.assertCountEqual(pytd_utils.UnpackUnion(c3), [c3])

    def test_concat(self):
        if False:
            while True:
                i = 10
        'Test for concatenating two pytd ASTs.'
        ast1 = self.Parse('\n      c1 = ...  # type: int\n\n      def f1() -> int: ...\n\n      class Class1:\n        pass\n    ')
        ast2 = self.Parse('\n      c2 = ...  # type: int\n\n      def f2() -> int: ...\n\n      class Class2:\n        pass\n    ')
        expected = textwrap.dedent('\n      c1 = ...  # type: int\n      c2 = ...  # type: int\n\n      def f1() -> int: ...\n      def f2() -> int: ...\n\n      class Class1:\n          pass\n\n      class Class2:\n          pass\n    ')
        combined = pytd_utils.Concat(ast1, ast2)
        self.AssertSourceEquals(combined, expected)

    def test_concat3(self):
        if False:
            i = 10
            return i + 15
        'Test for concatenating three pytd ASTs.'
        ast1 = self.Parse('c1 = ...  # type: int')
        ast2 = self.Parse('c2 = ...  # type: float')
        ast3 = self.Parse('c3 = ...  # type: bool')
        combined = pytd_utils.Concat(ast1, ast2, ast3)
        expected = textwrap.dedent('\n      c1 = ...  # type: int\n      c2 = ...  # type: float\n      c3 = ...  # type: bool\n    ')
        self.AssertSourceEquals(combined, expected)

    def test_concat_type_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for concatenating ASTs with type parameters.'
        ast1 = self.Parse('T = TypeVar("T")', name='builtins')
        ast2 = self.Parse('T = TypeVar("T")')
        combined = pytd_utils.Concat(ast1, ast2)
        self.assertEqual(combined.Lookup('builtins.T'), pytd.TypeParameter('T', scope='builtins'))
        self.assertEqual(combined.Lookup('T'), pytd.TypeParameter('T', scope=None))

    def test_join_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that JoinTypes() does recursive flattening.'
        (n1, n2, n3, n4, n5, n6) = (pytd.NamedType('n%d' % i) for i in range(6))
        nested1 = pytd.UnionType((n1, pytd.UnionType((n2, pytd.UnionType((n3,))))))
        nested2 = pytd.UnionType((pytd.UnionType((pytd.UnionType((n4,)), n5)), n6))
        joined = pytd_utils.JoinTypes([nested1, nested2])
        self.assertEqual(joined.type_list, (n1, n2, n3, n4, n5, n6))

    def test_join_single_type(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that JoinTypes() returns single types as-is.'
        a = pytd.NamedType('a')
        self.assertEqual(pytd_utils.JoinTypes([a]), a)
        self.assertEqual(pytd_utils.JoinTypes([a, a]), a)

    def test_join_nothing_type(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that JoinTypes() removes or collapses 'nothing'."
        a = pytd.NamedType('a')
        nothing = pytd.NothingType()
        self.assertEqual(pytd_utils.JoinTypes([a, nothing]), a)
        self.assertEqual(pytd_utils.JoinTypes([nothing]), nothing)
        self.assertEqual(pytd_utils.JoinTypes([nothing, nothing]), nothing)

    def test_join_empty_types_to_nothing(self):
        if False:
            i = 10
            return i + 15
        "Test that JoinTypes() simplifies empty unions to 'nothing'."
        self.assertIsInstance(pytd_utils.JoinTypes([]), pytd.NothingType)

    def test_join_anything_types(self):
        if False:
            while True:
                i = 10
        "Test that JoinTypes() simplifies unions containing 'Any'."
        types = [pytd.AnythingType(), pytd.NamedType('a')]
        self.assertIsInstance(pytd_utils.JoinTypes(types), pytd.AnythingType)

    def test_join_optional_anything_types(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that JoinTypes() simplifies unions containing 'Any' and 'None'."
        any_type = pytd.AnythingType()
        none_type = pytd.NamedType('builtins.NoneType')
        types = [pytd.NamedType('a'), any_type, none_type]
        self.assertEqual(pytd_utils.JoinTypes(types), pytd.UnionType((any_type, none_type)))

    def test_type_matcher(self):
        if False:
            print('Hello World!')
        'Test for the TypeMatcher class.'

        class MyTypeMatcher(pytd_utils.TypeMatcher):

            def default_match(self, t1, t2, mykeyword):
                if False:
                    i = 10
                    return i + 15
                assert mykeyword == 'foobar'
                return t1 == t2

            def match_Function_against_Function(self, f1, f2, mykeyword):
                if False:
                    while True:
                        i = 10
                assert mykeyword == 'foobar'
                return all((self.match(sig1, sig2, mykeyword) for (sig1, sig2) in zip(f1.signatures, f2.signatures)))
        s1 = pytd.Signature((), None, None, pytd.NothingType(), (), ())
        s2 = pytd.Signature((), None, None, pytd.AnythingType(), (), ())
        self.assertTrue(MyTypeMatcher().match(pytd.Function('f1', (s1, s2), pytd.MethodKind.METHOD), pytd.Function('f2', (s1, s2), pytd.MethodKind.METHOD), mykeyword='foobar'))
        self.assertFalse(MyTypeMatcher().match(pytd.Function('f1', (s1, s2), pytd.MethodKind.METHOD), pytd.Function('f2', (s2, s2), pytd.MethodKind.METHOD), mykeyword='foobar'))

    def test_named_type_with_module(self):
        if False:
            return 10
        'Test NamedTypeWithModule().'
        self.assertEqual(pytd_utils.NamedTypeWithModule('name'), pytd.NamedType('name'))
        self.assertEqual(pytd_utils.NamedTypeWithModule('name', None), pytd.NamedType('name'))
        self.assertEqual(pytd_utils.NamedTypeWithModule('name', 'package'), pytd.NamedType('package.name'))

    def test_ordered_set(self):
        if False:
            return 10
        ordered_set = pytd_utils.OrderedSet((n // 2 for n in range(10)))
        ordered_set.add(-42)
        ordered_set.add(3)
        self.assertEqual(tuple(ordered_set), (0, 1, 2, 3, 4, -42))

    def test_wrap_type_decl_unit(self):
        if False:
            while True:
                i = 10
        'Test WrapTypeDeclUnit.'
        ast1 = self.Parse('\n      c = ...  # type: int\n      def f(x: int) -> int: ...\n      def f(x: float) -> float: ...\n      class A:\n        pass\n    ')
        ast2 = self.Parse('\n      c = ...  # type: float\n      d = ...  # type: int\n      def f(x: complex) -> complex: ...\n      class B:\n        pass\n    ')
        w = pytd_utils.WrapTypeDeclUnit('combined', ast1.classes + ast1.functions + ast1.constants + ast2.classes + ast2.functions + ast2.constants)
        expected = textwrap.dedent('\n      from typing import Union\n      c = ...  # type: Union[int, float]\n      d = ...  # type: int\n      def f(x: int) -> int: ...\n      def f(x: float) -> float: ...\n      def f(x: complex) -> complex: ...\n      class A:\n        pass\n      class B:\n        pass\n    ')
        self.AssertSourceEquals(w, expected)

    def test_builtin_alias(self):
        if False:
            print('Hello World!')
        src = 'Number = int'
        ast = parser.parse_string(src, options=self.options)
        self.assertMultiLineEqual(pytd_utils.Print(ast), src)

    def test_typing_name_conflict1(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      import typing\n\n      x: typing.List[str]\n\n      def List() -> None: ...\n    ')
        ast = parser.parse_string(src, options=self.options)
        self.assertMultiLineEqual(pytd_utils.Print(ast).strip('\n'), src.strip('\n'))

    def test_typing_name_conflict2(self):
        if False:
            print('Hello World!')
        ast = parser.parse_string(textwrap.dedent('\n      import typing\n      from typing import Any\n\n      x = ...  # type: typing.List[str]\n\n      class MyClass:\n          List = ...  # type: Any\n          x = ...  # type: typing.List[str]\n    '), options=self.options)
        expected = textwrap.dedent('\n      import typing\n      from typing import Any, List\n\n      x: List[str]\n\n      class MyClass:\n          List: Any\n          x: typing.List[str]\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(ast).strip('\n'), expected.strip('\n'))

    def test_dummy_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('def foo() -> Any: ...', pytd_utils.Print(pytd_utils.DummyMethod('foo')))
        self.assertEqual('def foo(x) -> Any: ...', pytd_utils.Print(pytd_utils.DummyMethod('foo', 'x')))
        self.assertEqual('def foo(x, y) -> Any: ...', pytd_utils.Print(pytd_utils.DummyMethod('foo', 'x', 'y')))

    def test_asteq(self):
        if False:
            print('Hello World!')
        src1 = textwrap.dedent("\n        from typing import Union\n        def foo(a: Union[int, str]) -> C: ...\n        T = TypeVar('T')\n        class C(typing.Generic[T], object):\n            def bar(x: T) -> NoneType: ...\n        CONSTANT = ...  # type: C[float]\n        ")
        src2 = textwrap.dedent("\n        from typing import Union\n        CONSTANT = ...  # type: C[float]\n        T = TypeVar('T')\n        class C(typing.Generic[T], object):\n            def bar(x: T) -> NoneType: ...\n        def foo(a: Union[str, int]) -> C: ...\n        ")
        tree1 = parser.parse_string(src1, options=self.options)
        tree2 = parser.parse_string(src2, options=self.options)
        tree1.Visit(visitors.VerifyVisitor())
        tree2.Visit(visitors.VerifyVisitor())
        self.assertTrue(tree1.constants)
        self.assertTrue(tree1.classes)
        self.assertTrue(tree1.functions)
        self.assertTrue(tree2.constants)
        self.assertTrue(tree2.classes)
        self.assertTrue(tree2.functions)
        self.assertIsInstance(tree1, pytd.TypeDeclUnit)
        self.assertIsInstance(tree2, pytd.TypeDeclUnit)
        self.assertTrue(tree1 == tree1)
        self.assertTrue(tree2 == tree2)
        self.assertFalse(tree1 == tree2)
        self.assertFalse(tree2 == tree1)
        self.assertFalse(tree1 != tree1)
        self.assertFalse(tree2 != tree2)
        self.assertTrue(tree1 != tree2)
        self.assertTrue(tree2 != tree1)
        self.assertEqual(tree1, tree1)
        self.assertEqual(tree2, tree2)
        self.assertNotEqual(tree1, tree2)
        self.assertTrue(pytd_utils.ASTeq(tree1, tree2))
        self.assertTrue(pytd_utils.ASTeq(tree1, tree1))
        self.assertTrue(pytd_utils.ASTeq(tree2, tree1))
        self.assertTrue(pytd_utils.ASTeq(tree2, tree2))

    def test_type_builder(self):
        if False:
            for i in range(10):
                print('nop')
        t = pytd_utils.TypeBuilder()
        self.assertFalse(t)
        t.add_type(pytd.AnythingType())
        self.assertTrue(t)

class PrintTest(parser_test_base.ParserTest):
    """Test pytd_utils.Print."""

    def test_smoke(self):
        if False:
            i = 10
            return i + 15
        'Smoketest for printing pytd.'
        ast = self.Parse("\n      from typing import Any, Union\n      c1 = ...  # type: int\n      T = TypeVar('T')\n      class A(typing.Generic[T], object):\n        bar = ...  # type: T\n        def foo(self, x: list[int], y: T) -> Union[list[T], float]:\n          raise ValueError()\n      X = TypeVar('X')\n      Y = TypeVar('Y')\n      def bar(x: Union[X, Y]) -> Any: ...\n    ")
        pytd_utils.Print(ast)

    def test_literal(self):
        if False:
            i = 10
            return i + 15
        ast = self.Parse('\n      from typing import Literal\n      x1: Literal[""]\n      x2: Literal[b""]\n      x3: Literal[0]\n      x4: Literal[True]\n      x5: Literal[None]\n    ')
        ast = ast.Visit(visitors.LookupBuiltins(self.loader.builtins))
        self.assertMultiLineEqual(pytd_utils.Print(ast), textwrap.dedent("\n      from typing import Literal\n\n      x1: Literal['']\n      x2: Literal[b'']\n      x3: Literal[0]\n      x4: Literal[True]\n      x5: None\n    ").strip())

    def test_literal_union(self):
        if False:
            print('Hello World!')
        ast = self.Parse('\n      from typing import Literal, Union\n      x: Union[Literal["x"], Literal["y"]]\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(ast), textwrap.dedent("\n      from typing import Literal\n\n      x: Literal['x', 'y']\n    ").strip())

    def test_reuse_union_name(self):
        if False:
            for i in range(10):
                print('nop')
        src = '\n      import typing\n      from typing import Callable, Iterable, Tuple\n\n      class Node: ...\n\n      class Union:\n          _predicates: Tuple[Callable[[typing.Union[Iterable[Node], Node]], bool], ...]\n          def __init__(self, *predicates: Callable[[typing.Union[Iterable[Node], Node]], bool]) -> None: ...\n    '
        ast = self.Parse(src)
        self.assertMultiLineEqual(pytd_utils.Print(ast), textwrap.dedent(src).strip())
if __name__ == '__main__':
    unittest.main()