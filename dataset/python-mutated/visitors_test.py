import textwrap
from pytype.pytd import escape
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.pytd.parse import parser_test_base
import unittest
DEFAULT_PYI = '\nfrom typing import Any\ndef __getattr__(name) -> Any: ...\n'

def pytd_src(text):
    if False:
        for i in range(10):
            print('nop')
    text = textwrap.dedent(escape.preprocess_pytd(text))
    text = text.replace('`', '')
    return text

class TestVisitors(parser_test_base.ParserTest):
    """Tests the classes in parse/visitors."""

    def test_lookup_classes(self):
        if False:
            return 10
        src = textwrap.dedent('\n        from typing import Union\n        class object:\n            pass\n\n        class A:\n            def a(self, a: A, b: B) -> Union[A, B]:\n                raise A()\n                raise B()\n\n        class B:\n            def b(self, a: A, b: B) -> Union[A, B]:\n                raise A()\n                raise B()\n    ')
        tree = self.Parse(src)
        new_tree = visitors.LookupClasses(tree)
        self.AssertSourceEquals(new_tree, src)
        new_tree.Visit(visitors.VerifyLookup())

    def test_maybe_fill_in_local_pointers(self):
        if False:
            while True:
                i = 10
        src = textwrap.dedent('\n        from typing import Union\n        class A:\n            def a(self, a: A, b: B) -> Union[A, B]:\n                raise A()\n                raise B()\n    ')
        tree = self.Parse(src)
        ty_a = pytd.ClassType('A')
        ty_a.Visit(visitors.FillInLocalPointers({'': tree}))
        self.assertIsNotNone(ty_a.cls)
        ty_b = pytd.ClassType('B')
        ty_b.Visit(visitors.FillInLocalPointers({'': tree}))
        self.assertIsNone(ty_b.cls)

    def test_deface_unresolved(self):
        if False:
            while True:
                i = 10
        builtins = self.Parse(textwrap.dedent('\n      class int:\n        pass\n    '))
        src = textwrap.dedent('\n        class A(X):\n            def a(self, a: A, b: X, c: int) -> X:\n                raise X()\n            def b(self) -> X[int]: ...\n    ')
        expected = textwrap.dedent('\n        from typing import Any\n        class A(Any):\n            def a(self, a: A, b: Any, c: int) -> Any:\n                raise Any\n            def b(self) -> Any: ...\n    ')
        tree = self.Parse(src)
        new_tree = tree.Visit(visitors.DefaceUnresolved([tree, builtins]))
        new_tree.Visit(visitors.VerifyVisitor())
        self.AssertSourceEquals(new_tree, expected)

    def test_deface_unresolved2(self):
        if False:
            while True:
                i = 10
        builtins = self.Parse(textwrap.dedent('\n      from typing import Generic, TypeVar\n      class int:\n        pass\n      T = TypeVar("T")\n      class list(Generic[T]):\n        pass\n    '))
        src = textwrap.dedent('\n        from typing import Union\n        class A(X):\n            def a(self, a: A, b: X, c: int) -> X:\n                raise X()\n            def c(self) -> Union[list[X], int]: ...\n    ')
        expected = textwrap.dedent('\n        from typing import Any, Union\n        class A(Any):\n            def a(self, a: A, b: Any, c: int) -> Any:\n                raise Any\n            def c(self) -> Union[list[Any], int]: ...\n    ')
        tree = self.Parse(src)
        new_tree = tree.Visit(visitors.DefaceUnresolved([tree, builtins]))
        new_tree.Visit(visitors.VerifyVisitor())
        self.AssertSourceEquals(new_tree, expected)

    def test_replace_types_by_name(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n        from typing import Union\n        class A:\n            def a(self, a: Union[A, B]) -> Union[A, B]:\n                raise A()\n                raise B()\n    ')
        expected = textwrap.dedent('\n        from typing import Union\n        class A:\n            def a(self: A2, a: Union[A2, B]) -> Union[A2, B]:\n                raise A2()\n                raise B()\n    ')
        tree = self.Parse(src)
        tree2 = tree.Visit(visitors.ReplaceTypesByName({'A': pytd.NamedType('A2')}))
        self.AssertSourceEquals(tree2, expected)

    def test_replace_types_by_matcher(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n        from typing import Union\n        class A:\n            def a(self, a: Union[A, B]) -> Union[A, B]:\n                raise A()\n                raise B()\n    ')
        expected = textwrap.dedent('\n        from typing import Union\n        class A:\n            def a(self: A2, a: Union[A2, B]) -> Union[A2, B]:\n                raise A2()\n                raise B()\n    ')
        tree = self.Parse(src)
        tree2 = tree.Visit(visitors.ReplaceTypesByMatcher(lambda node: node.name == 'A', pytd.NamedType('A2')))
        self.AssertSourceEquals(tree2, expected)

    def test_superclasses_by_name(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      class A():\n          pass\n      class B():\n          pass\n      class C(A):\n          pass\n      class D(A,B):\n          pass\n      class E(C,D,A):\n          pass\n    ')
        tree = self.Parse(src)
        data = tree.Visit(visitors.ExtractSuperClassesByName())
        self.assertCountEqual(('object',), data['A'])
        self.assertCountEqual(('object',), data['B'])
        self.assertCountEqual(('A',), data['C'])
        self.assertCountEqual(('A', 'B'), data['D'])
        self.assertCountEqual(('A', 'C', 'D'), data['E'])

    def test_remove_unknown_classes(self):
        if False:
            print('Hello World!')
        src = pytd_src('\n        from typing import Union\n        class `~unknown1`():\n            pass\n        class `~unknown2`():\n            pass\n        class A:\n            def foobar(x: `~unknown1`, y: `~unknown2`) -> Union[`~unknown1`, int]: ...\n    ')
        expected = textwrap.dedent('\n        from typing import Any, Union\n        class A:\n            def foobar(x, y) -> Union[Any, int]: ...\n    ')
        tree = self.Parse(src)
        tree = tree.Visit(visitors.RemoveUnknownClasses())
        self.AssertSourceEquals(tree, expected)

    def test_in_place_lookup_external_classes(self):
        if False:
            print('Hello World!')
        src1 = textwrap.dedent('\n      def f1() -> bar.Bar: ...\n      class Foo:\n        pass\n    ')
        src2 = textwrap.dedent('\n      def f2() -> foo.Foo: ...\n      class Bar:\n        pass\n    ')
        ast1 = self.Parse(src1, name='foo')
        ast2 = self.Parse(src2, name='bar')
        ast1 = ast1.Visit(visitors.LookupExternalTypes(dict(foo=ast1, bar=ast2)))
        ast2 = ast2.Visit(visitors.LookupExternalTypes(dict(foo=ast1, bar=ast2)))
        (f1,) = ast1.Lookup('foo.f1').signatures
        (f2,) = ast2.Lookup('bar.f2').signatures
        self.assertIs(ast2.Lookup('bar.Bar'), f1.return_type.cls)
        self.assertIs(ast1.Lookup('foo.Foo'), f2.return_type.cls)

    def test_lookup_constant(self):
        if False:
            i = 10
            return i + 15
        src1 = textwrap.dedent('\n      Foo = ...  # type: type\n    ')
        src2 = textwrap.dedent('\n      class Bar:\n        bar = ...  # type: foo.Foo\n    ')
        ast1 = self.Parse(src1, name='foo').Visit(visitors.LookupBuiltins(self.loader.builtins))
        ast2 = self.Parse(src2, name='bar')
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2}))
        self.assertEqual(ast2.Lookup('bar.Bar').constants[0], pytd.Constant(name='bar', type=pytd.AnythingType()))

    def test_lookup_star_alias(self):
        if False:
            for i in range(10):
                print('nop')
        src1 = textwrap.dedent('\n      x = ...  # type: int\n      T = TypeVar("T")\n      class A: ...\n      def f(x: T) -> T: ...\n      B = A\n    ')
        src2 = 'from foo import *'
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2}, self_name='bar'))
        self.assertEqual('bar', ast2.name)
        self.assertSetEqual({a.name for a in ast2.aliases}, {'bar.x', 'bar.T', 'bar.A', 'bar.f', 'bar.B'})

    def test_lookup_star_alias_in_unnamed_module(self):
        if False:
            i = 10
            return i + 15
        src1 = textwrap.dedent('\n      class A: ...\n    ')
        src2 = 'from foo import *'
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2)
        name = ast2.name
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1}, self_name=None))
        self.assertEqual(name, ast2.name)
        self.assertEqual(pytd_utils.Print(ast2), 'from foo import A')

    def test_lookup_two_star_aliases(self):
        if False:
            while True:
                i = 10
        src1 = 'class A: ...'
        src2 = 'class B: ...'
        src3 = textwrap.dedent('\n      from foo import *\n      from bar import *\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast3 = self.Parse(src3).Replace(name='baz').Visit(visitors.AddNamePrefix())
        ast3 = ast3.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2, 'baz': ast3}, self_name='baz'))
        self.assertSetEqual({a.name for a in ast3.aliases}, {'baz.A', 'baz.B'})

    def test_lookup_two_star_aliases_with_same_class(self):
        if False:
            i = 10
            return i + 15
        src1 = 'class A: ...'
        src2 = 'class A: ...'
        src3 = textwrap.dedent('\n      from foo import *\n      from bar import *\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast3 = self.Parse(src3).Replace(name='baz').Visit(visitors.AddNamePrefix())
        self.assertRaises(KeyError, ast3.Visit, visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2, 'baz': ast3}, self_name='baz'))

    def test_lookup_star_alias_with_duplicate_class(self):
        if False:
            i = 10
            return i + 15
        src1 = 'class A: ...'
        src2 = textwrap.dedent('\n      from foo import *\n      class A:\n        x = ...  # type: int\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2}, self_name='bar'))
        self.assertMultiLineEqual(pytd_utils.Print(ast2), textwrap.dedent('\n      class bar.A:\n          x: int\n    ').strip())

    def test_lookup_two_star_aliases_with_default_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        src1 = DEFAULT_PYI
        src2 = DEFAULT_PYI
        src3 = textwrap.dedent('\n      from foo import *\n      from bar import *\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast3 = self.Parse(src3).Replace(name='baz').Visit(visitors.AddNamePrefix())
        ast3 = ast3.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2, 'baz': ast3}, self_name='baz'))
        self.assertMultiLineEqual(pytd_utils.Print(ast3), textwrap.dedent('\n      from typing import Any\n\n      def baz.__getattr__(name) -> Any: ...\n    ').strip())

    def test_lookup_star_alias_with_duplicate_getattr(self):
        if False:
            print('Hello World!')
        src1 = DEFAULT_PYI
        src2 = textwrap.dedent('\n      from typing import Any\n      from foo import *\n      def __getattr__(name) -> Any: ...\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2}, self_name='bar'))
        self.assertMultiLineEqual(pytd_utils.Print(ast2), textwrap.dedent('\n      from typing import Any\n\n      def bar.__getattr__(name) -> Any: ...\n    ').strip())

    def test_lookup_two_star_aliases_with_different_getattrs(self):
        if False:
            i = 10
            return i + 15
        src1 = 'def __getattr__(name) -> int: ...'
        src2 = 'def __getattr__(name) -> str: ...'
        src3 = textwrap.dedent('\n      from foo import *\n      from bar import *\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast3 = self.Parse(src3).Replace(name='baz').Visit(visitors.AddNamePrefix())
        self.assertRaises(KeyError, ast3.Visit, visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2, 'baz': ast3}, self_name='baz'))

    def test_lookup_star_alias_with_different_getattr(self):
        if False:
            return 10
        src1 = 'def __getattr__(name) -> int: ...'
        src2 = textwrap.dedent('\n      from foo import *\n      def __getattr__(name) -> str: ...\n    ')
        ast1 = self.Parse(src1).Replace(name='foo').Visit(visitors.AddNamePrefix())
        ast2 = self.Parse(src2).Replace(name='bar').Visit(visitors.AddNamePrefix())
        ast2 = ast2.Visit(visitors.LookupExternalTypes({'foo': ast1, 'bar': ast2}, self_name='bar'))
        self.assertMultiLineEqual(pytd_utils.Print(ast2), textwrap.dedent('\n      def bar.__getattr__(name) -> str: ...\n    ').strip())

    def test_collect_dependencies(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n      from typing import Union\n      l = ... # type: list[Union[int, baz.BigInt]]\n      def f1() -> bar.Bar: ...\n      def f2() -> foo.bar.Baz: ...\n    ')
        deps = visitors.CollectDependencies()
        self.Parse(src).Visit(deps)
        self.assertCountEqual({'baz', 'bar', 'foo.bar'}, deps.dependencies)

    def test_expand(self):
        if False:
            while True:
                i = 10
        src = textwrap.dedent('\n        from typing import Union\n        def foo(a: Union[int, float], z: Union[complex, str], u: bool) -> file: ...\n        def bar(a: int) -> Union[str, unicode]: ...\n    ')
        new_src = textwrap.dedent('\n        from typing import Union\n        def foo(a: int, z: complex, u: bool) -> file: ...\n        def foo(a: int, z: str, u: bool) -> file: ...\n        def foo(a: float, z: complex, u: bool) -> file: ...\n        def foo(a: float, z: str, u: bool) -> file: ...\n        def bar(a: int) -> Union[str, unicode]: ...\n    ')
        self.AssertSourceEquals(self.ApplyVisitorToString(src, visitors.ExpandSignatures()), new_src)

    def test_print_imports(self):
        if False:
            i = 10
            return i + 15
        src = textwrap.dedent('\n      from typing import Any, List, Tuple, Union\n      def f(x: Union[int, slice]) -> List[Any]: ...\n      def g(x: foo.C.C2) -> None: ...\n    ')
        expected = textwrap.dedent('\n      import foo\n      from typing import Any, List, Union\n\n      def f(x: Union[int, slice]) -> List[Any]: ...\n      def g(x: foo.C.C2) -> None: ...\n    ').strip()
        tree = self.Parse(src)
        res = pytd_utils.Print(tree)
        self.AssertSourceEquals(res, expected)
        self.assertMultiLineEqual(res, expected)

    def test_print_imports_named_type(self):
        if False:
            while True:
                i = 10
        node = pytd.Constant('x', pytd.NamedType('typing.List'))
        tree = pytd_utils.CreateModule(name=None, constants=(node,))
        expected_src = textwrap.dedent('\n      from typing import List\n\n      x: List\n    ').strip()
        res = pytd_utils.Print(tree)
        self.assertMultiLineEqual(res, expected_src)

    def test_print_imports_ignores_existing(self):
        if False:
            while True:
                i = 10
        src = 'from foo import b'
        tree = self.Parse(src)
        res = pytd_utils.Print(tree)
        self.assertMultiLineEqual(res, src)

    @unittest.skip('depended on `or`')
    def test_print_union_name_conflict(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n      class Union: ...\n      def g(x: Union) -> Union[int, float]: ...\n    ')
        tree = self.Parse(src)
        res = pytd_utils.Print(tree)
        self.AssertSourceEquals(res, src)

    def test_adjust_type_parameters(self):
        if False:
            i = 10
            return i + 15
        ast = self.Parse('\n      from typing import Union\n      T = TypeVar("T")\n      T2 = TypeVar("T2")\n      def f(x: T) -> T: ...\n      class A(Generic[T]):\n        def a(self, x: T2) -> None:\n          self = A[Union[T, T2]]\n    ')
        f = ast.Lookup('f')
        (sig,) = f.signatures
        (p_x,) = sig.params
        self.assertEqual(sig.template, (pytd.TemplateItem(pytd.TypeParameter('T', scope='f')),))
        self.assertEqual(p_x.type, pytd.TypeParameter('T', scope='f'))
        cls = ast.Lookup('A')
        (f_cls,) = cls.methods
        (sig_cls,) = f_cls.signatures
        (p_self, p_x_cls) = sig_cls.params
        self.assertEqual(cls.template, (pytd.TemplateItem(pytd.TypeParameter('T', scope='A')),))
        self.assertEqual(sig_cls.template, (pytd.TemplateItem(pytd.TypeParameter('T2', scope='A.a')),))
        self.assertEqual(p_self.type.parameters, (pytd.TypeParameter('T', scope='A'),))
        self.assertEqual(p_x_cls.type, pytd.TypeParameter('T2', scope='A.a'))

    def test_adjust_type_parameters_with_builtins(self):
        if False:
            print('Hello World!')
        ast = self.ParseWithBuiltins('\n      T = TypeVar("T")\n      K = TypeVar("K")\n      V = TypeVar("V")\n      class Foo(List[int]): pass\n      class Bar(Dict[T, int]): pass\n      class Baz(Generic[K, V]): pass\n      class Qux(Baz[str, int]): pass\n    ')
        foo = ast.Lookup('Foo')
        bar = ast.Lookup('Bar')
        qux = ast.Lookup('Qux')
        (foo_base,) = foo.bases
        (bar_base,) = bar.bases
        (qux_base,) = qux.bases
        self.assertEqual((pytd.ClassType('int'),), foo_base.parameters)
        self.assertEqual((), foo.template)
        self.assertEqual((pytd.TypeParameter('T', scope='Bar'), pytd.ClassType('int')), bar_base.parameters)
        self.assertEqual((pytd.TemplateItem(pytd.TypeParameter('T', scope='Bar')),), bar.template)
        self.assertEqual((pytd.ClassType('str'), pytd.ClassType('int')), qux_base.parameters)
        self.assertEqual((), qux.template)

    def test_adjust_type_parameters_with_duplicates(self):
        if False:
            while True:
                i = 10
        ast = self.ParseWithBuiltins('\n      T = TypeVar("T")\n      class A(Dict[T, T], Generic[T]): pass\n    ')
        a = ast.Lookup('A')
        self.assertEqual((pytd.TemplateItem(pytd.TypeParameter('T', (), None, 'A')),), a.template)

    def test_adjust_type_parameters_with_duplicates_in_generic(self):
        if False:
            return 10
        src = textwrap.dedent('\n      T = TypeVar("T")\n      class A(Generic[T, T]): pass\n    ')
        self.assertRaises(visitors.ContainerError, lambda : self.Parse(src))

    def test_verify_containers(self):
        if False:
            return 10
        ast1 = self.ParseWithBuiltins('\n      from typing import SupportsInt, TypeVar\n      T = TypeVar("T")\n      class Foo(SupportsInt[T]): pass\n    ')
        ast2 = self.ParseWithBuiltins('\n      from typing import SupportsInt\n      class Foo(SupportsInt[int]): pass\n    ')
        ast3 = self.ParseWithBuiltins('\n      from typing import Generic\n      class Foo(Generic[int]): pass\n    ')
        ast4 = self.ParseWithBuiltins('\n      from typing import List\n      class Foo(List[int, str]): pass\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast1.Visit(visitors.VerifyContainers()))
        self.assertRaises(visitors.ContainerError, lambda : ast2.Visit(visitors.VerifyContainers()))
        self.assertRaises(visitors.ContainerError, lambda : ast3.Visit(visitors.VerifyContainers()))
        self.assertRaises(visitors.ContainerError, lambda : ast4.Visit(visitors.VerifyContainers()))

    def test_clear_class_pointers(self):
        if False:
            while True:
                i = 10
        cls = pytd.Class('foo', (), (), (), (), (), (), None, ())
        t = pytd.ClassType('foo', cls)
        t = t.Visit(visitors.ClearClassPointers())
        self.assertIsNone(t.cls)

    def test_add_name_prefix(self):
        if False:
            i = 10
            return i + 15
        src = textwrap.dedent('\n      from typing import TypeVar\n      def f(a: T) -> T: ...\n      T = TypeVar("T")\n      class X(Generic[T]):\n        pass\n    ')
        tree = self.Parse(src)
        self.assertIsNone(tree.Lookup('T').scope)
        self.assertEqual('X', tree.Lookup('X').template[0].type_param.scope)
        tree = tree.Replace(name='foo').Visit(visitors.AddNamePrefix())
        self.assertIsNotNone(tree.Lookup('foo.f'))
        self.assertIsNotNone(tree.Lookup('foo.X'))
        self.assertEqual('foo', tree.Lookup('foo.T').scope)
        self.assertEqual('foo.X', tree.Lookup('foo.X').template[0].type_param.scope)

    def test_add_name_prefix_twice(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n      from typing import Any, TypeVar\n      x = ...  # type: Any\n      T = TypeVar("T")\n      class X(Generic[T]): ...\n    ')
        tree = self.Parse(src)
        tree = tree.Replace(name='foo').Visit(visitors.AddNamePrefix())
        tree = tree.Replace(name='foo').Visit(visitors.AddNamePrefix())
        self.assertIsNotNone(tree.Lookup('foo.foo.x'))
        self.assertEqual('foo.foo', tree.Lookup('foo.foo.T').scope)
        self.assertEqual('foo.foo.X', tree.Lookup('foo.foo.X').template[0].type_param.scope)

    def test_add_name_prefix_on_class_type(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n        x = ...  # type: y\n        class Y: ...\n    ')
        tree = self.Parse(src)
        x = tree.Lookup('x')
        x = x.Replace(type=pytd.ClassType('Y'))
        tree = tree.Replace(constants=(x,), name='foo')
        tree = tree.Visit(visitors.AddNamePrefix())
        self.assertEqual('foo.Y', tree.Lookup('foo.x').type.name)

    def test_add_name_prefix_on_nested_class_alias(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      class A:\n        class B:\n          class C: ...\n          D = A.B.C\n    ')
        expected = textwrap.dedent('\n      from typing import Type\n\n      class foo.A:\n          class foo.A.B:\n              class foo.A.B.C: ...\n              D: Type[foo.A.B.C]\n    ').strip()
        self.assertMultiLineEqual(expected, pytd_utils.Print(self.Parse(src).Replace(name='foo').Visit(visitors.AddNamePrefix())))

    def test_add_name_prefix_on_nested_class_outside_ref(self):
        if False:
            return 10
        src = textwrap.dedent('\n      class A:\n        class B: ...\n      b: A.B\n      C = A.B\n      def f(x: A.B) -> A.B: ...\n      class D:\n        b: A.B\n        def f(self, x: A.B) -> A.B: ...\n    ')
        expected = textwrap.dedent('\n      from typing import Type\n\n      foo.b: foo.A.B\n      foo.C: Type[foo.A.B]\n\n      class foo.A:\n          class foo.A.B: ...\n\n      class foo.D:\n          b: foo.A.B\n          def f(self, x: foo.A.B) -> foo.A.B: ...\n\n      def foo.f(x: foo.A.B) -> foo.A.B: ...\n    ').strip()
        self.assertMultiLineEqual(expected, pytd_utils.Print(self.Parse(src).Replace(name='foo').Visit(visitors.AddNamePrefix())))

    def test_add_name_prefix_on_nested_class_method(self):
        if False:
            return 10
        src = textwrap.dedent('\n      class A:\n        class B:\n          def copy(self) -> A.B: ...\n    ')
        expected = textwrap.dedent('\n      class foo.A:\n          class foo.A.B:\n              def copy(self) -> foo.A.B: ...\n    ').strip()
        self.assertMultiLineEqual(expected, pytd_utils.Print(self.Parse(src).Replace(name='foo').Visit(visitors.AddNamePrefix())))

    def test_print_merge_types(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      from typing import Union\n      def a(a: float) -> int: ...\n      def b(a: Union[int, float]) -> int: ...\n      def c(a: object) -> Union[float, int]: ...\n      def d(a: float) -> int: ...\n      def e(a: Union[bool, None]) -> Union[bool, None]: ...\n    ')
        expected = textwrap.dedent('\n      from typing import Optional, Union\n\n      def a(a: float) -> int: ...\n      def b(a: float) -> int: ...\n      def c(a: object) -> Union[float, int]: ...\n      def d(a: float) -> int: ...\n      def e(a: bool) -> Optional[bool]: ...\n    ')
        self.assertMultiLineEqual(expected.strip(), pytd_utils.Print(self.ToAST(src)).strip())

    def test_print_heterogeneous_tuple(self):
        if False:
            while True:
                i = 10
        t = pytd.TupleType(pytd.NamedType('tuple'), (pytd.NamedType('str'), pytd.NamedType('float')))
        self.assertEqual('Tuple[str, float]', pytd_utils.Print(t))

    def test_verify_heterogeneous_tuple(self):
        if False:
            i = 10
            return i + 15
        base = pytd.ClassType('tuple')
        base.cls = pytd.Class('tuple', (), (), (), (), (), (), None, ())
        t1 = pytd.TupleType(base, (pytd.NamedType('str'), pytd.NamedType('float')))
        self.assertRaises(visitors.ContainerError, lambda : t1.Visit(visitors.VerifyContainers()))
        gen = pytd.ClassType('typing.Generic')
        gen.cls = pytd.Class('typing.Generic', (), (), (), (), (), (), None, ())
        t2 = pytd.TupleType(gen, (pytd.NamedType('str'), pytd.NamedType('float')))
        self.assertRaises(visitors.ContainerError, lambda : t2.Visit(visitors.VerifyContainers()))
        param = pytd.TypeParameter('T')
        generic_base = pytd.GenericType(gen, (param,))
        base.cls = pytd.Class('tuple', (), (generic_base,), (), (), (), (), None, (pytd.TemplateItem(param),))
        t3 = pytd.TupleType(base, (pytd.NamedType('str'), pytd.NamedType('float')))
        t3.Visit(visitors.VerifyContainers())

    def test_typevar_value_conflict(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import List\n      class A(List[int], List[str]): ...\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast.Visit(visitors.VerifyContainers()))

    def test_typevar_value_conflict_hidden(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import List\n      class A(List[int]): ...\n      class B(A, List[str]): ...\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast.Visit(visitors.VerifyContainers()))

    def test_typevar_value_conflict_related_containers(self):
        if False:
            return 10
        ast = self.ParseWithBuiltins('\n      from typing import List, Sequence\n      class A(List[int], Sequence[str]): ...\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast.Visit(visitors.VerifyContainers()))

    def test_typevar_value_no_conflict(self):
        if False:
            print('Hello World!')
        ast = self.ParseWithBuiltins('\n      from typing import ContextManager, SupportsAbs\n      class Foo(SupportsAbs[float], ContextManager[Foo]): ...\n    ')
        ast.Visit(visitors.VerifyContainers())

    def test_typevar_value_consistency(self):
        if False:
            print('Hello World!')
        ast = self.ParseWithBuiltins('\n      from typing import Generic, TypeVar\n      T1 = TypeVar("T1")\n      T2 = TypeVar("T2")\n      T3 = TypeVar("T3")\n      T4 = TypeVar("T4")\n      T5 = TypeVar("T5")\n      class A(Generic[T1]): ...\n      class B1(A[T2]): ...\n      class B2(A[T3]): ...\n      class C(B1[T4], B2[T5]): ...\n      class D(C[str, str], A[str]): ...\n    ')
        ast.Visit(visitors.VerifyContainers())

    def test_typevar_value_and_alias_conflict(self):
        if False:
            while True:
                i = 10
        ast = self.ParseWithBuiltins('\n      from typing import Generic, TypeVar\n      T = TypeVar("T")\n      class A(Generic[T]): ...\n      class B(A[int], A[T]): ...\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast.Visit(visitors.VerifyContainers()))

    def test_typevar_alias_and_value_conflict(self):
        if False:
            print('Hello World!')
        ast = self.ParseWithBuiltins('\n      from typing import Generic, TypeVar\n      T = TypeVar("T")\n      class A(Generic[T]): ...\n      class B(A[T], A[int]): ...\n    ')
        self.assertRaises(visitors.ContainerError, lambda : ast.Visit(visitors.VerifyContainers()))

    def test_verify_container_with_mro_error(self):
        if False:
            i = 10
            return i + 15
        ast = self.ParseWithBuiltins('\n      from typing import List\n      class A(List[str]): ...\n      class B(List[str], A): ...\n    ')
        ast.Visit(visitors.VerifyContainers())

    def test_alias_printing(self):
        if False:
            while True:
                i = 10
        a = pytd.Alias('MyList', pytd.GenericType(pytd.NamedType('typing.List'), (pytd.AnythingType(),)))
        ty = pytd_utils.CreateModule('test', aliases=(a,))
        expected = textwrap.dedent('\n      from typing import Any, List\n\n      MyList = List[Any]')
        self.assertMultiLineEqual(expected.strip(), pytd_utils.Print(ty).strip())

    def test_print_none_union(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n      from typing import Union\n      def f(x: Union[str, None]) -> None: ...\n      def g(x: Union[str, int, None]) -> None: ...\n      def h(x: Union[None]) -> None: ...\n    ')
        expected = textwrap.dedent('\n      from typing import Optional, Union\n\n      def f(x: Optional[str]) -> None: ...\n      def g(x: Optional[Union[str, int]]) -> None: ...\n      def h(x: None) -> None: ...\n    ')
        self.assertMultiLineEqual(expected.strip(), pytd_utils.Print(self.ToAST(src)).strip())

    def test_lookup_typing_class(self):
        if False:
            print('Hello World!')
        node = visitors.LookupClasses(pytd.NamedType('typing.Sequence'), self.loader.concat_all())
        assert node.cls

    def test_create_type_parameters_from_unknowns(self):
        if False:
            return 10
        src = pytd_src('\n      from typing import Dict\n      def f(x: `~unknown1`) -> `~unknown1`: ...\n      def g(x: `~unknown2`, y: `~unknown2`) -> None: ...\n      def h(x: `~unknown3`) -> None: ...\n      def i(x: Dict[`~unknown4`, `~unknown4`]) -> None: ...\n\n      # Should not be changed\n      class `~unknown5`:\n        def __add__(self, x: `~unknown6`) -> `~unknown6`: ...\n      def `~f`(x: `~unknown7`) -> `~unknown7`: ...\n    ')
        expected = pytd_src("\n      from typing import Dict\n\n      _T0 = TypeVar('_T0')\n\n      def f(x: _T0) -> _T0: ...\n      def g(x: _T0, y: _T0) -> None: ...\n      def h(x: `~unknown3`) -> None: ...\n      def i(x: Dict[_T0, _T0]) -> None: ...\n\n      class `~unknown5`:\n        def __add__(self, x: `~unknown6`) -> `~unknown6`: ...\n      def `~f`(x: `~unknown7`) -> `~unknown7`: ...\n    ")
        ast1 = self.Parse(src)
        ast1 = ast1.Visit(visitors.CreateTypeParametersForSignatures())
        self.AssertSourceEquals(ast1, expected)

    @unittest.skip('We no longer support redefining TypeVar')
    def test_redefine_typevar(self):
        if False:
            return 10
        src = pytd_src('\n      def f(x: `~unknown1`) -> `~unknown1`: ...\n      class `TypeVar`: ...\n    ')
        ast = self.Parse(src).Visit(visitors.CreateTypeParametersForSignatures())
        self.assertMultiLineEqual(pytd_utils.Print(ast), textwrap.dedent("\n      import typing\n\n      _T0 = TypeVar('_T0')\n\n      class `TypeVar`: ...\n\n      def f(x: _T0) -> _T0: ...").strip())

    def test_create_type_parameters_for_new(self):
        if False:
            print('Hello World!')
        src = textwrap.dedent('\n      class Foo:\n          def __new__(cls: Type[Foo]) -> Foo: ...\n      class Bar:\n          def __new__(cls: Type[Bar], x, y, z) -> Bar: ...\n    ')
        ast = self.Parse(src).Visit(visitors.CreateTypeParametersForSignatures())
        self.assertMultiLineEqual(pytd_utils.Print(ast), textwrap.dedent("\n      from typing import TypeVar\n\n      _TBar = TypeVar('_TBar', bound=Bar)\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n\n      class Foo:\n          def __new__(cls: Type[_TFoo]) -> _TFoo: ...\n\n      class Bar:\n          def __new__(cls: Type[_TBar], x, y, z) -> _TBar: ...\n    ").strip())

    def test_keep_custom_new(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      class Foo:\n          def __new__(cls: Type[X]) -> X: ...\n\n      class Bar:\n          def __new__(cls, x: Type[Bar]) -> Bar: ...\n    ').strip()
        ast = self.Parse(src).Visit(visitors.CreateTypeParametersForSignatures())
        self.assertMultiLineEqual(pytd_utils.Print(ast), src)

    def test_print_type_parameter_bound(self):
        if False:
            i = 10
            return i + 15
        src = textwrap.dedent('\n      from typing import TypeVar\n      T = TypeVar("T", bound=str)\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(self.Parse(src)), textwrap.dedent("\n      from typing import TypeVar\n\n      T = TypeVar('T', bound=str)").lstrip())

    def test_print_cls(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      class A:\n          def __new__(cls: Type[A]) -> A: ...\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(self.Parse(src)), textwrap.dedent('\n      class A:\n          def __new__(cls) -> A: ...\n    ').strip())

    def test_print_never(self):
        if False:
            i = 10
            return i + 15
        src = textwrap.dedent('\n      def f() -> nothing: ...\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(self.Parse(src)), textwrap.dedent('\n      from typing import Never\n\n      def f() -> Never: ...').lstrip())

    def test_print_multiline_signature(self):
        if False:
            while True:
                i = 10
        src = textwrap.dedent('\n      def f(x: int, y: str, z: bool) -> list[str]:\n        pass\n    ')
        self.assertMultiLineEqual(pytd_utils.Print(self.Parse(src), multiline_args=True), textwrap.dedent('\n           from typing import List\n\n           def f(\n               x: int,\n               y: str,\n               z: bool\n           ) -> List[str]: ...\n        ').strip())

class RemoveNamePrefixTest(parser_test_base.ParserTest):
    """Tests for RemoveNamePrefix."""

    def test_remove_name_prefix(self):
        if False:
            return 10
        src = textwrap.dedent('\n      from typing import TypeVar\n      def f(a: T) -> T: ...\n      T = TypeVar("T")\n      class X(Generic[T]):\n        pass\n    ')
        expected = textwrap.dedent("\n      from typing import TypeVar\n\n      T = TypeVar('T')\n\n      class X(Generic[T]): ...\n\n      def f(a: T) -> T: ...\n    ").strip()
        tree = self.Parse(src)
        t = tree.Lookup('T').Replace(scope='foo')
        x = tree.Lookup('X')
        x_template = x.template[0]
        x_type_param = x_template.type_param.Replace(scope='foo.X')
        x_template = x_template.Replace(type_param=x_type_param)
        x = x.Replace(name='foo.X', template=(x_template,))
        f = tree.Lookup('f')
        f_sig = f.signatures[0]
        f_param = f_sig.params[0]
        f_type_param = f_param.type.Replace(scope='foo.f')
        f_param = f_param.Replace(type=f_type_param)
        f_template = f_sig.template[0].Replace(type_param=f_type_param)
        f_sig = f_sig.Replace(params=(f_param,), return_type=f_type_param, template=(f_template,))
        f = f.Replace(name='foo.f', signatures=(f_sig,))
        tree = tree.Replace(classes=(x,), functions=(f,), type_params=(t,), name='foo')
        tree = tree.Visit(visitors.RemoveNamePrefix())
        self.assertMultiLineEqual(expected, pytd_utils.Print(tree))

    def test_remove_name_prefix_twice(self):
        if False:
            while True:
                i = 10
        src = textwrap.dedent('\n      from typing import Any, TypeVar\n      x = ...  # type: Any\n      T = TypeVar("T")\n      class X(Generic[T]): ...\n    ')
        expected_one = textwrap.dedent("\n      from typing import Any, TypeVar\n\n      foo.x: Any\n\n      T = TypeVar('T')\n\n      class foo.X(Generic[T]): ...\n    ").strip()
        expected_two = textwrap.dedent("\n      from typing import Any, TypeVar\n\n      x: Any\n\n      T = TypeVar('T')\n\n      class X(Generic[T]): ...\n    ").strip()
        tree = self.Parse(src)
        x = tree.Lookup('x').Replace(name='foo.foo.x')
        t = tree.Lookup('T').Replace(scope='foo.foo')
        x_cls = tree.Lookup('X')
        x_template = x_cls.template[0]
        x_type_param = x_template.type_param.Replace(scope='foo.foo.X')
        x_template = x_template.Replace(type_param=x_type_param)
        x_cls = x_cls.Replace(name='foo.foo.X', template=(x_template,))
        tree = tree.Replace(classes=(x_cls,), constants=(x,), type_params=(t,), name='foo')
        tree = tree.Visit(visitors.RemoveNamePrefix())
        self.assertMultiLineEqual(expected_one, pytd_utils.Print(tree))
        tree = tree.Visit(visitors.RemoveNamePrefix())
        self.assertMultiLineEqual(expected_two, pytd_utils.Print(tree))

    def test_remove_name_prefix_on_class_type(self):
        if False:
            i = 10
            return i + 15
        src = textwrap.dedent('\n        x = ...  # type: y\n        class Y: ...\n    ')
        expected = textwrap.dedent('\n        x: Y\n\n        class Y: ...\n    ').strip()
        tree = self.Parse(src)
        x = tree.Lookup('x').Replace(name='foo.x', type=pytd.ClassType('foo.Y'))
        y = tree.Lookup('Y').Replace(name='foo.Y')
        tree = tree.Replace(classes=(y,), constants=(x,), name='foo')
        tree = tree.Visit(visitors.RemoveNamePrefix())
        self.assertMultiLineEqual(expected, pytd_utils.Print(tree))

    def test_remove_name_prefix_on_nested_class(self):
        if False:
            return 10
        src = textwrap.dedent('\n      class A:\n        class B:\n          class C: ...\n          D = A.B.C\n    ')
        expected = textwrap.dedent('\n      from typing import Type\n\n      class A:\n          class B:\n              class C: ...\n              D: Type[A.B.C]\n    ').strip()
        tree = self.Parse(src)
        a = tree.Lookup('A')
        b = a.Lookup('B')
        c = b.Lookup('C').Replace(name='foo.A.B.C')
        d = b.Lookup('D')
        d_type = d.type
        d_generic = d.type.parameters[0].Replace(name='foo.A.B.C')
        d_type = d_type.Replace(parameters=(d_generic,))
        d = d.Replace(type=d_type)
        b = b.Replace(classes=(c,), constants=(d,), name='foo.A.B')
        a = a.Replace(classes=(b,), name='foo.A')
        tree = tree.Replace(classes=(a,), name='foo')
        tree = tree.Visit(visitors.RemoveNamePrefix())
        self.assertMultiLineEqual(expected, pytd_utils.Print(tree))

class ReplaceModulesWithAnyTest(unittest.TestCase):

    def test_any_replacement(self):
        if False:
            i = 10
            return i + 15
        class_type_match = pytd.ClassType('match.foo')
        named_type_match = pytd.NamedType('match.bar')
        class_type_no_match = pytd.ClassType('match_no.foo')
        named_type_no_match = pytd.NamedType('match_no.bar')
        generic_type_match = pytd.GenericType(class_type_match, ())
        generic_type_no_match = pytd.GenericType(class_type_no_match, ())
        visitor = visitors.ReplaceModulesWithAny(['match.'])
        self.assertEqual(class_type_no_match, class_type_no_match.Visit(visitor))
        self.assertEqual(named_type_no_match, named_type_no_match.Visit(visitor))
        self.assertEqual(generic_type_no_match, generic_type_no_match.Visit(visitor))
        self.assertEqual(pytd.AnythingType, class_type_match.Visit(visitor).__class__)
        self.assertEqual(pytd.AnythingType, named_type_match.Visit(visitor).__class__)
        self.assertEqual(pytd.AnythingType, generic_type_match.Visit(visitor).__class__)

class ReplaceUnionsWithAnyTest(unittest.TestCase):

    def test_any_replacement(self):
        if False:
            print('Hello World!')
        union = pytd.UnionType((pytd.NamedType('a'), pytd.NamedType('b')))
        self.assertEqual(union.Visit(visitors.ReplaceUnionsWithAny()), pytd.AnythingType())
if __name__ == '__main__':
    unittest.main()