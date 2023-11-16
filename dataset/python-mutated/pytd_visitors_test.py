"""Tests for pytd_visitors."""
import textwrap
from pytype.pytd import pytd_visitors
from pytype.pytd import visitors
from pytype.pytd.parse import parser_test_base
import unittest

class PytdVisitorsTest(parser_test_base.ParserTest):

    def test_rename_module(self):
        if False:
            print('Hello World!')
        module_name = 'foo.bar'
        src = '\n        import module2\n        from module2 import f\n        from typing import List\n\n        constant = True\n\n        x = List[int]\n        b = List[int]\n\n        class SomeClass:\n          def __init__(self, a: module2.ObjectMod2):\n            pass\n\n        def ModuleFunction():\n          pass\n    '
        ast = self.Parse(src, name=module_name)
        new_ast = ast.Visit(pytd_visitors.RenameModuleVisitor(module_name, 'other.name'))
        self.assertEqual('other.name', new_ast.name)
        self.assertTrue(new_ast.Lookup('other.name.SomeClass'))
        self.assertTrue(new_ast.Lookup('other.name.constant'))
        self.assertTrue(new_ast.Lookup('other.name.ModuleFunction'))
        with self.assertRaises(KeyError):
            new_ast.Lookup('foo.bar.SomeClass')

    def test_rename_module_with_type_parameter(self):
        if False:
            while True:
                i = 10
        module_name = 'foo.bar'
        src = "\n      import typing\n\n      T = TypeVar('T')\n\n      class SomeClass(typing.Generic[T]):\n        def __init__(self, foo: T) -> None:\n          pass\n    "
        ast = self.Parse(src, name=module_name)
        new_ast = ast.Visit(pytd_visitors.RenameModuleVisitor(module_name, 'other.name'))
        some_class = new_ast.Lookup('other.name.SomeClass')
        self.assertTrue(some_class)
        init_function = some_class.Lookup('__init__')
        self.assertTrue(init_function)
        self.assertEqual(len(init_function.signatures), 1)
        (signature,) = init_function.signatures
        (_, param2) = signature.params
        self.assertEqual(param2.type.scope, 'other.name.SomeClass')

    def test_canonical_ordering_visitor(self):
        if False:
            i = 10
            return i + 15
        src1 = '\n      from typing import Any, TypeVar, Union\n      def f() -> Any:\n        raise MemoryError()\n        raise IOError()\n      def f(x: list[a]) -> Any: ...\n      def f(x: list[Union[b, c]]) -> Any: ...\n      def f(x: list[tuple[d]]) -> Any: ...\n      A = TypeVar("A")\n      C = TypeVar("C")\n      B = TypeVar("B")\n      D = TypeVar("D")\n      def f(d: A, c: B, b: C, a: D) -> Any: ...\n    '
        src2 = '\n      from typing import Any, Union\n      def f() -> Any:\n        raise IOError()\n        raise MemoryError()\n      def f(x: list[a]) -> Any: ...\n      def f(x: list[Union[b, c]]) -> Any: ...\n      def f(x: list[tuple[d]]) -> Any: ...\n      A = TypeVar("A")\n      C = TypeVar("C")\n      B = TypeVar("B")\n      D = TypeVar("D")\n      def f(d: A, c: B, b: C, a: D) -> Any: ...\n    '
        tree1 = self.Parse(src1)
        tree1 = tree1.Visit(pytd_visitors.CanonicalOrderingVisitor())
        tree2 = self.Parse(src2)
        tree2 = tree2.Visit(pytd_visitors.CanonicalOrderingVisitor())
        self.AssertSourceEquals(tree1, tree2)
        self.assertEqual(tree1.Lookup('f').signatures[0].template, tree2.Lookup('f').signatures[0].template)

    def test_superclasses(self):
        if False:
            for i in range(10):
                print('nop')
        src = textwrap.dedent('\n      class object:\n          pass\n      class A():\n          pass\n      class B():\n          pass\n      class C(A):\n          pass\n      class D(A,B):\n          pass\n      class E(C,D,A):\n          pass\n    ')
        ast = visitors.LookupClasses(self.Parse(src))
        data = ast.Visit(pytd_visitors.ExtractSuperClasses())
        self.assertCountEqual(['object'], [t.name for t in data[ast.Lookup('A')]])
        self.assertCountEqual(['object'], [t.name for t in data[ast.Lookup('B')]])
        self.assertCountEqual(['A'], [t.name for t in data[ast.Lookup('C')]])
        self.assertCountEqual(['A', 'B'], [t.name for t in data[ast.Lookup('D')]])
        self.assertCountEqual(['C', 'D', 'A'], [t.name for t in data[ast.Lookup('E')]])
if __name__ == '__main__':
    unittest.main()