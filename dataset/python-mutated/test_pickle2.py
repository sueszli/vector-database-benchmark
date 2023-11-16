"""Tests for loading and saving pickled files."""
import pickle
from pytype.tests import test_base
from pytype.tests import test_utils

class PickleTest(test_base.BaseTest):
    """Tests for loading and saving pickled files."""

    def test_container(self):
        if False:
            for i in range(10):
                print('nop')
        pickled = self.Infer('\n      import collections, json\n      def f() -> collections.OrderedDict[int, int]:\n        return collections.OrderedDict({1: 1})\n      def g() -> json.JSONDecoder:\n        return json.JSONDecoder()\n    ', pickle=True, module_name='foo')
        with test_utils.Tempdir() as d:
            u = d.create_file('u.pickled', pickled)
            ty = self.Infer('\n        import u\n        r = u.f()\n      ', deep=False, pythonpath=[''], imports_map={'u': u})
            self.assertTypesMatchPytd(ty, '\n        from typing import OrderedDict\n        import u\n        r = ...  # type: OrderedDict[int, int]\n      ')

    def test_nested_class_name_clash(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        pass\n      class Bar:\n        class Foo(Foo):\n          pass\n    ', module_name='foo', pickle=True)
        ast = pickle.loads(ty).ast
        (base,) = ast.Lookup('foo.Bar').Lookup('foo.Bar.Foo').bases
        self.assertEqual(base.name, 'foo.Foo')

    def test_late_type_indirection(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      class Foo:\n        pass\n    ', {'pickle': True}), ('bar.py', '\n      import foo\n      Bar = foo.Foo\n    ', {'pickle': True}), ('baz.pyi', '\n      import bar\n      class Baz:\n        x: bar.Bar\n    ', {'pickle': True})]):
            self.Check("\n        import baz\n        assert_type(baz.Baz.x, 'foo.Foo')\n      ")
if __name__ == '__main__':
    test_base.main()