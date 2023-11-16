"""Tests for loader module."""
import os
import textwrap
import gast
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect

class LoaderTest(test.TestCase):

    def assertAstMatches(self, actual_node, expected_node_src):
        if False:
            while True:
                i = 10
        expected_node = gast.parse(expected_node_src).body[0]
        msg = 'AST did not match expected:\n{}\nActual:\n{}'.format(pretty_printer.fmt(expected_node), pretty_printer.fmt(actual_node))
        self.assertTrue(ast_util.matches(actual_node, expected_node), msg)

    def test_parse_load_identity(self):
        if False:
            while True:
                i = 10

        def test_fn(x):
            if False:
                while True:
                    i = 10
            a = True
            b = ''
            if a:
                b = x + 1
            return b
        (node, _) = parser.parse_entity(test_fn, future_features=())
        (module, _, _) = loader.load_ast(node)
        source = tf_inspect.getsource(module.test_fn)
        expected_node_src = textwrap.dedent(tf_inspect.getsource(test_fn))
        self.assertAstMatches(node, source)
        self.assertAstMatches(node, expected_node_src)

    def test_load_ast(self):
        if False:
            i = 10
            return i + 15
        node = gast.FunctionDef(name='f', args=gast.arguments(args=[gast.Name('a', ctx=gast.Param(), annotation=None, type_comment=None)], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[gast.Return(gast.BinOp(op=gast.Add(), left=gast.Name('a', ctx=gast.Load(), annotation=None, type_comment=None), right=gast.Constant(1, kind=None)))], decorator_list=[], returns=None, type_comment=None)
        (module, source, _) = loader.load_ast(node)
        expected_node_src = '\n      # coding=utf-8\n      def f(a):\n          return (a + 1)\n    '
        expected_node_src = textwrap.dedent(expected_node_src)
        self.assertAstMatches(node, source)
        self.assertAstMatches(node, expected_node_src)
        self.assertEqual(2, module.f(1))
        with open(module.__file__, 'r') as temp_output:
            self.assertAstMatches(node, temp_output.read())

    def test_load_source(self):
        if False:
            print('Hello World!')
        test_source = textwrap.dedent(u"\n      # coding=utf-8\n      def f(a):\n        '日本語 Δθₜ ← Δθₜ₋₁ + ∇Q(sₜ, aₜ)(rₜ + γₜ₊₁ max Q(⋅))'\n        return a + 1\n    ")
        (module, _) = loader.load_source(test_source, delete_on_exit=True)
        self.assertEqual(module.f(1), 2)
        self.assertEqual(module.f.__doc__, '日本語 Δθₜ ← Δθₜ₋₁ + ∇Q(sₜ, aₜ)(rₜ + γₜ₊₁ max Q(⋅))')

    def test_cleanup(self):
        if False:
            return 10
        test_source = textwrap.dedent('')
        (_, filename) = loader.load_source(test_source, delete_on_exit=True)
        os.unlink(filename)
if __name__ == '__main__':
    test.main()