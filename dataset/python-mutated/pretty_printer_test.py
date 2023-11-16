"""Tests for pretty_printer module."""
import ast
import textwrap
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.platform import test

class PrettyPrinterTest(test.TestCase):

    def test_unicode_bytes(self):
        if False:
            return 10
        source = textwrap.dedent("\n    def f():\n      return b'b', u'u', 'depends_py2_py3'\n    ")
        node = ast.parse(source)
        self.assertIsNotNone(pretty_printer.fmt(node))

    def test_format(self):
        if False:
            print('Hello World!')
        node = ast.FunctionDef(name='f', args=ast.arguments(args=[ast.Name(id='a', ctx=ast.Param())], vararg=None, kwarg=None, defaults=[]), body=[ast.Return(ast.BinOp(op=ast.Add(), left=ast.Name(id='a', ctx=ast.Load()), right=ast.Num(1)))], decorator_list=[], returns=None)
        self.assertIsNotNone(pretty_printer.fmt(node))
if __name__ == '__main__':
    test.main()