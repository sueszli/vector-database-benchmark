import ast
import sys
import textwrap
import unittest
from paddle.utils import gast

class GastNodeTransformer(gast.NodeTransformer):

    def __init__(self, root):
        if False:
            print('Hello World!')
        self.root = root

    def apply(self):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(self.root)

    def visit_Name(self, node):
        if False:
            return 10
        '\n        Param in func is ast.Name in PY2, but ast.arg in PY3.\n        It will be generally represented by gast.Name in gast.\n        '
        if isinstance(node.ctx, gast.Param) and node.id != 'self':
            node.id += '_new'
        return node

    def visit_With(self, node):
        if False:
            i = 10
            return i + 15
        '\n        The fileds `context_expr/optional_vars` of `ast.With` in PY2\n        is moved into `ast.With.items.withitem` in PY3.\n        It will be generally represented by gast.With.items.withitem in gast.\n        '
        assert hasattr(node, 'items')
        if node.items:
            withitem = node.items[0]
            assert isinstance(withitem, gast.withitem)
            if isinstance(withitem.context_expr, gast.Call):
                func = withitem.context_expr.func
                if isinstance(func, gast.Name):
                    func.id += '_new'
        return node

    def visit_Call(self, node):
        if False:
            print('Hello World!')
        '\n        The fileds `starargs/kwargs` of `ast.Call` in PY2\n        is moved into `Starred/keyword` in PY3.\n        It will be generally represented by gast.Starred/keyword in gast.\n        '
        assert hasattr(node, 'args')
        if node.args:
            assert isinstance(node.args[0], gast.Starred)
            if isinstance(node.args[0].value, gast.Name):
                node.args[0].value.id += '_new'
        assert hasattr(node, 'keywords')
        if node.keywords:
            assert isinstance(node.keywords[0], gast.keyword)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if False:
            while True:
                i = 10
        '\n        In PY3.8, ast.Num/Str/Bytes/None/False/True are merged into ast.Constant.\n        But these types are still available and will be deprecated in future versions.\n        ast.Num corresponds to gast.Num in PY2, and corresponds to gast.Constant in PY3.\n        '
        if isinstance(node.value, int):
            node.value *= 2
        return node

    def visit_Num(self, node):
        if False:
            print('Hello World!')
        '\n        ast.Num is available before PY3.8, and see visit_Constant for details.\n        '
        node.n *= 2
        return node

    def visit_Subscript(self, node):
        if False:
            print('Hello World!')
        "\n        Before PY3.8, the fields of ast.subscript keeps exactly same between PY2 and PY3.\n        After PY3.8, the field `slice` with ast.Slice will be changed into ast.Index(Tuple).\n        It will be generally represented by gast.Index or gast.Slice in gast.\n        Note: Paddle doesn't support PY3.8 currently.\n        "
        self.generic_visit(node)
        return node

def code_gast_ast(source):
    if False:
        i = 10
        return i + 15
    '\n    Transform source_code into gast.Node and modify it,\n    then back to ast.Node.\n    '
    source = textwrap.dedent(source)
    root = gast.parse(source)
    new_root = GastNodeTransformer(root).apply()
    ast_root = gast.gast_to_ast(new_root)
    return ast.dump(ast_root)

def code_ast(source):
    if False:
        i = 10
        return i + 15
    '\n    Transform source_code into ast.Node, then dump it.\n    '
    source = textwrap.dedent(source)
    root = ast.parse(source)
    return ast.dump(root)

class TestPythonCompatibility(unittest.TestCase):

    def _check_compatibility(self, source, target):
        if False:
            for i in range(10):
                print('nop')
        source_dump = code_gast_ast(source)
        target_dump = code_ast(target)
        self.assertEqual(source_dump, target_dump)

    def test_param_of_func(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Param in func is ast.Name in PY2, but ast.arg in PY3.\n        It will be generally represented by ast.Name in gast.\n        '
        source = '\n            def foo(x, y):\n                return x + y\n        '
        target = '\n            def foo(x_new, y_new):\n                return x + y\n        '
        self._check_compatibility(source, target)
    if sys.version_info < (3, 8):

        def test_with(self):
            if False:
                i = 10
                return i + 15
            '\n            The fileds `context_expr/optional_vars` of `ast.With` in PY2\n            is moved into `ast.With.items.withitem` in PY3.\n            '
            source = '\n            with guard():\n                a = 1\n            '
            target = '\n            with guard_new():\n                a = 1\n            '
            self._check_compatibility(source, target)

        def test_subscript_Index(self):
            if False:
                print('Hello World!')
            source = '\n                x = y()[10]\n            '
            target = '\n                x = y()[20]\n            '
            self._check_compatibility(source, target)

        def test_subscript_Slice(self):
            if False:
                return 10
            source = '\n                x = y()[10:20]\n            '
            target = '\n                x = y()[20:40]\n            '
            self._check_compatibility(source, target)

        def test_call(self):
            if False:
                i = 10
                return i + 15
            source = '\n                y = foo(*arg)\n            '
            target = '\n                y = foo(*arg_new)\n            '
            self._check_compatibility(source, target)
if __name__ == '__main__':
    unittest.main()