"""Tests for traces.visitor."""
import ast
import sys
import textwrap
from pytype.ast import visitor
import unittest

class _VisitOrderVisitor(visitor.BaseVisitor):
    """Tests visit order."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.funcs = []

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        self.funcs.append(node.name)

class _VisitReplaceVisitor(visitor.BaseVisitor):
    """Tests visit()'s node replacement functionality."""

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.id == 'x':
            return True
        elif node.id == 'y':
            return False
        else:
            return None

class _GenericVisitVisitor(visitor.BaseVisitor):
    """Tests generic_visit()."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.nodes = []

    def generic_visit(self, node):
        if False:
            while True:
                i = 10
        self.nodes.append(node.__class__.__name__)

class _EnterVisitor(visitor.BaseVisitor):
    """Tests enter() by recording names."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.names = []

    def enter_Name(self, node):
        if False:
            return 10
        self.names.append(node.id)

class _LeaveVisitor(_EnterVisitor):
    """Tests leave() by discarding names recorded by enter()."""

    def leave_Name(self, node):
        if False:
            i = 10
            return i + 15
        self.names.pop()

class custom_ast:
    """Tests a custom ast module."""

    class AST:
        pass

    class Thing(AST):
        pass

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return type(name, (custom_ast.AST,), {})

    def iter_fields(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node, custom_ast.Thing):
            return []
        elif isinstance(node, custom_ast.AST):
            return [('thing', node.thing)]

    def parse(self, unused_src):
        if False:
            while True:
                i = 10
        module = custom_ast.AST()
        module.thing = custom_ast.Thing()
        return module

class BaseVisitorTest(unittest.TestCase):
    """Tests for visitor.BaseVisitor."""

    def test_visit_order(self):
        if False:
            while True:
                i = 10
        module = ast.parse(textwrap.dedent('\n      def f():\n        def g():\n          def h():\n            pass\n    '))
        v = _VisitOrderVisitor(ast)
        v.visit(module)
        self.assertEqual(v.funcs, ['h', 'g', 'f'])

    def test_visit_replace(self):
        if False:
            return 10
        module = ast.parse(textwrap.dedent('\n      x.upper()\n      y.upper()\n      z.upper()\n    '))
        v = _VisitReplaceVisitor(ast)
        v.visit(module)
        x = module.body[0].value.func.value
        y = module.body[1].value.func.value
        z = module.body[2].value.func.value
        self.assertIs(x, True)
        self.assertIs(y, False)
        self.assertIsInstance(z, ast.Name)

    def test_generic_visit(self):
        if False:
            print('Hello World!')
        module = ast.parse('x = 0')
        v = _GenericVisitVisitor(ast)
        v.visit(module)
        if sys.hexversion >= 50855936:
            constant = 'Constant'
        else:
            constant = 'Num'
        self.assertEqual(v.nodes, ['Store', 'Name', constant, 'Assign', 'Module'])

    def test_enter(self):
        if False:
            for i in range(10):
                print('nop')
        module = ast.parse(textwrap.dedent('\n      x = 0\n      y = 1\n      z = 2\n    '))
        v = _EnterVisitor(ast)
        v.visit(module)
        self.assertEqual(v.names, ['x', 'y', 'z'])

    def test_leave(self):
        if False:
            while True:
                i = 10
        module = ast.parse(textwrap.dedent('\n      x = 0\n      y = 1\n      z = 2\n    '))
        v = _LeaveVisitor(ast)
        v.visit(module)
        self.assertFalse(v.names)

    def test_custom_ast(self):
        if False:
            print('Hello World!')
        custom_ast_module = custom_ast()
        module = custom_ast_module.parse('')
        v = _GenericVisitVisitor(custom_ast_module)
        v.visit(module)
        self.assertEqual(v.nodes, ['Thing', 'AST'])
if __name__ == '__main__':
    unittest.main()