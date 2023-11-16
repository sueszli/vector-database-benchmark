import ast
from . import gast

def _generate_translators(to):
    if False:
        while True:
            i = 10

    class Translator(ast.NodeTransformer):

        def _visit(self, node):
            if False:
                return 10
            if isinstance(node, list):
                return [self._visit(n) for n in node]
            elif isinstance(node, ast.AST):
                return self.visit(node)
            else:
                return node

        def generic_visit(self, node):
            if False:
                for i in range(10):
                    print('nop')
            cls = type(node).__name__
            if not hasattr(to, cls):
                return
            new_node = getattr(to, cls)()
            for field in node._fields:
                setattr(new_node, field, self._visit(getattr(node, field)))
            for attr in getattr(node, '_attributes'):
                if hasattr(node, attr):
                    setattr(new_node, attr, getattr(node, attr))
            return new_node
    return Translator
AstToGAst = _generate_translators(gast)
GAstToAst = _generate_translators(ast)