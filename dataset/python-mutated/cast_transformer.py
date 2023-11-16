from paddle.jit.dy2static.utils import ast_to_source_code
from paddle.utils import gast
from .base_transformer import BaseTransformer
__all__ = []

class CastTransformer(BaseTransformer):
    """
    This class transforms type casting into Static Graph Ast.
    """

    def __init__(self, root):
        if False:
            while True:
                i = 10
        self.root = root
        self._castable_type = {'bool', 'int', 'float'}

    def transform(self):
        if False:
            print('Hello World!')
        self.visit(self.root)

    def visit_Call(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        func_str = ast_to_source_code(node.func).strip()
        if func_str in self._castable_type and len(node.args) > 0:
            args_str = ast_to_source_code(node.args[0]).strip()
            new_func_str = f"_jst.AsDtype({args_str}, '{func_str}')"
            new_node = gast.parse(new_func_str).body[0].value
            return new_node
        return node