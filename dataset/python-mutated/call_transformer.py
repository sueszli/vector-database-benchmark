from paddle.jit.dy2static.utils import ast_to_source_code, is_paddle_api
from paddle.utils import gast
from .base_transformer import BaseTransformer
from .utils import is_builtin
PDB_SET = 'pdb.set_trace'
__all__ = []

class CallTransformer(BaseTransformer):
    """
    This class transforms function calls into Static Graph Ast.
    """

    def __init__(self, root):
        if False:
            while True:
                i = 10
        self.root = root

    def _no_need_convert_call(self, node):
        if False:
            return 10
        "\n        Determines whether a function needs to be transformed by `convert_call`.\n        It doesn't need to be transformed when a function satisfies the following conditions:\n          1. It's a api of paddle\n          2. It's a python builtin function not include `len`, `zip`, `range` and `enumerate`\n        "
        assert isinstance(node, gast.Call)
        if is_paddle_api(node):
            return True
        func_str = ast_to_source_code(node.func).strip()
        try:
            need_convert_builtin_func_list = {'len', 'zip', 'range', 'enumerate', 'print'}
            is_builtin = eval(f'is_builtin({func_str})')
            need_convert = func_str in need_convert_builtin_func_list
            return is_builtin and (not need_convert)
        except Exception:
            return False

    def transform(self):
        if False:
            print('Hello World!')
        self.visit(self.root)

    def visit_Call(self, node):
        if False:
            return 10
        self.generic_visit(node)
        if self._no_need_convert_call(node):
            return node
        func_str = ast_to_source_code(node.func).strip()
        if PDB_SET in func_str:
            return node
        new_func_str = f'_jst.Call({func_str})'
        new_func_ast = gast.parse(new_func_str).body[0].value
        node.func = new_func_ast
        return node