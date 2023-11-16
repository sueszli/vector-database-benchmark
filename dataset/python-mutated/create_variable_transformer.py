from .base_transformer import BaseTransformer
from .utils import FunctionNameLivenessAnalysis
from .variable_trans_func import create_undefined_var
__all__ = []

class CreateVariableTransformer(BaseTransformer):
    """ """

    def __init__(self, root):
        if False:
            while True:
                i = 10
        self.root = root
        FunctionNameLivenessAnalysis(self.root)

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Main function to transform AST.\n        '
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        bodys = node.body
        names = sorted(node.pd_scope.created_vars())
        for name in names:
            bodys[0:0] = [create_undefined_var(name)]
        return node