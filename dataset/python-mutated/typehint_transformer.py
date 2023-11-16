from .base_transformer import BaseTransformer
__all__ = []

class TypeHintTransformer(BaseTransformer):
    """
    A class remove all the typehint in gast.Name(annotation).
    Please put it behind other transformers because other transformer may relay on typehints.
    """

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        self.root = root

    def transform(self):
        if False:
            return 10
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        if False:
            return 10
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.annotation = None
        self.generic_visit(node)
        return node