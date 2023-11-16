from paddle.utils import gast
from .base_transformer import BaseTransformer
from .utils import ast_to_source_code
__all__ = []

class TensorShapeTransformer(BaseTransformer):
    """
    This class transforms variable.shape  into Static Graph Ast.
    All 'xxx.shape' will be converted int '_jst.Shape(x)'.
    """

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        self.root = root

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        self.visit(self.root)

    def visit_Attribute(self, node):
        if False:
            i = 10
            return i + 15
        self.generic_visit(node)
        if node.attr == 'shape':
            args = ast_to_source_code(node.value).strip()
            if args != 'paddle':
                convert_shape_func = f'_jst.Shape({args})'
                shape_node = gast.parse(convert_shape_func).body[0].value
                return shape_node
        return node