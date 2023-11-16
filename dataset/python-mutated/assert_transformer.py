from paddle.jit.dy2static.utils import ast_to_source_code
from paddle.utils import gast
from .base_transformer import BaseTransformer
__all__ = []

class AssertTransformer(BaseTransformer):
    """
    A class transforms python assert to convert_assert.
    """

    def __init__(self, root):
        if False:
            return 10
        self.root = root

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        self.visit(self.root)

    def visit_Assert(self, node):
        if False:
            for i in range(10):
                print('nop')
        convert_assert_node = gast.parse('_jst.Assert({test}, {msg})'.format(test=ast_to_source_code(node.test), msg=ast_to_source_code(node.msg) if node.msg else '')).body[0].value
        return gast.Expr(value=convert_assert_node)