from paddle.utils import gast
from .base_transformer import BaseTransformer
__all__ = []

class EarlyReturnTransformer(BaseTransformer):
    """
    Transform if/else return statement of Dygraph into Static Graph.
    """

    def __init__(self, root):
        if False:
            print('Hello World!')
        self.root = root

    def transform(self):
        if False:
            while True:
                i = 10
        '\n        Main function to transform AST.\n        '
        self.visit(self.root)

    def is_define_return_in_if(self, node):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(node, gast.If), 'Type of input node should be gast.If, but received %s .' % type(node)
        for child in node.body:
            if isinstance(child, gast.Return):
                return True
        return False

    def visit_block_nodes(self, nodes):
        if False:
            return 10
        result_nodes = []
        destination_nodes = result_nodes
        for node in nodes:
            rewritten_node = self.visit(node)
            if isinstance(rewritten_node, (list, tuple)):
                destination_nodes.extend(rewritten_node)
            else:
                destination_nodes.append(rewritten_node)
            if isinstance(node, gast.If) and self.is_define_return_in_if(node):
                destination_nodes = node.orelse
                while len(destination_nodes) > 0 and isinstance(destination_nodes[0], gast.If) and self.is_define_return_in_if(destination_nodes[0]):
                    destination_nodes = destination_nodes[0].orelse
        return result_nodes

    def visit_If(self, node):
        if False:
            while True:
                i = 10
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_While(self, node):
        if False:
            print('Hello World!')
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_For(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        node.body = self.visit_block_nodes(node.body)
        return node