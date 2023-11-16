"""Node for an OPENQASM expression list."""
from .node import Node

class ExpressionList(Node):
    """Node for an OPENQASM expression list.

    children are expression nodes.
    """

    def __init__(self, children):
        if False:
            i = 10
            return i + 15
        'Create the expression list node.'
        super().__init__('expression_list', children, None)

    def size(self):
        if False:
            while True:
                i = 10
        'Return the number of expressions.'
        return len(self.children)

    def qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding OPENQASM string.'
        return ','.join([self.children[j].qasm() for j in range(self.size())])