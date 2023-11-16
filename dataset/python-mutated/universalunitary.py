"""Node for an OPENQASM U statement."""
from .node import Node

class UniversalUnitary(Node):
    """Node for an OPENQASM U statement.

    children[0] is an expressionlist node.
    children[1] is a primary node (id or indexedid).
    """

    def __init__(self, children):
        if False:
            i = 10
            return i + 15
        'Create the U node.'
        super().__init__('universal_unitary', children)
        self.arguments = children[0]
        self.bitlist = children[1]

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        return 'U(' + self.children[0].qasm() + ') ' + self.children[1].qasm() + ';'