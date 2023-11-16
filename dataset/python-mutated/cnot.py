"""Node for an OPENQASM CNOT statement."""
from .node import Node

class Cnot(Node):
    """Node for an OPENQASM CNOT statement.

    children[0], children[1] are id nodes if CX is inside a gate body,
    otherwise they are primary nodes.
    """

    def __init__(self, children):
        if False:
            return 10
        'Create the cnot node.'
        super().__init__('cnot', children, None)

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        return 'CX ' + self.children[0].qasm() + ',' + self.children[1].qasm() + ';'