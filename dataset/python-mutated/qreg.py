"""Node for an OPENQASM qreg statement."""
from .node import Node

class Qreg(Node):
    """Node for an OPENQASM qreg statement.

    children[0] is an indexedid node.
    """

    def __init__(self, children):
        if False:
            while True:
                i = 10
        'Create the qreg node.'
        super().__init__('qreg', children, None)
        self.id = children[0]
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = self.id.index

    def to_string(self, indent):
        if False:
            return 10
        'Print the node data, with indent.'
        ind = indent * ' '
        print(ind, 'qreg')
        self.children[0].to_string(indent + 3)

    def qasm(self):
        if False:
            print('Hello World!')
        'Return the corresponding OPENQASM string.'
        return 'qreg ' + self.id.qasm() + ';'