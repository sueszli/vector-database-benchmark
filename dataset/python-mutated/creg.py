"""Node for an OPENQASM creg statement."""
from .node import Node

class Creg(Node):
    """Node for an OPENQASM creg statement.

    children[0] is an indexedid node.
    """

    def __init__(self, children):
        if False:
            return 10
        'Create the creg node.'
        super().__init__('creg', children, None)
        self.id = children[0]
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = self.id.index

    def to_string(self, indent):
        if False:
            i = 10
            return i + 15
        'Print the node data, with indent.'
        ind = indent * ' '
        print(ind, 'creg')
        self.children[0].to_string(indent + 3)

    def qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding OPENQASM string.'
        return 'creg ' + self.id.qasm() + ';'