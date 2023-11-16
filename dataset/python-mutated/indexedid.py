"""Node for an OPENQASM indexed id."""
from .node import Node

class IndexedId(Node):
    """Node for an OPENQASM indexed id.

    children[0] is an id node.
    children[1] is an Int node.
    """

    def __init__(self, children):
        if False:
            i = 10
            return i + 15
        'Create the indexed id node.'
        super().__init__('indexed_id', children, None)
        self.id = children[0]
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = children[1].value

    def to_string(self, indent):
        if False:
            print('Hello World!')
        'Print with indent.'
        ind = indent * ' '
        print(ind, 'indexed_id', self.name, self.index)

    def qasm(self):
        if False:
            print('Hello World!')
        'Return the corresponding OPENQASM string.'
        return self.name + '[%d]' % self.index