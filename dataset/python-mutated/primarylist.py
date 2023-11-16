"""Node for an OPENQASM primarylist."""
from .node import Node

class PrimaryList(Node):
    """Node for an OPENQASM primarylist.

    children is a list of primary nodes. Primary nodes are indexedid or id.
    """

    def __init__(self, children):
        if False:
            print('Hello World!')
        'Create the primarylist node.'
        super().__init__('primary_list', children, None)

    def size(self):
        if False:
            return 10
        'Return the size of the list.'
        return len(self.children)

    def qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding OPENQASM string.'
        return ','.join([self.children[j].qasm() for j in range(self.size())])