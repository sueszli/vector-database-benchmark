"""Node for an OPENQASM idlist."""
from .node import Node

class IdList(Node):
    """Node for an OPENQASM idlist.

    children is a list of id nodes.
    """

    def __init__(self, children):
        if False:
            for i in range(10):
                print('nop')
        'Create the idlist node.'
        super().__init__('id_list', children, None)

    def size(self):
        if False:
            print('Hello World!')
        'Return the length of the list.'
        return len(self.children)

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        return ','.join([self.children[j].qasm() for j in range(self.size())])