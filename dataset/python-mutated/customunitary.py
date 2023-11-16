"""Node for an OPENQASM custom gate statement."""
from .node import Node

class CustomUnitary(Node):
    """Node for an OPENQASM custom gate statement.

    children[0] is an id node.
    children[1] is an exp_list (if len==3) or primary_list.
    children[2], if present, is a primary_list.

    Has properties:
    .id = id node
    .name = gate name string
    .arguments = None or exp_list node
    .bitlist = primary_list node
    """

    def __init__(self, children):
        if False:
            print('Hello World!')
        'Create the custom gate node.'
        super().__init__('custom_unitary', children, None)
        self.id = children[0]
        self.name = self.id.name
        if len(children) == 3:
            self.arguments = children[1]
            self.bitlist = children[2]
        else:
            self.arguments = None
            self.bitlist = children[1]

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        string = self.name
        if self.arguments is not None:
            string += '(' + self.arguments.qasm() + ')'
        string += ' ' + self.bitlist.qasm() + ';'
        return string