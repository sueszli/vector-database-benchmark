"""Node for an OPENQASM custom gate body."""
from .node import Node

class GateBody(Node):
    """Node for an OPENQASM custom gate body.

    children is a list of gate operation nodes.
    These are one of barrier, custom_unitary, U, or CX.
    """

    def __init__(self, children):
        if False:
            return 10
        'Create the gatebody node.'
        super().__init__('gate_body', children, None)

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        string = ''
        for children in self.children:
            string += '  ' + children.qasm() + '\n'
        return string

    def calls(self):
        if False:
            print('Hello World!')
        'Return a list of custom gate names in this gate body.'
        lst = []
        for children in self.children:
            if children.type == 'custom_unitary':
                lst.append(children.name)
        return lst