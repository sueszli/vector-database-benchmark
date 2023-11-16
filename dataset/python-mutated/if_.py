"""Node for an OPENQASM if statement."""
from .node import Node

class If(Node):
    """Node for an OPENQASM if statement.

    children[0] is an id node.
    children[1] is an integer node.
    children[2] is quantum operation node, including U, CX, custom_unitary,
    measure, reset, (and BUG: barrier, if).
    """

    def __init__(self, children):
        if False:
            print('Hello World!')
        'Create the if node.'
        super().__init__('if', children, None)

    def qasm(self):
        if False:
            i = 10
            return i + 15
        'Return the corresponding OPENQASM string.'
        return 'if(' + self.children[0].qasm() + '==' + str(self.children[1].value) + ') ' + self.children[2].qasm()