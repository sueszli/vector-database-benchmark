"""Node for an OpenQASM 2 unary operator."""
import operator
from .node import Node
from .nodeexception import NodeException
VALID_OPERATORS = {'+': operator.pos, '-': operator.neg}

class UnaryOperator(Node):
    """Node for an OpenQASM 2 unary operator.

    This node has no children. The data is in the value field.
    """

    def __init__(self, operation):
        if False:
            return 10
        'Create the operator node.'
        super().__init__('unary_operator', None, None)
        self.value = operation

    def operation(self):
        if False:
            return 10
        '\n        Return the operator as a function f(left, right).\n        '
        try:
            return VALID_OPERATORS[self.value]
        except KeyError as ex:
            raise NodeException(f"internal error: undefined prefix '{self.value}'") from ex

    def qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Return OpenQASM 2 representation.'
        return self.value