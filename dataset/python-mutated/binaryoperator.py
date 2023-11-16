"""Node for an OPENQASM binary operator."""
import operator
from .node import Node
from .nodeexception import NodeException
VALID_OPERATORS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv, '^': operator.pow}

class BinaryOperator(Node):
    """Node for an OPENQASM binary operator.

    This node has no children. The data is in the value field.
    """

    def __init__(self, operation):
        if False:
            return 10
        'Create the operator node.'
        super().__init__('operator', None, None)
        self.value = operation

    def operation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the operator as a function f(left, right).\n        '
        try:
            return VALID_OPERATORS[self.value]
        except KeyError as ex:
            raise NodeException(f"internal error: undefined operator '{self.value}'") from ex

    def qasm(self):
        if False:
            while True:
                i = 10
        'Return the OpenQASM 2 representation.'
        return self.value