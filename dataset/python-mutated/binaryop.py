"""Node for an OPENQASM binary operation expression."""
from qiskit.exceptions import MissingOptionalLibraryError
from .node import Node

class BinaryOp(Node):
    """Node for an OPENQASM binary operation expression.

    children[0] is the operation, as a binary operator node.
    children[1] is the left expression.
    children[2] is the right expression.
    """

    def __init__(self, children):
        if False:
            while True:
                i = 10
        'Create the binaryop node.'
        super().__init__('binop', children, None)

    def qasm(self):
        if False:
            return 10
        'Return the corresponding OPENQASM string.'
        return '(' + self.children[1].qasm() + self.children[0].value + self.children[2].qasm() + ')'

    def latex(self):
        if False:
            print('Hello World!')
        'Return the corresponding math mode latex string.'
        try:
            from pylatexenc.latexencode import utf8tolatex
        except ImportError as ex:
            raise MissingOptionalLibraryError('pylatexenc', 'latex-from-qasm exporter', 'pip install pylatexenc') from ex
        return utf8tolatex(self.sym())

    def real(self):
        if False:
            print('Hello World!')
        'Return the correspond floating point number.'
        operation = self.children[0].operation()
        lhs = self.children[1].real()
        rhs = self.children[2].real()
        return operation(lhs, rhs)

    def sym(self, nested_scope=None):
        if False:
            return 10
        'Return the correspond symbolic number.'
        operation = self.children[0].operation()
        lhs = self.children[1].sym(nested_scope)
        rhs = self.children[2].sym(nested_scope)
        return operation(lhs, rhs)