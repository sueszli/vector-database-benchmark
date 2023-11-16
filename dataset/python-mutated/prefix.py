"""Node for an OPENQASM prefix expression."""
from qiskit.exceptions import MissingOptionalLibraryError
from .node import Node

class Prefix(Node):
    """Node for an OPENQASM prefix expression.

    children[0] is a unary operator node.
    children[1] is an expression node.
    """

    def __init__(self, children):
        if False:
            print('Hello World!')
        'Create the prefix node.'
        super().__init__('prefix', children, None)

    def qasm(self):
        if False:
            while True:
                i = 10
        'Return the corresponding OPENQASM string.'
        return self.children[0].value + '(' + self.children[1].qasm() + ')'

    def latex(self):
        if False:
            i = 10
            return i + 15
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
        expr = self.children[1].real()
        return operation(expr)

    def sym(self, nested_scope=None):
        if False:
            print('Hello World!')
        'Return the correspond symbolic number.'
        operation = self.children[0].operation()
        expr = self.children[1].sym(nested_scope)
        return operation(expr)