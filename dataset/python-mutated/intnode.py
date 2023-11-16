"""Node for an OPENQASM integer."""
from .node import Node

class Int(Node):
    """Node for an OPENQASM integer.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        if False:
            i = 10
            return i + 15
        'Create the integer node.'
        super().__init__('int', None, None)
        self.value = id

    def to_string(self, indent):
        if False:
            i = 10
            return i + 15
        'Print with indent.'
        ind = indent * ' '
        print(ind, 'int', self.value)

    def qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding OPENQASM string.'
        return '%d' % self.value

    def latex(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the corresponding math mode latex string.'
        return '%d' % self.value

    def sym(self, nested_scope=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the correspond symbolic number.'
        del nested_scope
        return float(self.value)

    def real(self, nested_scope=None):
        if False:
            return 10
        'Return the correspond floating point number.'
        del nested_scope
        return float(self.value)