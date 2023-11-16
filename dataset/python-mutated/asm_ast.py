from builtins import int as int_types

class AstNode(object):
    """
    Ast node object
    """

    def __neg__(self):
        if False:
            while True:
                i = 10
        if isinstance(self, AstInt):
            value = AstInt(-self.value)
        else:
            value = AstOp('-', self)
        return value

    def __add__(self, other):
        if False:
            print('Hello World!')
        return AstOp('+', self, other)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        return AstOp('-', self, other)

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        return AstOp('/', self, other)

    def __mod__(self, other):
        if False:
            return 10
        return AstOp('%', self, other)

    def __mul__(self, other):
        if False:
            print('Hello World!')
        return AstOp('*', self, other)

    def __lshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return AstOp('<<', self, other)

    def __rshift__(self, other):
        if False:
            while True:
                i = 10
        return AstOp('>>', self, other)

    def __xor__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return AstOp('^', self, other)

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return AstOp('|', self, other)

    def __and__(self, other):
        if False:
            i = 10
            return i + 15
        return AstOp('&', self, other)

class AstInt(AstNode):
    """
    Ast integer
    """

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s' % self.value

class AstId(AstNode):
    """
    Ast Id
    """

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s' % self.name

class AstMem(AstNode):
    """
    Ast memory deref
    """

    def __init__(self, ptr, size):
        if False:
            return 10
        assert isinstance(ptr, AstNode)
        assert isinstance(size, int_types)
        self.ptr = ptr
        self.size = size

    def __str__(self):
        if False:
            return 10
        return '@%d[%s]' % (self.size, self.ptr)

class AstOp(AstNode):
    """
    Ast operator
    """

    def __init__(self, op, *args):
        if False:
            i = 10
            return i + 15
        assert all((isinstance(arg, AstNode) for arg in args))
        self.op = op
        self.args = args

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.args) == 1:
            return '(%s %s)' % (self.op, self.args[0])
        return '(' + ('%s' % self.op).join((str(x) for x in self.args)) + ')'