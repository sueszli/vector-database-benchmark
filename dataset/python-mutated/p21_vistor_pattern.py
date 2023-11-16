"""
Topic: 观察者模式
Desc : 
"""

class Node:
    pass

class UnaryOperator(Node):

    def __init__(self, operand):
        if False:
            print('Hello World!')
        self.operand = operand

class BinaryOperator(Node):

    def __init__(self, left, right):
        if False:
            i = 10
            return i + 15
        self.left = left
        self.right = right

class Add(BinaryOperator):
    pass

class Sub(BinaryOperator):
    pass

class Mul(BinaryOperator):
    pass

class Div(BinaryOperator):
    pass

class Negate(UnaryOperator):
    pass

class Number(Node):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value
t1 = Sub(Number(3), Number(4))
t2 = Mul(Number(2), t1)
t3 = Div(t2, Number(5))
t4 = Add(Number(1), t3)

class NodeVisitor:

    def visit(self, node):
        if False:
            print('Hello World!')
        methname = 'visit_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)

    def generic_visit(self, node):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))

class Evaluator(NodeVisitor):

    def visit_Number(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node.value

    def visit_Add(self, node):
        if False:
            return 10
        return self.visit(node.left) + self.visit(node.right)

    def visit_Sub(self, node):
        if False:
            print('Hello World!')
        return self.visit(node.left) - self.visit(node.right)

    def visit_Mul(self, node):
        if False:
            while True:
                i = 10
        return self.visit(node.left) * self.visit(node.right)

    def visit_Div(self, node):
        if False:
            print('Hello World!')
        return self.visit(node.left) / self.visit(node.right)

    def visit_Negate(self, node):
        if False:
            for i in range(10):
                print('nop')
        return -node.operand
e = Evaluator()
print(e.visit(t4))

class StackCode(NodeVisitor):

    def generate_code(self, node):
        if False:
            i = 10
            return i + 15
        self.instructions = []
        self.visit(node)
        return self.instructions

    def visit_Number(self, node):
        if False:
            print('Hello World!')
        self.instructions.append(('PUSH', node.value))

    def binop(self, node, instruction):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.left)
        self.visit(node.right)
        self.instructions.append((instruction,))

    def visit_Add(self, node):
        if False:
            i = 10
            return i + 15
        self.binop(node, 'ADD')

    def visit_Sub(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.binop(node, 'SUB')

    def visit_Mul(self, node):
        if False:
            while True:
                i = 10
        self.binop(node, 'MUL')

    def visit_Div(self, node):
        if False:
            print('Hello World!')
        self.binop(node, 'DIV')

    def unaryop(self, node, instruction):
        if False:
            return 10
        self.visit(node.operand)
        self.instructions.append((instruction,))

    def visit_Negate(self, node):
        if False:
            i = 10
            return i + 15
        self.unaryop(node, 'NEG')

class HTTPHandler:

    def handle(self, request):
        if False:
            print('Hello World!')
        methname = 'do_' + request.request_method
        getattr(self, methname)(request)

    def do_GET(self, request):
        if False:
            i = 10
            return i + 15
        pass

    def do_POST(self, request):
        if False:
            for i in range(10):
                print('nop')
        pass

    def do_HEAD(self, request):
        if False:
            print('Hello World!')
        pass