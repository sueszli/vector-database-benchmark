"""
Topic: 非递归的观察者模式
Desc : 
"""
import types

class Node:
    pass

class NodeVisitor:

    def visit(self, node):
        if False:
            while True:
                i = 10
        stack = [node]
        last_result = None
        while stack:
            try:
                last = stack[-1]
                if isinstance(last, types.GeneratorType):
                    stack.append(last.send(last_result))
                    last_result = None
                elif isinstance(last, Node):
                    stack.append(self._visit(stack.pop()))
                else:
                    last_result = stack.pop()
            except StopIteration:
                stack.pop()
        return last_result

    def _visit(self, node):
        if False:
            for i in range(10):
                print('nop')
        methname = 'visit_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)

    def generic_visit(self, node):
        if False:
            while True:
                i = 10
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))

class UnaryOperator(Node):

    def __init__(self, operand):
        if False:
            return 10
        self.operand = operand

class BinaryOperator(Node):

    def __init__(self, left, right):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        self.value = value

class Evaluator(NodeVisitor):

    def visit_Number(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node.value

    def visit_Add(self, node):
        if False:
            while True:
                i = 10
        yield ((yield node.left) + (yield node.right))

    def visit_Sub(self, node):
        if False:
            i = 10
            return i + 15
        yield ((yield node.left) - (yield node.right))

    def visit_Mul(self, node):
        if False:
            return 10
        yield ((yield node.left) * (yield node.right))

    def visit_Div(self, node):
        if False:
            for i in range(10):
                print('nop')
        yield ((yield node.left) / (yield node.right))

    def visit_Negate(self, node):
        if False:
            for i in range(10):
                print('nop')
        yield (-(yield node.operand))
if __name__ == '__main__':
    t1 = Sub(Number(3), Number(4))
    t2 = Mul(Number(2), t1)
    t3 = Div(t2, Number(5))
    t4 = Add(Number(1), t3)
    e = Evaluator()
    print(e.visit(t4))