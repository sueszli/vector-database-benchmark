import ast

class AssignmentVisitor(ast.NodeVisitor):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.assignment_nodes = []

    def visit_Assign(self, node):
        if False:
            print('Hello World!')
        self.assignment_nodes.append(node)
        self.generic_visit(node)

class CallVisitor(ast.NodeVisitor):

    def __init__(self):
        if False:
            return 10
        self.call_nodes = []

    def visit_Call(self, node):
        if False:
            return 10
        self.call_nodes.append(node)
        self.generic_visit(node)