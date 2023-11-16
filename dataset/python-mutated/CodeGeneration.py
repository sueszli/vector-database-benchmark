from __future__ import absolute_import
from .Visitor import VisitorTransform
from .Nodes import StatListNode

class ExtractPxdCode(VisitorTransform):
    """
    Finds nodes in a pxd file that should generate code, and
    returns them in a StatListNode.

    The result is a tuple (StatListNode, ModuleScope), i.e.
    everything that is needed from the pxd after it is processed.

    A purer approach would be to separately compile the pxd code,
    but the result would have to be slightly more sophisticated
    than pure strings (functions + wanted interned strings +
    wanted utility code + wanted cached objects) so for now this
    approach is taken.
    """

    def __call__(self, root):
        if False:
            print('Hello World!')
        self.funcs = []
        self.visitchildren(root)
        return (StatListNode(root.pos, stats=self.funcs), root.scope)

    def visit_FuncDefNode(self, node):
        if False:
            while True:
                i = 10
        self.funcs.append(node)
        return node

    def visit_Node(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visitchildren(node)
        return node