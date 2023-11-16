""" Operations on the tree.

This is mostly for the different kinds of visits that the node tree can have.
You can visit a scope, a tree (module), or every scope of a tree (module).

"""

def visitTree(tree, visitor):
    if False:
        for i in range(10):
            print('nop')
    visitor.onEnterNode(tree)
    for visitable in tree.getVisitableNodes():
        if visitable is None:
            raise AssertionError("'None' child encountered", tree, tree.source_ref)
        visitTree(visitable, visitor)
    visitor.onLeaveNode(tree)

class VisitorNoopMixin(object):

    def onEnterNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Overloaded for operation before the node children were done.'

    def onLeaveNode(self, node):
        if False:
            while True:
                i = 10
        'Overloaded for operation after the node children were done.'