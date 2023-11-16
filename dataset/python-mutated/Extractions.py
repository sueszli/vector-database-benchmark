""" Extracting visitors.

This is used for lookahead supporting abstract execution. We need to e.g.
know the variables written by a piece of code ahead of abstractly executing a
loop.
"""
from .Operations import VisitorNoopMixin, visitTree

class VariableUsageUpdater(VisitorNoopMixin):

    def __init__(self, old_variable, new_variable):
        if False:
            print('Hello World!')
        self.old_variable = old_variable
        self.new_variable = new_variable

    def onEnterNode(self, node):
        if False:
            return 10
        if node.isStatementAssignmentVariable() or node.isStatementDelVariable() or node.isStatementReleaseVariable():
            if node.getVariable() is self.old_variable:
                node.setVariable(self.new_variable)

def updateVariableUsage(provider, old_variable, new_variable):
    if False:
        i = 10
        return i + 15
    visitor = VariableUsageUpdater(old_variable=old_variable, new_variable=new_variable)
    visitTree(provider, visitor)