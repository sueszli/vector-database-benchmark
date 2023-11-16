""" Finalizations. Last steps directly before code creation is called.

Here the final tasks are executed. Things normally volatile during optimization
can be computed here, so the code generation can be quick and doesn't have to
check it many times.

"""
from nuitka.tree import Operations
from .FinalizeMarkups import FinalizeMarkups

def prepareCodeGeneration(tree):
    if False:
        return 10
    visitor = FinalizeMarkups()
    Operations.visitTree(tree, visitor)