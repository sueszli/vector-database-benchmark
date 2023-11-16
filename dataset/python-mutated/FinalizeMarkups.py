""" Finalize the markups

Set flags on functions and classes to indicate if a locals dict is really
needed.

Set a flag on loops if they really need to catch Continue and Break exceptions
or if it can be more simple code.

Set a flag on return statements and functions that require the use of
"ReturnValue" exceptions, or if it can be more simple code.

Set a flag on re-raises of exceptions if they can be simple throws or if they
are in another context.

"""
from nuitka import Tracing
from nuitka.__past__ import unicode
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.PythonVersions import python_version
from nuitka.tree.Operations import VisitorNoopMixin
imported_names = OrderedSet()

def getImportedNames():
    if False:
        while True:
            i = 10
    return imported_names

class FinalizeMarkups(VisitorNoopMixin):

    def onEnterNode(self, node):
        if False:
            print('Hello World!')
        try:
            self._onEnterNode(node)
        except Exception:
            Tracing.printError('Problem with %r at %s' % (node, node.getSourceReference().getAsString()))
            raise

    def _onEnterNode(self, node):
        if False:
            while True:
                i = 10
        if node.isStatementReturn() or node.isStatementGeneratorReturn():
            search = node
            in_tried_block = False
            search = search.getParentReturnConsumer()
            if search.isExpressionGeneratorObjectBody() or search.isExpressionCoroutineObjectBody() or search.isExpressionAsyncgenObjectBody():
                if in_tried_block:
                    search.markAsNeedsGeneratorReturnHandling(2)
                else:
                    search.markAsNeedsGeneratorReturnHandling(1)
        if node.isExpressionBuiltinImport() and node.follow_attempted:
            module_name = node.subnode_name
            if module_name.isCompileTimeConstant():
                imported_module_name = module_name.getCompileTimeConstant()
                if type(imported_module_name) in (str, unicode):
                    if imported_module_name:
                        imported_names.add(imported_module_name)
        if node.isExpressionFunctionCreation():
            if not node.getParent().isExpressionFunctionCall() or node.getParent().subnode_function is not node:
                node.subnode_function_ref.getFunctionBody().markAsNeedsCreation()
        if node.isExpressionFunctionCall():
            node.subnode_function.subnode_function_ref.getFunctionBody().markAsDirectlyCalled()
        if node.isExpressionFunctionRef():
            function_body = node.getFunctionBody()
            parent_module = function_body.getParentModule()
            node_module = node.getParentModule()
            if node_module is not parent_module:
                function_body.markAsCrossModuleUsed()
                node_module.addCrossUsedFunction(function_body)
        if node.isStatementAssignmentVariable():
            target_var = node.getVariable()
            assign_source = node.subnode_source
            if assign_source.isExpressionOperationBinary():
                left_arg = assign_source.subnode_left
                if left_arg.isExpressionVariableRef() or left_arg.isExpressionTempVariableRef():
                    if assign_source.subnode_left.getVariable() is target_var:
                        if assign_source.isInplaceSuspect():
                            node.markAsInplaceSuspect()
                elif left_arg.isExpressionLocalsVariableRefOrFallback():
                    assign_source.removeMarkAsInplaceSuspect()
        if python_version < 768 and node.isStatementPublishException():
            node.getParentStatementsFrame().markAsFrameExceptionPreserving()
        if python_version >= 768:
            if node.isExpressionYield() or node.isExpressionYieldFrom() or node.isExpressionYieldFromAwaitable():
                search = node.getParent()
                while not search.isExpressionGeneratorObjectBody() and (not search.isExpressionCoroutineObjectBody()) and (not search.isExpressionAsyncgenObjectBody()):
                    last_search = search
                    search = search.getParent()
                    if search.isStatementTry() and last_search == search.subnode_except_handler:
                        node.markAsExceptionPreserving()
                        break