""" In-lining of functions.

Done by assigning the argument values to variables, and producing an outline
from the in-lined function.
"""
from nuitka.nodes.OutlineNodes import ExpressionOutlineBody
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableReleaseNodes import makeStatementsReleaseVariables
from nuitka.tree.Operations import VisitorNoopMixin, visitTree
from nuitka.tree.ReformulationTryFinallyStatements import makeTryFinallyStatement
from nuitka.tree.TreeHelpers import makeStatementsSequence

class VariableScopeUpdater(VisitorNoopMixin):

    def __init__(self, locals_scope, variable_translation):
        if False:
            print('Hello World!')
        self.locals_scope = locals_scope
        self.variable_translation = variable_translation

    def onEnterNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(node, 'variable'):
            if node.variable in self.variable_translation:
                node.variable = self.variable_translation[node.variable]
        if hasattr(node, 'locals_scope'):
            node.locals_scope = self.locals_scope

def updateLocalsScope(provider, locals_scope, variable_translation):
    if False:
        print('Hello World!')
    visitor = VariableScopeUpdater(locals_scope=locals_scope, variable_translation=variable_translation)
    visitTree(provider, visitor)

def convertFunctionCallToOutline(provider, function_body, values, call_source_ref):
    if False:
        return 10
    function_source_ref = function_body.getSourceReference()
    outline_body = ExpressionOutlineBody(provider=provider, name='inline', source_ref=function_source_ref)
    clone = function_body.subnode_body.makeClone()
    (locals_scope_clone, variable_translation) = function_body.locals_scope.makeClone(clone)
    updateLocalsScope(clone, locals_scope=locals_scope_clone, variable_translation=variable_translation)
    argument_names = function_body.getParameters().getParameterNames()
    assert len(argument_names) == len(values), (argument_names, values)
    statements = []
    for (argument_name, value) in zip(argument_names, values):
        statements.append(makeStatementAssignmentVariable(variable=variable_translation[argument_name], source=value, source_ref=call_source_ref))
    body = makeStatementsSequence(statements=(statements, clone), allow_none=False, source_ref=function_source_ref)
    auto_releases = function_body.getFunctionVariablesWithAutoReleases()
    if auto_releases:
        body = makeTryFinallyStatement(provider=outline_body, tried=body, final=makeStatementsReleaseVariables(variables=auto_releases, source_ref=function_source_ref), source_ref=function_source_ref)
    outline_body.setChildBody(body)
    return outline_body