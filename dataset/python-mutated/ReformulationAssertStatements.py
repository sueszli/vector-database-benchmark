""" Reformulation of assert statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.BuiltinRefNodes import ExpressionBuiltinExceptionRef
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.ContainerMakingNodes import makeExpressionMakeTuple
from nuitka.nodes.ExceptionNodes import StatementRaiseException
from nuitka.nodes.OperatorNodesUnary import ExpressionOperationNot
from nuitka.Options import hasPythonFlagNoAsserts
from nuitka.PythonVersions import python_version
from .TreeHelpers import buildNode

def buildAssertNode(provider, node, source_ref):
    if False:
        print('Hello World!')
    exception_value = buildNode(provider, node.msg, source_ref, True)
    if hasPythonFlagNoAsserts():
        return None
    if exception_value is not None and python_version >= 626:
        exception_value = makeExpressionMakeTuple(elements=(exception_value,), source_ref=source_ref)
    raise_statement = StatementRaiseException(exception_type=ExpressionBuiltinExceptionRef(exception_name='AssertionError', source_ref=source_ref), exception_value=exception_value, exception_trace=None, exception_cause=None, source_ref=source_ref)
    return makeStatementConditional(condition=ExpressionOperationNot(operand=buildNode(provider, node.test, source_ref), source_ref=source_ref), yes_branch=raise_statement, no_branch=None, source_ref=source_ref)