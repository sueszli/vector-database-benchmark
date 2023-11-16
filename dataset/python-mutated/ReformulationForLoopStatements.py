""" Reformulation of for loop statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.BuiltinIteratorNodes import ExpressionAsyncIter, ExpressionAsyncNext, ExpressionBuiltinIter1
from nuitka.nodes.BuiltinNextNodes import ExpressionBuiltinNext1
from nuitka.nodes.ComparisonNodes import ExpressionComparisonIs
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.LoopNodes import StatementLoop, StatementLoopBreak
from nuitka.nodes.StatementNodes import StatementsSequence
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from nuitka.nodes.YieldNodes import ExpressionYieldFromAwaitable
from .ReformulationAssignmentStatements import buildAssignmentStatements
from .ReformulationTryExceptStatements import makeTryExceptSingleHandlerNode
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, buildStatementsNode, makeStatementsSequence, makeStatementsSequenceFromStatements, popBuildContext, pushBuildContext

def _buildForLoopNode(provider, node, sync, source_ref):
    if False:
        i = 10
        return i + 15
    source = buildNode(provider, node.iter, source_ref)
    temp_scope = provider.allocateTempScope('for_loop')
    tmp_iter_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='for_iterator', temp_type='object')
    tmp_value_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='iter_value', temp_type='object')
    else_block = buildStatementsNode(provider=provider, nodes=node.orelse if node.orelse else None, source_ref=source_ref)
    if else_block is not None:
        tmp_break_indicator = provider.allocateTempVariable(temp_scope=temp_scope, name='break_indicator', temp_type='bool')
        statements = [makeStatementAssignmentVariable(variable=tmp_break_indicator, source=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref)]
    else:
        statements = []
    statements.append(StatementLoopBreak(source_ref=source_ref))
    handler_body = makeStatementsSequence(statements=statements, allow_none=False, source_ref=source_ref)
    if sync:
        next_node = ExpressionBuiltinNext1(value=ExpressionTempVariableRef(variable=tmp_iter_variable, source_ref=source_ref), source_ref=source_ref)
    else:
        next_node = ExpressionYieldFromAwaitable(expression=ExpressionAsyncNext(value=ExpressionTempVariableRef(variable=tmp_iter_variable, source_ref=source_ref), source_ref=source_ref), source_ref=source_ref)
    statements = (makeTryExceptSingleHandlerNode(tried=makeStatementAssignmentVariable(variable=tmp_value_variable, source=next_node, source_ref=source_ref), exception_name='StopIteration' if sync else 'StopAsyncIteration', handler_body=handler_body, source_ref=source_ref), buildAssignmentStatements(provider=provider, node=node.target, source=ExpressionTempVariableRef(variable=tmp_value_variable, source_ref=source_ref), source_ref=source_ref))
    pushBuildContext('loop_body')
    statements += (buildStatementsNode(provider=provider, nodes=node.body, source_ref=source_ref),)
    popBuildContext()
    loop_body = makeStatementsSequence(statements=statements, allow_none=True, source_ref=source_ref)
    cleanup_statements = (makeStatementReleaseVariable(variable=tmp_value_variable, source_ref=source_ref), makeStatementReleaseVariable(variable=tmp_iter_variable, source_ref=source_ref))
    if else_block is not None:
        statements = [makeStatementAssignmentVariable(variable=tmp_break_indicator, source=makeConstantRefNode(constant=False, source_ref=source_ref), source_ref=source_ref)]
    else:
        statements = []
    if sync:
        iter_source = ExpressionBuiltinIter1(value=source, source_ref=source.getSourceReference())
    else:
        iter_source = ExpressionYieldFromAwaitable(expression=ExpressionAsyncIter(value=source, source_ref=source.getSourceReference()), source_ref=source.getSourceReference())
    statements += (makeStatementAssignmentVariable(variable=tmp_iter_variable, source=iter_source, source_ref=source_ref), makeTryFinallyStatement(provider=provider, tried=StatementLoop(loop_body=loop_body, source_ref=source_ref), final=StatementsSequence(statements=cleanup_statements, source_ref=source_ref), source_ref=source_ref))
    if else_block is not None:
        statements.append(makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=tmp_break_indicator, source_ref=source_ref), right=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref), yes_branch=else_block, no_branch=None, source_ref=source_ref))
    return makeStatementsSequenceFromStatements(*statements)

def buildForLoopNode(provider, node, source_ref):
    if False:
        while True:
            i = 10
    return _buildForLoopNode(provider, node, True, source_ref)

def buildAsyncForLoopNode(provider, node, source_ref):
    if False:
        i = 10
        return i + 15
    return _buildForLoopNode(provider, node, False, source_ref)