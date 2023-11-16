""" Reformulation of with statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka import Options
from nuitka.nodes.AttributeLookupNodes import ExpressionAttributeLookupSpecial
from nuitka.nodes.AttributeNodes import makeExpressionAttributeLookup
from nuitka.nodes.CallNodes import ExpressionCallEmpty, ExpressionCallNoKeywords
from nuitka.nodes.ComparisonNodes import ExpressionComparisonIs
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.ContainerMakingNodes import makeExpressionMakeTuple
from nuitka.nodes.CoroutineNodes import ExpressionAsyncWaitEnter, ExpressionAsyncWaitExit
from nuitka.nodes.ExceptionNodes import ExpressionCaughtExceptionTracebackRef, ExpressionCaughtExceptionTypeRef, ExpressionCaughtExceptionValueRef
from nuitka.nodes.StatementNodes import StatementExpressionOnly, StatementsSequence
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from nuitka.nodes.YieldNodes import ExpressionYieldFromAwaitable
from nuitka.PythonVersions import python_version
from .ReformulationAssignmentStatements import buildAssignmentStatements
from .ReformulationTryExceptStatements import makeTryExceptSingleHandlerNodeWithPublish
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, buildStatementsNode, makeReraiseExceptionStatement, makeStatementsSequence

def _buildWithNode(provider, context_expr, assign_target, body, sync, source_ref):
    if False:
        i = 10
        return i + 15
    with_source = buildNode(provider, context_expr, source_ref)
    if python_version < 896 and Options.is_full_compat:
        source_ref = with_source.getCompatibleSourceReference()
    temp_scope = provider.allocateTempScope('with')
    tmp_source_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='source', temp_type='object')
    tmp_exit_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='exit', temp_type='object')
    tmp_enter_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='enter', temp_type='object')
    tmp_indicator_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='indicator', temp_type='bool')
    statements = (buildAssignmentStatements(provider=provider, node=assign_target, allow_none=True, source=ExpressionTempVariableRef(variable=tmp_enter_variable, source_ref=source_ref), source_ref=source_ref), body)
    with_body = makeStatementsSequence(statements=statements, allow_none=True, source_ref=source_ref)
    if body and python_version < 928:
        deepest = body
        while deepest.getVisitableNodes():
            deepest = deepest.getVisitableNodes()[-1]
        if python_version < 880:
            body_lineno = deepest.getCompatibleSourceReference().getLineNumber()
        else:
            body_lineno = deepest.getSourceReference().getLineNumber()
        with_exit_source_ref = source_ref.atLineNumber(body_lineno)
    else:
        with_exit_source_ref = source_ref
    if python_version < 624:
        attribute_lookup_maker = makeExpressionAttributeLookup
    else:
        attribute_lookup_maker = ExpressionAttributeLookupSpecial
    enter_value = ExpressionCallEmpty(called=attribute_lookup_maker(expression=ExpressionTempVariableRef(variable=tmp_source_variable, source_ref=source_ref), attribute_name='__enter__' if sync else '__aenter__', source_ref=source_ref), source_ref=source_ref)
    exit_value_exception = ExpressionCallNoKeywords(called=ExpressionTempVariableRef(variable=tmp_exit_variable, source_ref=with_exit_source_ref), args=makeExpressionMakeTuple(elements=(ExpressionCaughtExceptionTypeRef(source_ref=with_exit_source_ref), ExpressionCaughtExceptionValueRef(source_ref=with_exit_source_ref), ExpressionCaughtExceptionTracebackRef(source_ref=source_ref)), source_ref=source_ref), source_ref=with_exit_source_ref)
    exit_value_no_exception = ExpressionCallNoKeywords(called=ExpressionTempVariableRef(variable=tmp_exit_variable, source_ref=source_ref), args=makeConstantRefNode(constant=(None, None, None), source_ref=source_ref), source_ref=with_exit_source_ref)
    if not sync:
        exit_value_exception = ExpressionYieldFromAwaitable(expression=ExpressionAsyncWaitExit(expression=exit_value_exception, source_ref=source_ref), source_ref=source_ref)
        exit_value_no_exception = ExpressionYieldFromAwaitable(ExpressionAsyncWaitExit(expression=exit_value_no_exception, source_ref=source_ref), source_ref=source_ref)
    statements = [makeStatementAssignmentVariable(variable=tmp_source_variable, source=with_source, source_ref=source_ref)]
    if not sync and python_version < 912:
        enter_value = ExpressionYieldFromAwaitable(expression=ExpressionAsyncWaitEnter(expression=enter_value, source_ref=source_ref), source_ref=source_ref)
    attribute_enter_assignment = makeStatementAssignmentVariable(variable=tmp_enter_variable, source=enter_value, source_ref=source_ref)
    attribute_exit_assignment = makeStatementAssignmentVariable(variable=tmp_exit_variable, source=attribute_lookup_maker(expression=ExpressionTempVariableRef(variable=tmp_source_variable, source_ref=source_ref), attribute_name='__exit__' if sync else '__aexit__', source_ref=source_ref), source_ref=source_ref)
    if python_version >= 912 and (not sync):
        enter_await_statement = makeStatementAssignmentVariable(variable=tmp_enter_variable, source=ExpressionYieldFromAwaitable(expression=ExpressionAsyncWaitEnter(expression=ExpressionTempVariableRef(variable=tmp_enter_variable, source_ref=source_ref), source_ref=source_ref), source_ref=source_ref), source_ref=source_ref)
        attribute_assignments = (attribute_enter_assignment, attribute_exit_assignment, enter_await_statement)
    elif python_version >= 864 and sync:
        attribute_assignments = (attribute_enter_assignment, attribute_exit_assignment)
    else:
        attribute_assignments = (attribute_exit_assignment, attribute_enter_assignment)
    statements.extend(attribute_assignments)
    statements.append(makeStatementAssignmentVariable(variable=tmp_indicator_variable, source=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref))
    statements += (makeTryFinallyStatement(provider=provider, tried=makeTryExceptSingleHandlerNodeWithPublish(provider=provider, tried=with_body, exception_name='BaseException', handler_body=StatementsSequence(statements=(makeStatementAssignmentVariable(variable=tmp_indicator_variable, source=makeConstantRefNode(constant=False, source_ref=source_ref), source_ref=source_ref), makeStatementConditional(condition=exit_value_exception, no_branch=makeReraiseExceptionStatement(source_ref=with_exit_source_ref), yes_branch=None, source_ref=with_exit_source_ref)), source_ref=source_ref), public_exc=python_version >= 624, source_ref=source_ref), final=makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=tmp_indicator_variable, source_ref=source_ref), right=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref), yes_branch=StatementExpressionOnly(expression=exit_value_no_exception, source_ref=source_ref), no_branch=None, source_ref=source_ref), source_ref=source_ref),)
    return makeTryFinallyStatement(provider=provider, tried=statements, final=(makeStatementReleaseVariable(variable=tmp_source_variable, source_ref=with_exit_source_ref), makeStatementReleaseVariable(variable=tmp_enter_variable, source_ref=with_exit_source_ref), makeStatementReleaseVariable(variable=tmp_exit_variable, source_ref=with_exit_source_ref)), source_ref=source_ref)

def buildWithNode(provider, node, source_ref):
    if False:
        while True:
            i = 10
    if hasattr(node, 'items'):
        context_exprs = [item.context_expr for item in node.items]
        assign_targets = [item.optional_vars for item in node.items]
    else:
        context_exprs = [node.context_expr]
        assign_targets = [node.optional_vars]
    body = buildStatementsNode(provider, node.body, source_ref)
    assert context_exprs and len(context_exprs) == len(assign_targets)
    context_exprs.reverse()
    assign_targets.reverse()
    for (context_expr, assign_target) in zip(context_exprs, assign_targets):
        body = _buildWithNode(provider=provider, body=body, context_expr=context_expr, assign_target=assign_target, sync=True, source_ref=source_ref)
    return body

def buildAsyncWithNode(provider, node, source_ref):
    if False:
        return 10
    context_exprs = [item.context_expr for item in node.items]
    assign_targets = [item.optional_vars for item in node.items]
    body = buildStatementsNode(provider, node.body, source_ref)
    assert context_exprs and len(context_exprs) == len(assign_targets)
    context_exprs.reverse()
    assign_targets.reverse()
    for (context_expr, assign_target) in zip(context_exprs, assign_targets):
        body = _buildWithNode(provider=provider, body=body, context_expr=context_expr, assign_target=assign_target, sync=False, source_ref=source_ref)
    return body