""" Reformulation of "exec" statements

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.BuiltinRefNodes import ExpressionBuiltinExceptionRef
from nuitka.nodes.ComparisonNodes import ExpressionComparisonIs
from nuitka.nodes.ConditionalNodes import ExpressionConditional, makeStatementConditional
from nuitka.nodes.ConstantRefNodes import ExpressionConstantNoneRef, makeConstantRefNode
from nuitka.nodes.ExceptionNodes import StatementRaiseException
from nuitka.nodes.ExecEvalNodes import StatementExec, StatementLocalsDictSync
from nuitka.nodes.GlobalsLocalsNodes import ExpressionBuiltinGlobals
from nuitka.nodes.NodeMakingHelpers import makeExpressionBuiltinLocals
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementsReleaseVariables
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, getKind, makeStatementsSequence, makeStatementsSequenceFromStatement, makeStatementsSequenceFromStatements

def wrapEvalGlobalsAndLocals(provider, globals_node, locals_node, temp_scope, source_ref):
    if False:
        return 10
    'Wrap the locals and globals arguments for "eval".\n\n    This is called from the outside, and when the node tree\n    already exists.\n    '
    locals_scope = provider.getLocalsScope()
    globals_keeper_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='globals', temp_type='object')
    locals_keeper_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='locals', temp_type='object')
    if locals_node is None:
        locals_node = ExpressionConstantNoneRef(source_ref=source_ref)
    if globals_node is None:
        globals_node = ExpressionConstantNoneRef(source_ref=source_ref)
    post_statements = []
    if provider.isExpressionClassBodyBase():
        post_statements.append(StatementLocalsDictSync(locals_scope=locals_scope, locals_arg=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), source_ref=source_ref.atInternal()))
    post_statements += makeStatementsReleaseVariables(variables=(globals_keeper_variable, locals_keeper_variable), source_ref=source_ref)
    locals_default = ExpressionConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), expression_no=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), expression_yes=makeExpressionBuiltinLocals(locals_scope=locals_scope, source_ref=source_ref), source_ref=source_ref)
    pre_statements = [makeStatementAssignmentVariable(variable=globals_keeper_variable, source=globals_node, source_ref=source_ref), makeStatementAssignmentVariable(variable=locals_keeper_variable, source=locals_node, source_ref=source_ref), makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=makeStatementAssignmentVariable(variable=locals_keeper_variable, source=locals_default, source_ref=source_ref), no_branch=None, source_ref=source_ref), makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=makeStatementAssignmentVariable(variable=globals_keeper_variable, source=ExpressionBuiltinGlobals(source_ref=source_ref), source_ref=source_ref), no_branch=None, source_ref=source_ref)]
    return (ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref if globals_node is None else globals_node.getSourceReference()), ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref if locals_node is None else locals_node.getSourceReference()), makeStatementsSequence(pre_statements, False, source_ref), makeStatementsSequence(post_statements, False, source_ref))

def buildExecNode(provider, node, source_ref):
    if False:
        return 10
    exec_globals = node.globals
    exec_locals = node.locals
    body = node.body
    if exec_locals is None and exec_globals is None and (getKind(body) == 'Tuple'):
        parts = body.elts
        body = parts[0]
        if len(parts) > 1:
            exec_globals = parts[1]
            if len(parts) > 2:
                exec_locals = parts[2]
        else:
            return StatementRaiseException(exception_type=ExpressionBuiltinExceptionRef(exception_name='TypeError', source_ref=source_ref), exception_value=makeConstantRefNode(constant='exec: arg 1 must be a string, file, or code object', source_ref=source_ref), exception_trace=None, exception_cause=None, source_ref=source_ref)
    temp_scope = provider.allocateTempScope('exec')
    locals_value = buildNode(provider, exec_locals, source_ref, True)
    if locals_value is None:
        locals_value = ExpressionConstantNoneRef(source_ref=source_ref)
    globals_value = buildNode(provider, exec_globals, source_ref, True)
    if globals_value is None:
        globals_value = ExpressionConstantNoneRef(source_ref=source_ref)
    source_code = buildNode(provider, body, source_ref)
    source_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='exec_source', temp_type='object')
    globals_keeper_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='globals', temp_type='object')
    locals_keeper_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='locals', temp_type='object')
    plain_indicator_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='plain', temp_type='bool')
    tried = (makeStatementAssignmentVariable(variable=source_variable, source=source_code, source_ref=source_ref), makeStatementAssignmentVariable(variable=globals_keeper_variable, source=globals_value, source_ref=source_ref), makeStatementAssignmentVariable(variable=locals_keeper_variable, source=locals_value, source_ref=source_ref), makeStatementAssignmentVariable(variable=plain_indicator_variable, source=makeConstantRefNode(constant=False, source_ref=source_ref), source_ref=source_ref), makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=makeStatementsSequenceFromStatements(makeStatementAssignmentVariable(variable=globals_keeper_variable, source=ExpressionBuiltinGlobals(source_ref=source_ref), source_ref=source_ref), makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=makeStatementsSequenceFromStatements(makeStatementAssignmentVariable(variable=locals_keeper_variable, source=makeExpressionBuiltinLocals(locals_scope=provider.getLocalsScope(), source_ref=source_ref), source_ref=source_ref), makeStatementAssignmentVariable(variable=plain_indicator_variable, source=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref)), no_branch=None, source_ref=source_ref)), no_branch=makeStatementsSequenceFromStatements(makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=makeStatementsSequenceFromStatement(statement=makeStatementAssignmentVariable(variable=locals_keeper_variable, source=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), source_ref=source_ref)), no_branch=None, source_ref=source_ref)), source_ref=source_ref), makeTryFinallyStatement(provider=provider, tried=StatementExec(source_code=ExpressionTempVariableRef(variable=source_variable, source_ref=source_ref), globals_arg=ExpressionTempVariableRef(variable=globals_keeper_variable, source_ref=source_ref), locals_arg=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), source_ref=source_ref), final=makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=plain_indicator_variable, source_ref=source_ref), right=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref), yes_branch=StatementLocalsDictSync(locals_scope=provider.getLocalsScope(), locals_arg=ExpressionTempVariableRef(variable=locals_keeper_variable, source_ref=source_ref), source_ref=source_ref), no_branch=None, source_ref=source_ref), source_ref=source_ref))
    return makeTryFinallyStatement(provider=provider, tried=tried, final=makeStatementsReleaseVariables(variables=(source_variable, globals_keeper_variable, locals_keeper_variable, plain_indicator_variable), source_ref=source_ref), source_ref=source_ref)
import nuitka.optimizations.OptimizeBuiltinCalls