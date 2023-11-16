""" Reformulation of print statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.ComparisonNodes import ExpressionComparisonIs
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.ConstantRefNodes import ExpressionConstantNoneRef
from nuitka.nodes.ImportNodes import makeExpressionImportModuleNameHard
from nuitka.nodes.PrintNodes import StatementPrintNewline, StatementPrintValue
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, buildNodeTuple, makeStatementsSequenceFromStatements

def buildPrintNode(provider, node, source_ref):
    if False:
        i = 10
        return i + 15
    if node.dest is not None:
        temp_scope = provider.allocateTempScope('print')
        tmp_target_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='target', temp_type='object')
        target_default_statement = makeStatementAssignmentVariable(variable=tmp_target_variable, source=makeExpressionImportModuleNameHard(module_name='sys', import_name='stdout', module_guaranteed=True, source_ref=source_ref), source_ref=source_ref)
        statements = [makeStatementAssignmentVariable(variable=tmp_target_variable, source=buildNode(provider=provider, node=node.dest, source_ref=source_ref), source_ref=source_ref), makeStatementConditional(condition=ExpressionComparisonIs(left=ExpressionTempVariableRef(variable=tmp_target_variable, source_ref=source_ref), right=ExpressionConstantNoneRef(source_ref=source_ref), source_ref=source_ref), yes_branch=target_default_statement, no_branch=None, source_ref=source_ref)]
    values = buildNodeTuple(provider=provider, nodes=node.values, source_ref=source_ref)
    if node.dest is not None:
        print_statements = [StatementPrintValue(dest=ExpressionTempVariableRef(variable=tmp_target_variable, source_ref=source_ref), value=value, source_ref=source_ref) for value in values]
        if node.nl:
            print_statements.append(StatementPrintNewline(dest=ExpressionTempVariableRef(variable=tmp_target_variable, source_ref=source_ref), source_ref=source_ref))
        statements.append(makeTryFinallyStatement(provider=provider, tried=print_statements, final=makeStatementReleaseVariable(variable=tmp_target_variable, source_ref=source_ref), source_ref=source_ref))
    else:
        statements = [StatementPrintValue(dest=None, value=value, source_ref=source_ref) for value in values]
        if node.nl:
            statements.append(StatementPrintNewline(dest=None, source_ref=source_ref))
    return makeStatementsSequenceFromStatements(*statements)