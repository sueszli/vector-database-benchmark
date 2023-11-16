""" Reformulation of comparison chain expressions.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.ComparisonNodes import makeComparisonExpression
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.OperatorNodesUnary import ExpressionOperationNot
from nuitka.nodes.OutlineNodes import ExpressionOutlineBody
from nuitka.nodes.ReturnNodes import StatementReturn
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, getKind, makeStatementsSequenceFromStatement

def _makeComparisonNode(left, right, comparator, source_ref):
    if False:
        return 10
    result = makeComparisonExpression(left, right, comparator, source_ref)
    result.setCompatibleSourceReference(source_ref=right.getCompatibleSourceReference())
    return result

def buildComparisonNode(provider, node, source_ref):
    if False:
        for i in range(10):
            print('nop')
    assert len(node.comparators) == len(node.ops)
    left = buildNode(provider, node.left, source_ref)
    rights = [buildNode(provider, comparator, source_ref) for comparator in node.comparators]
    comparators = [getKind(comparator) for comparator in node.ops]
    if len(rights) == 1:
        return _makeComparisonNode(left=left, right=rights[0], comparator=comparators[0], source_ref=source_ref)
    return buildComplexComparisonNode(provider, left, rights, comparators, source_ref)

def buildComplexComparisonNode(provider, left, rights, comparators, source_ref):
    if False:
        while True:
            i = 10
    outline_body = ExpressionOutlineBody(provider=provider, name='comparison_chain', source_ref=source_ref)
    variables = [outline_body.allocateTempVariable(temp_scope=None, name='operand_%d' % count, temp_type='object') for count in range(2, len(rights) + 2)]
    tmp_variable = outline_body.allocateTempVariable(temp_scope=None, name='comparison_result', temp_type='object')

    def makeTempAssignment(count, value):
        if False:
            i = 10
            return i + 15
        return makeStatementAssignmentVariable(variable=variables[count], source=value, source_ref=source_ref)

    def makeReleaseStatement(count):
        if False:
            return 10
        return makeStatementReleaseVariable(variable=variables[count], source_ref=source_ref)

    def makeValueComparisonReturn(left, right, comparator):
        if False:
            while True:
                i = 10
        yield makeStatementAssignmentVariable(variable=tmp_variable, source=_makeComparisonNode(left=left, right=right, comparator=comparator, source_ref=source_ref), source_ref=source_ref)
        yield makeStatementConditional(condition=ExpressionOperationNot(operand=ExpressionTempVariableRef(variable=tmp_variable, source_ref=source_ref), source_ref=source_ref), yes_branch=StatementReturn(expression=ExpressionTempVariableRef(variable=tmp_variable, source_ref=source_ref), source_ref=source_ref), no_branch=None, source_ref=source_ref)
    statements = []
    final = []
    for (count, value) in enumerate(rights):
        if value is not rights[-1]:
            statements.append(makeTempAssignment(count, value))
            final.append(makeReleaseStatement(count))
            right = ExpressionTempVariableRef(variable=variables[count], source_ref=source_ref)
        else:
            right = value
        if count != 0:
            left = ExpressionTempVariableRef(variable=variables[count - 1], source_ref=source_ref)
        comparator = comparators[count]
        if value is not rights[-1]:
            statements.extend(makeValueComparisonReturn(left, right, comparator))
        else:
            statements.append(StatementReturn(expression=_makeComparisonNode(left=left, right=right, comparator=comparator, source_ref=source_ref), source_ref=source_ref))
            final.append(makeStatementReleaseVariable(variable=tmp_variable, source_ref=source_ref))
    outline_body.setChildBody(makeStatementsSequenceFromStatement(statement=makeTryFinallyStatement(provider=outline_body, tried=statements, final=final, source_ref=source_ref)))
    return outline_body