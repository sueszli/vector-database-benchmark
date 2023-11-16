""" Reformulation of Python3.10 match statements.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
import ast
from nuitka.nodes.AttributeNodes import ExpressionAttributeCheck, makeExpressionAttributeLookup
from nuitka.nodes.BuiltinLenNodes import ExpressionBuiltinLen
from nuitka.nodes.BuiltinTypeNodes import ExpressionBuiltinList
from nuitka.nodes.ComparisonNodes import makeComparisonExpression
from nuitka.nodes.ConditionalNodes import makeStatementConditional
from nuitka.nodes.ConstantRefNodes import makeConstantRefNode
from nuitka.nodes.MatchNodes import ExpressionMatchArgs
from nuitka.nodes.OutlineNodes import ExpressionOutlineBody
from nuitka.nodes.ReturnNodes import makeStatementReturnConstant
from nuitka.nodes.SubscriptNodes import ExpressionSubscriptCheck, ExpressionSubscriptLookup
from nuitka.nodes.TypeMatchNodes import ExpressionMatchTypeCheckMapping, ExpressionMatchTypeCheckSequence
from nuitka.nodes.TypeNodes import ExpressionBuiltinIsinstance
from nuitka.nodes.VariableAssignNodes import makeStatementAssignmentVariable
from nuitka.nodes.VariableNameNodes import StatementAssignmentVariableName
from nuitka.nodes.VariableRefNodes import ExpressionTempVariableRef
from nuitka.nodes.VariableReleaseNodes import makeStatementReleaseVariable
from .ReformulationBooleanExpressions import makeAndNode, makeOrNode
from .ReformulationTryFinallyStatements import makeTryFinallyStatement
from .TreeHelpers import buildNode, buildStatementsNode, makeStatementsSequence, makeStatementsSequenceFromStatement

def _makeMatchComparison(left, right, source_ref):
    if False:
        for i in range(10):
            print('nop')
    if right.isExpressionConstantBoolRef() or right.isExpressionConstantNoneRef():
        comparator = 'Is'
    else:
        comparator = 'Eq'
    return makeComparisonExpression(left=left, right=right, comparator=comparator, source_ref=source_ref)

def _buildCaseBodyCode(provider, case, source_ref):
    if False:
        i = 10
        return i + 15
    guard_condition = buildNode(provider=provider, node=case.guard, source_ref=source_ref, allow_none=True)
    body_code = buildStatementsNode(provider, case.body, source_ref)
    return (body_code, guard_condition)

def _buildMatchAs(provider, pattern, make_against, source_ref):
    if False:
        return 10
    variable_name = pattern.name
    if variable_name is None:
        assignments = None
        conditions = None
    else:
        assert '.' not in variable_name, variable_name
        assert '!' not in variable_name, variable_name
        if pattern.pattern is not None:
            (conditions, assignments) = _buildMatch(provider=provider, pattern=pattern.pattern, make_against=make_against, source_ref=source_ref)
            if assignments is None:
                assignments = []
        else:
            assignments = []
            conditions = None
        assignments.append(StatementAssignmentVariableName(provider=provider, variable_name=variable_name, source=make_against(), source_ref=source_ref))
    return (conditions, assignments)

def _buildMatchValue(provider, make_against, pattern, source_ref):
    if False:
        i = 10
        return i + 15
    if type(pattern) is ast.MatchValue:
        right = buildNode(provider, pattern.value, source_ref)
    else:
        right = makeConstantRefNode(constant=pattern.value, source_ref=source_ref)
    return _makeMatchComparison(left=make_against(), right=right, source_ref=source_ref)

def _buildMatchSequence(provider, pattern, make_against, source_ref):
    if False:
        while True:
            i = 10
    conditions = [ExpressionMatchTypeCheckSequence(value=make_against(), source_ref=source_ref)]
    assignments = []
    min_length = len(tuple((seq_pattern for seq_pattern in pattern.patterns if seq_pattern.__class__ is not ast.MatchStar)))
    exact = all((seq_pattern.__class__ is not ast.MatchStar for seq_pattern in pattern.patterns))
    conditions.append(makeComparisonExpression(left=ExpressionBuiltinLen(value=make_against(), source_ref=source_ref), right=makeConstantRefNode(constant=min_length, source_ref=source_ref), comparator='Eq' if exact else 'GtE', source_ref=source_ref))
    star_pos = None
    count = seq_pattern = None
    for (count, seq_pattern) in enumerate(pattern.patterns):
        if star_pos is None:
            offset = count
        else:
            offset = -(len(pattern.patterns) - count)
        if seq_pattern.__class__ is ast.MatchStar:
            star_pos = count
            variable_name = seq_pattern.name
            if variable_name is not None:
                assert '.' not in variable_name, variable_name
                assert '!' not in variable_name, variable_name
                if star_pos == len(pattern.patterns):
                    slice_value = slice(count)
                else:
                    slice_value = slice(count, -(len(pattern.patterns) - (count + 1)))
                assignments.append(StatementAssignmentVariableName(provider=provider, variable_name=variable_name, source=ExpressionBuiltinList(ExpressionSubscriptLookup(expression=make_against(), subscript=makeConstantRefNode(constant=slice_value, source_ref=source_ref), source_ref=source_ref), source_ref=source_ref), source_ref=source_ref))
        else:
            (item_conditions, item_assignments) = _buildMatch(provider=provider, pattern=seq_pattern, make_against=lambda : ExpressionSubscriptLookup(expression=make_against(), subscript=makeConstantRefNode(constant=offset, source_ref=source_ref), source_ref=source_ref), source_ref=source_ref)
            if item_conditions:
                conditions.extend(item_conditions)
            if item_assignments:
                assignments.extend(item_assignments)
    return (conditions, assignments)

def _buildMatchMapping(provider, pattern, make_against, source_ref):
    if False:
        while True:
            i = 10
    conditions = [ExpressionMatchTypeCheckMapping(value=make_against(), source_ref=source_ref)]
    assignments = []
    assert len(pattern.keys) == len(pattern.patterns), ast.dump(pattern)
    key = kwd_pattern = None
    for (key, kwd_pattern) in zip(pattern.keys, pattern.patterns):
        conditions.append(ExpressionSubscriptCheck(expression=make_against(), subscript=buildNode(provider, key, source_ref), source_ref=source_ref))
        (item_conditions, item_assignments) = _buildMatch(provider=provider, make_against=lambda : ExpressionSubscriptLookup(expression=make_against(), subscript=buildNode(provider, key, source_ref), source_ref=source_ref), pattern=kwd_pattern, source_ref=source_ref)
        if item_conditions:
            conditions.extend(item_conditions)
        if item_assignments:
            assignments.extend(item_assignments)
    return (conditions, assignments)

def _buildMatchClass(provider, pattern, make_against, source_ref):
    if False:
        for i in range(10):
            print('nop')
    cls_node = buildNode(provider, pattern.cls, source_ref)
    assignments = []
    conditions = [ExpressionBuiltinIsinstance(instance=make_against(), classes=cls_node, source_ref=source_ref)]
    assert not (pattern.patterns and pattern.kwd_patterns), (source_ref, ast.dump(pattern))
    assert len(pattern.kwd_attrs) == len(pattern.kwd_patterns), (source_ref, ast.dump(pattern))
    for (count, pos_pattern) in enumerate(pattern.patterns):
        (item_conditions, item_assignments) = _buildMatch(provider=provider, make_against=lambda : ExpressionSubscriptLookup(expression=ExpressionMatchArgs(expression=make_against(), max_allowed=len(pattern.patterns), source_ref=source_ref), subscript=makeConstantRefNode(constant=count, source_ref=source_ref), source_ref=source_ref), pattern=pos_pattern, source_ref=source_ref)
        if item_conditions:
            conditions.extend(item_conditions)
        if item_assignments:
            assignments.extend(item_assignments)
    for (key, kwd_pattern) in zip(pattern.kwd_attrs, pattern.kwd_patterns):
        conditions.append(ExpressionAttributeCheck(expression=make_against(), attribute_name=key, source_ref=source_ref))
        (item_conditions, item_assignments) = _buildMatch(provider=provider, make_against=lambda : makeExpressionAttributeLookup(expression=make_against(), attribute_name=key, source_ref=source_ref), pattern=kwd_pattern, source_ref=source_ref)
        if item_conditions:
            conditions.extend(item_conditions)
        if item_assignments:
            assignments.extend(item_assignments)
    return (conditions, assignments)

def _buildMatchOr(provider, pattern, make_against, source_ref):
    if False:
        return 10
    or_condition_list = []
    or_assignments_list = []
    for or_pattern in pattern.patterns:
        (or_conditions, or_assignments) = _buildMatch(provider=provider, pattern=or_pattern, make_against=make_against, source_ref=source_ref)
        if or_conditions:
            or_condition_list.append(makeAndNode(values=or_conditions, source_ref=source_ref))
        else:
            or_condition_list.append(makeConstantRefNode(constant=True, source_ref=source_ref))
        or_assignments_list.append(or_assignments or None)
    if all((or_assignments is None for or_assignments in or_assignments_list)):
        condition = makeOrNode(values=or_condition_list, source_ref=source_ref)
        conditions = (condition,)
        assignments = None
    else:
        body = None
        for (or_condition, or_assignments) in zip(reversed(or_condition_list), reversed(or_assignments_list)):
            assert or_assignments is not None, (source_ref, or_assignments_list)
            statements = list(or_assignments)
            statements.append(makeStatementReturnConstant(constant=True, source_ref=source_ref))
            if body is None:
                body = makeStatementConditional(condition=or_condition, yes_branch=makeStatementsSequence(statements=statements, allow_none=False, source_ref=source_ref), no_branch=makeStatementReturnConstant(constant=False, source_ref=source_ref), source_ref=source_ref)
            else:
                body = makeStatementConditional(condition=or_condition, yes_branch=makeStatementsSequence(statements=statements, allow_none=False, source_ref=source_ref), no_branch=body, source_ref=source_ref)
        body = makeStatementsSequenceFromStatement(body)
        outline_body = ExpressionOutlineBody(provider=provider, name='match_or', source_ref=source_ref)
        outline_body.setChildBody(body)
        conditions = (outline_body,)
        assignments = None
    return (conditions, assignments)

def _buildMatch(provider, pattern, make_against, source_ref):
    if False:
        i = 10
        return i + 15
    if pattern.__class__ is ast.MatchOr:
        (conditions, assignments) = _buildMatchOr(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    elif pattern.__class__ is ast.MatchClass:
        (conditions, assignments) = _buildMatchClass(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    elif pattern.__class__ is ast.MatchMapping:
        (conditions, assignments) = _buildMatchMapping(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    elif pattern.__class__ is ast.MatchSequence:
        (conditions, assignments) = _buildMatchSequence(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    elif pattern.__class__ is ast.MatchAs:
        (conditions, assignments) = _buildMatchAs(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    elif pattern.__class__ is ast.MatchValue or pattern.__class__ is ast.MatchSingleton:
        conditions = [_buildMatchValue(provider=provider, make_against=make_against, pattern=pattern, source_ref=source_ref)]
        assignments = None
    else:
        assert False, ast.dump(pattern)
    return (conditions, assignments)

def _buildCase(provider, case, tmp_subject, source_ref):
    if False:
        return 10
    assert case.__class__ is ast.match_case, case
    pattern = case.pattern
    make_against = lambda : ExpressionTempVariableRef(variable=tmp_subject, source_ref=source_ref)
    (conditions, assignments) = _buildMatch(provider=provider, pattern=pattern, make_against=make_against, source_ref=source_ref)
    (branch_code, guard) = _buildCaseBodyCode(provider, case, source_ref)
    return (conditions, assignments, guard, branch_code)

def buildMatchNode(provider, node, source_ref):
    if False:
        i = 10
        return i + 15
    'Python3.10 or higher, match statements.'
    subject_node = buildNode(provider, node.subject, source_ref)
    temp_scope = provider.allocateTempScope('match_statement')
    tmp_subject = provider.allocateTempVariable(temp_scope, 'subject', temp_type='object')
    tmp_indicator_variable = provider.allocateTempVariable(temp_scope=temp_scope, name='indicator', temp_type='bool')
    cases = []
    for case in node.cases:
        cases.append(_buildCase(provider=provider, case=case, tmp_subject=tmp_subject, source_ref=source_ref))
    case_statements = []
    for case in cases:
        (conditions, assignments, guard, branch_code) = case
        if case is not cases[-1]:
            branch_code = makeStatementsSequence(statements=(branch_code, makeStatementAssignmentVariable(variable=tmp_indicator_variable, source=makeConstantRefNode(constant=True, source_ref=source_ref), source_ref=source_ref)), allow_none=True, source_ref=source_ref)
        if guard is not None:
            branch_code = makeStatementConditional(condition=guard, yes_branch=branch_code, no_branch=None, source_ref=source_ref)
        del guard
        branch_code = makeStatementsSequence(statements=(assignments, branch_code), allow_none=True, source_ref=source_ref)
        del assignments
        if conditions is not None:
            branch_code = makeStatementConditional(condition=makeAndNode(values=conditions, source_ref=source_ref), yes_branch=branch_code, no_branch=None, source_ref=source_ref)
        del conditions
        if case is not cases[0]:
            statement = makeStatementConditional(condition=makeComparisonExpression(comparator='Is', left=ExpressionTempVariableRef(variable=tmp_indicator_variable, source_ref=source_ref), right=makeConstantRefNode(constant=False, source_ref=source_ref), source_ref=source_ref), yes_branch=branch_code, no_branch=None, source_ref=source_ref)
        else:
            statement = branch_code
        case_statements.append(statement)
    return makeStatementsSequence(statements=(makeStatementAssignmentVariable(variable=tmp_indicator_variable, source=makeConstantRefNode(constant=False, source_ref=source_ref), source_ref=source_ref), makeStatementAssignmentVariable(variable=tmp_subject, source=subject_node, source_ref=subject_node.getSourceReference()), makeTryFinallyStatement(provider=provider, tried=case_statements, final=makeStatementReleaseVariable(variable=tmp_indicator_variable, source_ref=source_ref), source_ref=source_ref)), allow_none=False, source_ref=source_ref)