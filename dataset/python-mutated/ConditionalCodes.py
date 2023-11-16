""" Conditional statements related codes.

Branches, conditions, truth checks.
"""
from .CodeHelpers import decideConversionCheckNeeded, generateExpressionCode
from .Emission import SourceCodeCollector
from .ErrorCodes import getErrorExitBoolCode, getReleaseCode, getTakeReferenceCode
from .LabelCodes import getBranchingCode, getGotoCode, getLabelCode

def generateConditionCode(condition, emit, context):
    if False:
        print('Hello World!')
    if condition.mayRaiseExceptionBool(BaseException):
        compare_name = context.allocateTempName('condition_result', 'nuitka_bool')
    else:
        compare_name = context.allocateTempName('condition_result', 'bool')
    generateExpressionCode(to_name=compare_name, expression=condition, emit=emit, context=context)
    getBranchingCode(condition=compare_name.getCType().getTruthCheckCode(compare_name), emit=emit, context=context)
    getReleaseCode(compare_name, emit, context)

def generateConditionalAndOrCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    if expression.isExpressionConditionalOr():
        prefix = 'or_'
    else:
        prefix = 'and_'
    true_target = context.allocateLabel(prefix + 'left')
    false_target = context.allocateLabel(prefix + 'right')
    end_target = context.allocateLabel(prefix + 'end')
    old_true_target = context.getTrueBranchTarget()
    old_false_target = context.getFalseBranchTarget()
    truth_name = context.allocateTempName(prefix + 'left_truth', 'int')
    left_name = context.allocateTempName(prefix + 'left_value', to_name.c_type)
    right_name = context.allocateTempName(prefix + 'right_value', to_name.c_type)
    left_value = expression.subnode_left
    generateExpressionCode(to_name=left_name, expression=left_value, emit=emit, context=context)
    needs_ref1 = context.needsCleanup(left_name)
    if expression.isExpressionConditionalOr():
        context.setTrueBranchTarget(true_target)
        context.setFalseBranchTarget(false_target)
    else:
        context.setTrueBranchTarget(false_target)
        context.setFalseBranchTarget(true_target)
    left_name.getCType().emitTruthCheckCode(to_name=truth_name, value_name=left_name, emit=emit)
    needs_check = left_value.mayRaiseExceptionBool(BaseException)
    if needs_check:
        getErrorExitBoolCode(condition='%s == -1' % truth_name, needs_check=True, emit=emit, context=context)
    getBranchingCode(condition='%s == 1' % truth_name, emit=emit, context=context)
    getLabelCode(false_target, emit)
    getReleaseCode(release_name=left_name, emit=emit, context=context)
    right_value = expression.subnode_right
    generateExpressionCode(to_name=right_name, expression=right_value, emit=emit, context=context)
    needs_ref2 = context.needsCleanup(right_name)
    if needs_ref2:
        context.removeCleanupTempName(right_name)
    if not needs_ref2 and needs_ref1:
        getTakeReferenceCode(right_name, emit)
    to_name.getCType().emitAssignConversionCode(to_name=to_name, value_name=right_name, needs_check=decideConversionCheckNeeded(to_name, right_value), emit=emit, context=context)
    getGotoCode(end_target, emit)
    getLabelCode(true_target, emit)
    if not needs_ref1 and needs_ref2:
        getTakeReferenceCode(left_name, emit)
    to_name.getCType().emitAssignConversionCode(to_name=to_name, value_name=left_name, needs_check=decideConversionCheckNeeded(to_name, left_value), emit=emit, context=context)
    getLabelCode(end_target, emit)
    if needs_ref1 or needs_ref2:
        context.addCleanupTempName(to_name)
    context.setTrueBranchTarget(old_true_target)
    context.setFalseBranchTarget(old_false_target)

def generateConditionalCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    true_target = context.allocateLabel('condexpr_true')
    false_target = context.allocateLabel('condexpr_false')
    end_target = context.allocateLabel('condexpr_end')
    old_true_target = context.getTrueBranchTarget()
    old_false_target = context.getFalseBranchTarget()
    context.setTrueBranchTarget(true_target)
    context.setFalseBranchTarget(false_target)
    generateConditionCode(condition=expression.subnode_condition, emit=emit, context=context)
    getLabelCode(true_target, emit)
    generateExpressionCode(to_name=to_name, expression=expression.subnode_expression_yes, emit=emit, context=context)
    needs_ref1 = context.needsCleanup(to_name)
    if needs_ref1:
        context.removeCleanupTempName(to_name)
    real_emit = emit
    emit = SourceCodeCollector()
    generateExpressionCode(to_name=to_name, expression=expression.subnode_expression_no, emit=emit, context=context)
    needs_ref2 = context.needsCleanup(to_name)
    if needs_ref1 and (not needs_ref2):
        getGotoCode(end_target, real_emit)
        getLabelCode(false_target, real_emit)
        for line in emit.codes:
            real_emit(line)
        emit = real_emit
        getTakeReferenceCode(to_name, emit)
        context.addCleanupTempName(to_name)
    elif not needs_ref1 and needs_ref2:
        getTakeReferenceCode(to_name, real_emit)
        getGotoCode(end_target, real_emit)
        getLabelCode(false_target, real_emit)
        for line in emit.codes:
            real_emit(line)
        emit = real_emit
    else:
        getGotoCode(end_target, real_emit)
        getLabelCode(false_target, real_emit)
        for line in emit.codes:
            real_emit(line)
        emit = real_emit
    getLabelCode(end_target, emit)
    context.setTrueBranchTarget(old_true_target)
    context.setFalseBranchTarget(old_false_target)