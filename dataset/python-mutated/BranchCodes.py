""" Branch related codes.

"""
from .CodeHelpers import generateStatementSequenceCode
from .ConditionalCodes import generateConditionCode
from .Emission import withSubCollector
from .LabelCodes import getGotoCode, getLabelCode

def generateBranchCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    true_target = context.allocateLabel('branch_yes')
    false_target = context.allocateLabel('branch_no')
    end_target = context.allocateLabel('branch_end')
    old_true_target = context.getTrueBranchTarget()
    old_false_target = context.getFalseBranchTarget()
    context.setTrueBranchTarget(true_target)
    context.setFalseBranchTarget(false_target)
    with withSubCollector(emit, context) as condition_emit:
        generateConditionCode(condition=statement.subnode_condition, emit=condition_emit, context=context)
    context.setTrueBranchTarget(old_true_target)
    context.setFalseBranchTarget(old_false_target)
    getLabelCode(true_target, emit)
    generateStatementSequenceCode(statement_sequence=statement.subnode_yes_branch, emit=emit, context=context)
    if statement.subnode_no_branch is not None:
        getGotoCode(end_target, emit)
        getLabelCode(false_target, emit)
        generateStatementSequenceCode(statement_sequence=statement.subnode_no_branch, emit=emit, context=context)
        getLabelCode(end_target, emit)
    else:
        getLabelCode(false_target, emit)