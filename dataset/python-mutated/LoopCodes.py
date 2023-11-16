""" Loop codes.

Code generation for loops, breaking them, or continuing them. In Nuitka, there
are no for-loops or while-loops at this point. They have been re-formulated in
a simpler loop without a condition, and statements there-in that break under
certain conditions.

See Developer Manual for how the CPython loops are mapped to these nodes.
"""
from .CodeHelpers import generateStatementSequenceCode
from .ErrorCodes import getErrorExitBoolCode
from .ExceptionCodes import getExceptionUnpublishedReleaseCode
from .LabelCodes import getGotoCode, getLabelCode

def generateLoopBreakCode(statement, emit, context):
    if False:
        for i in range(10):
            print('nop')
    getExceptionUnpublishedReleaseCode(emit, context)
    break_target = context.getLoopBreakTarget()
    getGotoCode(break_target, emit)

def generateLoopContinueCode(statement, emit, context):
    if False:
        while True:
            i = 10
    getExceptionUnpublishedReleaseCode(emit, context)
    continue_target = context.getLoopContinueTarget()
    getGotoCode(continue_target, emit)

def generateLoopCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    loop_start_label = context.allocateLabel('loop_start')
    if not statement.isStatementAborting():
        loop_end_label = context.allocateLabel('loop_end')
    else:
        loop_end_label = None
    getLabelCode(loop_start_label, emit)
    old_loop_break = context.setLoopBreakTarget(loop_end_label)
    old_loop_continue = context.setLoopContinueTarget(loop_start_label)
    generateStatementSequenceCode(statement_sequence=statement.subnode_loop_body, allow_none=True, emit=emit, context=context)
    context.setLoopBreakTarget(old_loop_break)
    context.setLoopContinueTarget(old_loop_continue)
    with context.withCurrentSourceCodeReference(statement.getSourceReference()):
        getErrorExitBoolCode(condition='CONSIDER_THREADING(tstate) == false', emit=emit, context=context)
    getGotoCode(loop_start_label, emit)
    if loop_end_label is not None:
        getLabelCode(loop_end_label, emit)