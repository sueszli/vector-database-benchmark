""" Code generation for implicit and explicit exception raises.

Exceptions from other operations are consider ErrorCodes domain.

"""
from nuitka import Options
from .CodeHelpers import generateChildExpressionsCode, generateExpressionCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getFrameVariableTypeDescriptionCode
from .LabelCodes import getGotoCode
from .LineNumberCodes import emitErrorLineNumberUpdateCode, getErrorLineNumberUpdateCode
from .PythonAPICodes import getReferenceExportCode

def generateReraiseCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    with context.withCurrentSourceCodeReference(value=statement.getCompatibleSourceReference()):
        getReRaiseExceptionCode(emit=emit, context=context)

def generateRaiseCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    exception_type = statement.subnode_exception_type
    exception_value = statement.subnode_exception_value
    exception_tb = statement.subnode_exception_trace
    exception_cause = statement.subnode_exception_cause
    if exception_cause is not None:
        assert exception_type is not None
        assert exception_value is None
        assert exception_tb is None
        raise_type_name = context.allocateTempName('raise_type')
        generateExpressionCode(to_name=raise_type_name, expression=exception_type, emit=emit, context=context)
        raise_cause_name = context.allocateTempName('raise_cause')
        generateExpressionCode(to_name=raise_cause_name, expression=exception_cause, emit=emit, context=context)
        with context.withCurrentSourceCodeReference(exception_cause.getSourceReference()):
            _getRaiseExceptionWithCauseCode(raise_type_name=raise_type_name, raise_cause_name=raise_cause_name, emit=emit, context=context)
    elif exception_type is None:
        assert False, statement
    elif exception_value is None and exception_tb is None:
        raise_type_name = context.allocateTempName('raise_type')
        generateExpressionCode(to_name=raise_type_name, expression=exception_type, emit=emit, context=context)
        with context.withCurrentSourceCodeReference(value=exception_type.getCompatibleSourceReference()):
            _getRaiseExceptionWithTypeCode(raise_type_name=raise_type_name, emit=emit, context=context)
    elif exception_tb is None:
        raise_type_name = context.allocateTempName('raise_type')
        generateExpressionCode(to_name=raise_type_name, expression=exception_type, emit=emit, context=context)
        raise_value_name = context.allocateTempName('raise_value')
        generateExpressionCode(to_name=raise_value_name, expression=exception_value, emit=emit, context=context)
        with context.withCurrentSourceCodeReference(exception_value.getCompatibleSourceReference()):
            _getRaiseExceptionWithValueCode(raise_type_name=raise_type_name, raise_value_name=raise_value_name, implicit=statement.isStatementRaiseExceptionImplicit(), emit=emit, context=context)
    else:
        raise_type_name = context.allocateTempName('raise_type')
        generateExpressionCode(to_name=raise_type_name, expression=exception_type, emit=emit, context=context)
        raise_value_name = context.allocateTempName('raise_value')
        generateExpressionCode(to_name=raise_value_name, expression=exception_value, emit=emit, context=context)
        raise_tb_name = context.allocateTempName('raise_tb')
        generateExpressionCode(to_name=raise_tb_name, expression=exception_tb, emit=emit, context=context)
        with context.withCurrentSourceCodeReference(exception_tb.getSourceReference()):
            _getRaiseExceptionWithTracebackCode(raise_type_name=raise_type_name, raise_value_name=raise_value_name, raise_tb_name=raise_tb_name, emit=emit, context=context)

def generateRaiseExpressionCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    arg_names = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    if Options.is_debug:
        parent = expression.parent
        assert parent.isExpressionSideEffects() or parent.isExpressionConditional() or parent.isExpressionConditionalOr() or parent.isExpressionConditionalAnd() or parent.isExpressionLocalsVariableRefOrFallback(), (expression, expression.parent, expression.asXmlText())
    with withObjectCodeTemporaryAssignment(to_name, 'raise_exception_result', expression, emit, context) as value_name:
        emit('%s = NULL;' % value_name)
        _getRaiseExceptionWithValueCode(raise_type_name=arg_names[0], raise_value_name=arg_names[1], implicit=True, emit=emit, context=context)

def getReRaiseExceptionCode(emit, context):
    if False:
        while True:
            i = 10
    (exception_type, exception_value, exception_tb, exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    keeper_variables = context.getExceptionKeeperVariables()
    if keeper_variables[0] is None:
        emit('%(bool_res_name)s = RERAISE_EXCEPTION(&%(exception_type)s, &%(exception_value)s, &%(exception_tb)s);\nif (unlikely(%(bool_res_name)s == false)) {\n    %(update_code)s\n}\n' % {'exception_type': exception_type, 'exception_value': exception_value, 'exception_tb': exception_tb, 'bool_res_name': context.getBoolResName(), 'update_code': getErrorLineNumberUpdateCode(context)})
        frame_handle = context.getFrameHandle()
        if frame_handle:
            emit('if (%(exception_tb)s && %(exception_tb)s->tb_frame == &%(frame_identifier)s->m_frame) %(frame_identifier)s->m_frame.f_lineno = %(exception_tb)s->tb_lineno;' % {'exception_tb': exception_tb, 'frame_identifier': context.getFrameHandle()})
            emit(getFrameVariableTypeDescriptionCode(context))
    else:
        (keeper_type, keeper_value, keeper_tb, keeper_lineno) = context.getExceptionKeeperVariables()
        emit('// Re-raise.\n%(exception_type)s = %(keeper_type)s;\n%(exception_value)s = %(keeper_value)s;\n%(exception_tb)s = %(keeper_tb)s;\n%(exception_lineno)s = %(keeper_lineno)s;\n' % {'exception_type': exception_type, 'exception_value': exception_value, 'exception_tb': exception_tb, 'exception_lineno': exception_lineno, 'keeper_type': keeper_type, 'keeper_value': keeper_value, 'keeper_tb': keeper_tb, 'keeper_lineno': keeper_lineno})
    getGotoCode(context.getExceptionEscape(), emit)

def _getRaiseExceptionWithCauseCode(raise_type_name, raise_cause_name, emit, context):
    if False:
        for i in range(10):
            print('nop')
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    emit('%s = %s;' % (exception_type, raise_type_name))
    getReferenceExportCode(raise_type_name, emit, context)
    emit('%s = NULL;' % exception_value)
    getReferenceExportCode(raise_cause_name, emit, context)
    emitErrorLineNumberUpdateCode(emit, context)
    emit('RAISE_EXCEPTION_WITH_CAUSE(tstate, &%s, &%s, &%s, %s);' % (exception_type, exception_value, exception_tb, raise_cause_name))
    emit(getFrameVariableTypeDescriptionCode(context))
    getGotoCode(context.getExceptionEscape(), emit)
    if context.needsCleanup(raise_type_name):
        context.removeCleanupTempName(raise_type_name)
    if context.needsCleanup(raise_cause_name):
        context.removeCleanupTempName(raise_cause_name)

def _getRaiseExceptionWithTypeCode(raise_type_name, emit, context):
    if False:
        while True:
            i = 10
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    emit('%s = %s;' % (exception_type, raise_type_name))
    getReferenceExportCode(raise_type_name, emit, context)
    emitErrorLineNumberUpdateCode(emit, context)
    emit('RAISE_EXCEPTION_WITH_TYPE(tstate, &%s, &%s, &%s);' % (exception_type, exception_value, exception_tb))
    emit(getFrameVariableTypeDescriptionCode(context))
    getGotoCode(context.getExceptionEscape(), emit)
    if context.needsCleanup(raise_type_name):
        context.removeCleanupTempName(raise_type_name)

def _getRaiseExceptionWithValueCode(raise_type_name, raise_value_name, implicit, emit, context):
    if False:
        return 10
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    emit('%s = %s;' % (exception_type, raise_type_name))
    getReferenceExportCode(raise_type_name, emit, context)
    emit('%s = %s;' % (exception_value, raise_value_name))
    getReferenceExportCode(raise_value_name, emit, context)
    emitErrorLineNumberUpdateCode(emit, context)
    emit('RAISE_EXCEPTION_%s(tstate, &%s, &%s, &%s);' % ('IMPLICIT' if implicit else 'WITH_VALUE', exception_type, exception_value, exception_tb))
    emit(getFrameVariableTypeDescriptionCode(context))
    getGotoCode(context.getExceptionEscape(), emit)
    if context.needsCleanup(raise_type_name):
        context.removeCleanupTempName(raise_type_name)
    if context.needsCleanup(raise_value_name):
        context.removeCleanupTempName(raise_value_name)

def _getRaiseExceptionWithTracebackCode(raise_type_name, raise_value_name, raise_tb_name, emit, context):
    if False:
        for i in range(10):
            print('nop')
    (exception_type, exception_value, exception_tb, _exception_lineno) = context.variable_storage.getExceptionVariableDescriptions()
    emit('%s = %s;' % (exception_type, raise_type_name))
    getReferenceExportCode(raise_type_name, emit, context)
    emit('%s = %s;' % (exception_value, raise_value_name))
    getReferenceExportCode(raise_value_name, emit, context)
    emit('%s = (PyTracebackObject *)%s;' % (exception_tb, raise_tb_name))
    getReferenceExportCode(raise_tb_name, emit, context)
    emit('RAISE_EXCEPTION_WITH_TRACEBACK(tstate, &%s, &%s, &%s);' % (exception_type, exception_value, exception_tb))
    emitErrorLineNumberUpdateCode(emit, context)
    emit(getFrameVariableTypeDescriptionCode(context))
    getGotoCode(context.getExceptionEscape(), emit)
    if context.needsCleanup(raise_type_name):
        context.removeCleanupTempName(raise_type_name)
    if context.needsCleanup(raise_value_name):
        context.removeCleanupTempName(raise_value_name)
    if context.needsCleanup(raise_tb_name):
        context.removeCleanupTempName(raise_tb_name)