""" Exception handling.

"""
from nuitka.PythonVersions import python_version
from .CodeHelpers import generateExpressionCode, withObjectCodeTemporaryAssignment
from .templates.CodeTemplatesExceptions import template_publish_exception_to_handler

def getExceptionIdentifier(exception_type):
    if False:
        while True:
            i = 10
    assert 'PyExc' not in exception_type, exception_type
    if exception_type == 'NotImplemented':
        return 'Py_NotImplemented'
    return 'PyExc_%s' % exception_type

def generateExceptionRefCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    exception_type = expression.getExceptionName()
    with withObjectCodeTemporaryAssignment(to_name, 'exception_name', expression, emit, context) as value_name:
        emit('%s = %s;' % (value_name, getExceptionIdentifier(exception_type)))

def getTracebackMakingIdentifier(context, lineno_name):
    if False:
        i = 10
        return i + 15
    frame_handle = context.getFrameHandle()
    assert frame_handle is not None
    return 'MAKE_TRACEBACK(%s, %s)' % (frame_handle, lineno_name)

def generateExceptionCaughtTypeCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    keeper_variables = context.getExceptionKeeperVariables()
    with withObjectCodeTemporaryAssignment(to_name, 'exception_caught_type', expression, emit, context) as value_name:
        if keeper_variables[0] is None:
            emit('%s = EXC_TYPE(PyThreadState_GET());' % (value_name,))
        else:
            emit('%s = %s;' % (value_name, keeper_variables[0]))

def generateExceptionCaughtValueCode(to_name, expression, emit, context):
    if False:
        return 10
    keeper_variables = context.getExceptionKeeperVariables()
    with withObjectCodeTemporaryAssignment(to_name, 'exception_caught_value', expression, emit, context) as value_name:
        if keeper_variables[1] is None:
            emit('%s = EXC_VALUE(PyThreadState_GET());' % (value_name,))
        elif python_version >= 624:
            emit('%s = %s;' % (value_name, keeper_variables[1]))
        else:
            emit('%s = %s ? %s : Py_None;' % (value_name, keeper_variables[1], keeper_variables[1]))

def generateExceptionCaughtTracebackCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    keeper_variables = context.getExceptionKeeperVariables()
    with withObjectCodeTemporaryAssignment(to_name, 'exception_caught_tb', expression, emit, context) as value_name:
        if keeper_variables[2] is None:
            if python_version < 944:
                emit('%s = (PyObject *)EXC_TRACEBACK(PyThreadState_GET());' % (value_name,))
            else:
                emit('%s = (PyObject *)GET_EXCEPTION_TRACEBACK(EXC_VALUE(PyThreadState_GET()));' % (value_name,))
        else:
            emit('if (%(keeper_tb)s != NULL) {\n    %(to_name)s = (PyObject *)%(keeper_tb)s;\n    Py_INCREF(%(to_name)s);\n} else {\n    %(to_name)s = (PyObject *)%(tb_making)s;\n}\n' % {'to_name': value_name, 'keeper_tb': keeper_variables[2], 'tb_making': getTracebackMakingIdentifier(context=context, lineno_name=keeper_variables[3])})
            context.addCleanupTempName(value_name)

def getExceptionUnpublishedReleaseCode(emit, context):
    if False:
        i = 10
        return i + 15
    keeper_variables = context.getExceptionKeeperVariables()
    if keeper_variables[0] is not None:
        emit('Py_DECREF(%s);' % keeper_variables[0])
        emit('Py_XDECREF(%s);' % keeper_variables[1])
        emit('Py_XDECREF(%s);' % keeper_variables[2])

def generateExceptionPublishCode(statement, emit, context):
    if False:
        print('Hello World!')
    (keeper_type, keeper_value, keeper_tb, keeper_lineno) = context.setExceptionKeeperVariables((None, None, None, None))
    emit(template_publish_exception_to_handler % {'tb_making': getTracebackMakingIdentifier(context=context, lineno_name=keeper_lineno), 'keeper_tb': keeper_tb, 'keeper_lineno': keeper_lineno, 'frame_identifier': context.getFrameHandle()})
    emit('PUBLISH_CURRENT_EXCEPTION(tstate, &%s, &%s, &%s);' % (keeper_type, keeper_value, keeper_tb))

def generateBuiltinMakeExceptionCode(to_name, expression, emit, context):
    if False:
        return 10
    from .CallCodes import getCallCodeNoArgs, getCallCodePosArgsQuick
    exception_arg_names = []
    for exception_arg in expression.subnode_args:
        exception_arg_name = context.allocateTempName('make_exception_arg')
        generateExpressionCode(to_name=exception_arg_name, expression=exception_arg, emit=emit, context=context)
        exception_arg_names.append(exception_arg_name)
    exception_type = expression.getExceptionName()
    with withObjectCodeTemporaryAssignment(to_name, 'exception_made', expression, emit, context) as value_name:
        if exception_arg_names:
            getCallCodePosArgsQuick(to_name=value_name, called_name=getExceptionIdentifier(exception_type), expression=expression, arg_names=exception_arg_names, emit=emit, context=context)
        else:
            getCallCodeNoArgs(to_name=value_name, called_name=getExceptionIdentifier(exception_type), expression=expression, emit=emit, context=context)
        if expression.getExceptionName() == 'ImportError' and python_version >= 768:
            from .PythonAPICodes import getReferenceExportCode
            import_error_name_expression = expression.subnode_name
            if import_error_name_expression is not None:
                exception_importerror_name = context.allocateTempName('make_exception_importerror_name')
                generateExpressionCode(to_name=exception_importerror_name, expression=import_error_name_expression, emit=emit, context=context, allow_none=True)
                getReferenceExportCode(exception_importerror_name, emit, context)
                if context.needsCleanup(exception_importerror_name):
                    context.removeCleanupTempName(exception_importerror_name)
                emit('((PyImportErrorObject *)%s)->name = %s;' % (to_name, exception_importerror_name))
            import_error_path_expression = expression.subnode_path
            if import_error_path_expression is not None:
                exception_importerror_path = context.allocateTempName('make_exception_importerror_path')
                generateExpressionCode(to_name=exception_importerror_path, expression=import_error_path_expression, emit=emit, context=context, allow_none=True)
                getReferenceExportCode(exception_importerror_path, emit, context)
                if context.needsCleanup(exception_importerror_path):
                    context.removeCleanupTempName(exception_importerror_path)
                emit('((PyImportErrorObject *)%s)->path = %s;' % (to_name, exception_importerror_path))