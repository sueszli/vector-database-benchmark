""" Return codes

This handles code generation for return statements of normal functions and of
generator functions. Also the value currently being returned, and intercepted
by a try statement is accessible this way.

"""
from nuitka.PythonVersions import python_version
from .CodeHelpers import generateExpressionCode
from .ExceptionCodes import getExceptionUnpublishedReleaseCode
from .LabelCodes import getGotoCode

def generateReturnCode(statement, emit, context):
    if False:
        while True:
            i = 10
    getExceptionUnpublishedReleaseCode(emit, context)
    return_value = statement.subnode_expression
    return_value_name = context.getReturnValueName()
    if context.getReturnReleaseMode():
        emit('CHECK_OBJECT(%s);' % return_value_name)
        emit('Py_DECREF(%s);' % return_value_name)
    generateExpressionCode(to_name=return_value_name, expression=return_value, emit=emit, context=context)
    if context.needsCleanup(return_value_name):
        context.removeCleanupTempName(return_value_name)
    else:
        emit('Py_INCREF(%s);' % return_value_name)
    getGotoCode(label=context.getReturnTarget(), emit=emit)

def generateReturnedValueCode(statement, emit, context):
    if False:
        for i in range(10):
            print('nop')
    getExceptionUnpublishedReleaseCode(emit, context)
    getGotoCode(label=context.getReturnTarget(), emit=emit)

def generateReturnConstantCode(statement, emit, context):
    if False:
        while True:
            i = 10
    getExceptionUnpublishedReleaseCode(emit, context)
    return_value_name = context.getReturnValueName()
    if context.getReturnReleaseMode():
        emit('CHECK_OBJECT(%s);' % return_value_name)
        emit('Py_DECREF(%s);' % return_value_name)
    return_value_name.getCType().emitAssignmentCodeFromConstant(to_name=return_value_name, constant=statement.getConstant(), may_escape=True, emit=emit, context=context)
    if context.needsCleanup(return_value_name):
        context.removeCleanupTempName(return_value_name)
    else:
        emit('Py_INCREF(%s);' % return_value_name)
    getGotoCode(label=context.getReturnTarget(), emit=emit)

def generateGeneratorReturnValueCode(statement, emit, context):
    if False:
        print('Hello World!')
    if context.getOwner().isExpressionAsyncgenObjectBody():
        pass
    elif python_version >= 768:
        return_value_name = context.getGeneratorReturnValueName()
        expression = statement.subnode_expression
        if context.getReturnReleaseMode():
            emit('CHECK_OBJECT(%s);' % return_value_name)
            emit('Py_DECREF(%s);' % return_value_name)
        generateExpressionCode(to_name=return_value_name, expression=expression, emit=emit, context=context)
        if context.needsCleanup(return_value_name):
            context.removeCleanupTempName(return_value_name)
        else:
            emit('Py_INCREF(%s);' % return_value_name)
    elif statement.getParentVariableProvider().needsGeneratorReturnHandling():
        return_value_name = context.getGeneratorReturnValueName()
        generator_return_name = context.allocateTempName('generator_return', 'bool', unique=True)
        emit('%s = true;' % generator_return_name)
    getGotoCode(context.getReturnTarget(), emit)

def generateGeneratorReturnNoneCode(statement, emit, context):
    if False:
        for i in range(10):
            print('nop')
    if context.getOwner().isExpressionAsyncgenObjectBody():
        pass
    elif python_version >= 768:
        return_value_name = context.getGeneratorReturnValueName()
        if context.getReturnReleaseMode():
            emit('CHECK_OBJECT(%s);' % return_value_name)
            emit('Py_DECREF(%s);' % return_value_name)
        return_value_name.getCType().emitAssignmentCodeFromConstant(to_name=return_value_name, constant=None, may_escape=False, emit=emit, context=context)
        if context.needsCleanup(return_value_name):
            context.removeCleanupTempName(return_value_name)
        else:
            emit('Py_INCREF(%s);' % return_value_name)
    elif statement.getParentVariableProvider().needsGeneratorReturnHandling():
        return_value_name = context.getGeneratorReturnValueName()
        generator_return_name = context.allocateTempName('generator_return', 'bool', unique=True)
        emit('%s = true;' % generator_return_name)
    getGotoCode(context.getReturnTarget(), emit)