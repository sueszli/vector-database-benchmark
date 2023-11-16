""" Code generation for sets.

Right now only the creation, and set add code is done here. But more should be
added later on.
"""
from nuitka.PythonVersions import needsSetLiteralReverseInsertion
from .CodeHelpers import assignConstantNoneResult, decideConversionCheckNeeded, generateChildExpressionsCode, generateExpressionCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getAssertionCode, getErrorExitBoolCode
from .PythonAPICodes import generateCAPIObjectCode

def generateSetCreationCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    element_name = context.allocateTempName('set_element')
    elements = expression.subnode_elements
    assert elements, expression
    with withObjectCodeTemporaryAssignment(to_name, 'set_result', expression, emit, context) as result_name:
        for (count, element) in enumerate(elements):
            generateExpressionCode(to_name=element_name, expression=element, emit=emit, context=context)
            if count == 0:
                emit('%s = PySet_New(NULL);' % (result_name,))
                getAssertionCode(result_name, emit)
                context.addCleanupTempName(to_name)
            res_name = context.getIntResName()
            emit('%s = PySet_Add(%s, %s);' % (res_name, to_name, element_name))
            getErrorExitBoolCode(condition='%s != 0' % res_name, needs_check=not element.isKnownToBeHashable(), emit=emit, context=context)
            if context.needsCleanup(element_name):
                emit('Py_DECREF(%s);' % element_name)
                context.removeCleanupTempName(element_name)

def generateSetLiteralCreationCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    if not needsSetLiteralReverseInsertion():
        return generateSetCreationCode(to_name, expression, emit, context)
    with withObjectCodeTemporaryAssignment(to_name, 'set_result', expression, emit, context) as result_name:
        emit('%s = PySet_New(NULL);' % (result_name,))
        context.addCleanupTempName(result_name)
        elements = expression.subnode_elements
        element_names = []
        for (count, element) in enumerate(elements, 1):
            element_name = context.allocateTempName('set_element_%d' % count)
            element_names.append(element_name)
            generateExpressionCode(to_name=element_name, expression=element, emit=emit, context=context)
        for (count, element) in enumerate(elements):
            element_name = element_names[len(elements) - count - 1]
            if element.isKnownToBeHashable():
                emit('PySet_Add(%s, %s);' % (result_name, element_name))
            else:
                res_name = context.getIntResName()
                emit('%s = PySet_Add(%s, %s);' % (res_name, result_name, element_name))
                getErrorExitBoolCode(condition='%s != 0' % res_name, emit=emit, context=context)
            if context.needsCleanup(element_name):
                emit('Py_DECREF(%s);' % element_name)
                context.removeCleanupTempName(element_name)

def generateSetOperationAddCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    set_arg_name = context.allocateTempName('add_set')
    generateExpressionCode(to_name=set_arg_name, expression=statement.subnode_set_arg, emit=emit, context=context)
    value_arg_name = context.allocateTempName('add_value')
    generateExpressionCode(to_name=value_arg_name, expression=statement.subnode_value, emit=emit, context=context)
    context.setCurrentSourceCodeReference(statement.getSourceReference())
    res_name = context.getIntResName()
    emit('assert(PySet_Check(%s));' % set_arg_name)
    emit('%s = PySet_Add(%s, %s);' % (res_name, set_arg_name, value_arg_name))
    getErrorExitBoolCode(condition='%s == -1' % res_name, release_names=(set_arg_name, value_arg_name), emit=emit, context=context)

def generateSetOperationUpdateCode(to_name, expression, emit, context):
    if False:
        return 10
    res_name = context.getIntResName()
    (set_arg_name, value_arg_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    emit('assert(PySet_Check(%s));' % set_arg_name)
    emit('%s = _PySet_Update(%s, %s);' % (res_name, set_arg_name, value_arg_name))
    getErrorExitBoolCode(condition='%s == -1' % res_name, release_names=(set_arg_name, value_arg_name), emit=emit, context=context)
    assignConstantNoneResult(to_name, emit, context)

def generateBuiltinSetCode(to_name, expression, emit, context):
    if False:
        return 10
    generateCAPIObjectCode(to_name=to_name, capi='PySet_New', tstate=False, arg_desc=(('set_arg', expression.subnode_value),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateBuiltinFrozensetCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='PyFrozenSet_New', tstate=False, arg_desc=(('frozenset_arg', expression.subnode_value),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)