""" Tuple codes

"""
from .CodeHelpers import decideConversionCheckNeeded, generateExpressionCode, withCleanupFinally, withObjectCodeTemporaryAssignment
from .PythonAPICodes import generateCAPIObjectCode

def _areConstants(expressions):
    if False:
        print('Hello World!')
    for expression in expressions:
        if not expression.isExpressionConstantRef():
            return False
        if expression.isMutable():
            return False
    return True

def generateTupleCreationCode(to_name, expression, emit, context):
    if False:
        return 10
    with withObjectCodeTemporaryAssignment(to_name, 'tuple_value', expression, emit, context) as value_name:
        getTupleCreationCode(to_name=value_name, elements=expression.subnode_elements, emit=emit, context=context)

def getTupleCreationCode(to_name, elements, emit, context):
    if False:
        i = 10
        return i + 15
    if _areConstants(elements):
        to_name.getCType().emitAssignmentCodeFromConstant(to_name=to_name, constant=tuple((element.getCompileTimeConstant() for element in elements)), may_escape=True, emit=emit, context=context)
    else:
        element_name = context.allocateTempName('tuple_element')

        def generateElementCode(element):
            if False:
                return 10
            generateExpressionCode(to_name=element_name, expression=element, emit=emit, context=context)
            if context.needsCleanup(element_name):
                context.removeCleanupTempName(element_name)
                helper_code = 'PyTuple_SET_ITEM'
            else:
                helper_code = 'PyTuple_SET_ITEM0'
            return helper_code
        helper_code = generateElementCode(elements[0])
        emit('%s = MAKE_TUPLE_EMPTY(%d);' % (to_name, len(elements)))
        needs_exception_exit = any((element.mayRaiseException(BaseException) for element in elements[1:]))
        with withCleanupFinally('tuple_build', to_name, needs_exception_exit, emit, context) as guarded_emit:
            emit = guarded_emit.emit
            for (count, element) in enumerate(elements):
                if count > 0:
                    helper_code = generateElementCode(element)
                emit('%s(%s, %d, %s);' % (helper_code, to_name, count, element_name))

def generateBuiltinTupleCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    generateCAPIObjectCode(to_name=to_name, capi='PySequence_Tuple', tstate=False, arg_desc=(('tuple_arg', expression.subnode_value),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)