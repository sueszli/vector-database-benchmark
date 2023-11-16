""" Integer related codes, long and int.

"""
from nuitka.PythonVersions import python_version
from .CodeHelpers import decideConversionCheckNeeded, generateChildExpressionsCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitCode
from .PythonAPICodes import generateCAPIObjectCode

def generateBuiltinLong1Code(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    assert python_version < 768
    generateCAPIObjectCode(to_name=to_name, capi='PyNumber_Long', tstate=False, arg_desc=(('long_arg', expression.subnode_value),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateBuiltinLong2Code(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    assert python_version < 768
    (value_name, base_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'long_value', expression, emit, context) as result_name:
        emit('%s = BUILTIN_LONG2(tstate, %s, %s);' % (result_name, value_name, base_name))
        getErrorExitCode(check_name=result_name, release_names=(value_name, base_name), emit=emit, context=context)
        context.addCleanupTempName(result_name)

def generateBuiltinInt1Code(to_name, expression, emit, context):
    if False:
        return 10
    generateCAPIObjectCode(to_name=to_name, capi='PyNumber_Int', tstate=False, arg_desc=(('int_arg', expression.subnode_value),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateBuiltinInt2Code(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    (value_name, base_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'int_value', expression, emit, context) as result_name:
        emit('%s = BUILTIN_INT2(tstate, %s, %s);' % (result_name, value_name, base_name))
        getErrorExitCode(check_name=result_name, release_names=(value_name, base_name), emit=emit, context=context)
        context.addCleanupTempName(result_name)