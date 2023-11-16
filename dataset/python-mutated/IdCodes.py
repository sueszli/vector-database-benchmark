""" Codes for id and hash

"""
from .CodeHelpers import decideConversionCheckNeeded
from .PythonAPICodes import generateCAPIObjectCode

def generateBuiltinIdCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='PyLong_FromVoidPtr', tstate=False, arg_desc=(('id_arg', expression.subnode_value),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateBuiltinHashCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    generateCAPIObjectCode(to_name=to_name, capi='BUILTIN_HASH', tstate=True, arg_desc=(('hash_arg', expression.subnode_value),), may_raise=expression.mayRaiseExceptionOperation(), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)