""" Code generation for index values.

This is not for actual subscripts (see SubscriptCodes), but to convert
generic values to indexes. Also the maximum and minimum index values
are abstracted here.

"""
from .ErrorCodes import getErrorExitBoolCode

def getMaxIndexCode(to_name, emit):
    if False:
        i = 10
        return i + 15
    emit('%s = PY_SSIZE_T_MAX;' % to_name)

def getMinIndexCode(to_name, emit):
    if False:
        return 10
    emit('%s = 0;' % to_name)

def getIndexCode(to_name, value_name, emit, context):
    if False:
        i = 10
        return i + 15
    emit('%s = CONVERT_TO_INDEX(tstate, %s);' % (to_name, value_name))
    getErrorExitBoolCode(condition='%s == -1 && HAS_ERROR_OCCURRED(tstate)' % to_name, emit=emit, context=context)

def getIndexValueCode(to_name, value, emit):
    if False:
        return 10
    emit('%s = %d;' % (to_name, value))