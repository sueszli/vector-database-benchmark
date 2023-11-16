""" Code generation for match statement helpers.

"""
from .CodeHelpers import generateChildExpressionsCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitCode

def generateMatchArgsCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    (matched_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'match_args_value', expression, emit, context) as value_name:
        emit('%s = MATCH_CLASS_ARGS(tstate, %s, %d);' % (value_name, matched_name, expression.max_allowed))
        getErrorExitCode(check_name=value_name, release_name=matched_name, emit=emit, context=context)
        context.addCleanupTempName(value_name)