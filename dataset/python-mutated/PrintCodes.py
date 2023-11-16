""" Print related codes.

This is broken down to to level on printing one individual item, and
a new line potentially. The heavy lifting for 'softspace', etc. is
happening in the C helper functions.

"""
from .CodeHelpers import generateExpressionCode
from .ErrorCodes import getErrorExitBoolCode

def generatePrintValueCode(statement, emit, context):
    if False:
        return 10
    destination = statement.subnode_dest
    value = statement.subnode_value
    if destination is not None:
        dest_name = context.allocateTempName('print_dest', unique=True)
        generateExpressionCode(expression=destination, to_name=dest_name, emit=emit, context=context)
    else:
        dest_name = None
    value_name = context.allocateTempName('print_value', unique=True)
    generateExpressionCode(expression=value, to_name=value_name, emit=emit, context=context)
    with context.withCurrentSourceCodeReference(statement.getSourceReference()):
        res_name = context.getBoolResName()
        if dest_name is not None:
            print_code = '%s = PRINT_ITEM_TO(%s, %s);' % (res_name, dest_name, value_name)
        else:
            print_code = '%s = PRINT_ITEM(%s);' % (res_name, value_name)
        emit(print_code)
        getErrorExitBoolCode(condition='%s == false' % res_name, release_names=(dest_name, value_name), emit=emit, context=context)

def generatePrintNewlineCode(statement, emit, context):
    if False:
        return 10
    destination = statement.subnode_dest
    if destination is not None:
        dest_name = context.allocateTempName('print_dest', unique=True)
        generateExpressionCode(expression=destination, to_name=dest_name, emit=emit, context=context)
    else:
        dest_name = None
    with context.withCurrentSourceCodeReference(statement.getSourceReference()):
        if dest_name is not None:
            print_code = 'PRINT_NEW_LINE_TO(%s) == false' % (dest_name,)
        else:
            print_code = 'PRINT_NEW_LINE() == false'
        getErrorExitBoolCode(condition=print_code, release_name=dest_name, emit=emit, context=context)