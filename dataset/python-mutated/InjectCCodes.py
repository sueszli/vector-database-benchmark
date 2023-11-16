""" C code inject related nodes.

These are only coming from special purpose plugins.
"""

def generateInjectCCode(statement, emit, context):
    if False:
        for i in range(10):
            print('nop')
    emit(statement.c_code)