""" Expression codes, side effects, or statements that are an unused expression.

When you write "f()", i.e. you don't use the return value, that is an expression
only statement.

"""
from .CodeHelpers import generateExpressionCode
from .ErrorCodes import getReleaseCode

def generateExpressionOnlyCode(statement, emit, context):
    if False:
        for i in range(10):
            print('nop')
    return getStatementOnlyCode(value=statement.subnode_expression, emit=emit, context=context)

def getStatementOnlyCode(value, emit, context):
    if False:
        print('Hello World!')
    tmp_name = context.allocateTempName(base_name='unused', type_name='nuitka_void', unique=True)
    tmp_name.maybe_unused = True
    generateExpressionCode(expression=value, to_name=tmp_name, emit=emit, context=context)
    getReleaseCode(release_name=tmp_name, emit=emit, context=context)

def generateSideEffectsCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    for side_effect in expression.subnode_side_effects:
        getStatementOnlyCode(value=side_effect, emit=emit, context=context)
    generateExpressionCode(to_name=to_name, expression=expression.subnode_expression, emit=emit, context=context)