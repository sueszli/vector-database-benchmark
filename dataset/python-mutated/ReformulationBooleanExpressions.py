""" Reformulation of boolean and/or expressions.

Consult the Developer Manual for information. TODO: Add ability to sync
source code comments with Developer Manual sections.

"""
from nuitka.nodes.ConditionalNodes import ExpressionConditionalAnd, ExpressionConditionalOr, makeNotExpression
from .TreeHelpers import buildNode, buildNodeList, getKind

def buildBoolOpNode(provider, node, source_ref):
    if False:
        while True:
            i = 10
    bool_op = getKind(node.op)
    if bool_op == 'Or':
        values = buildNodeList(provider, node.values, source_ref)
        for value in values[:-1]:
            value.setCompatibleSourceReference(values[-1].getSourceReference())
        source_ref = values[-1].getSourceReference()
        return makeOrNode(values=values, source_ref=source_ref)
    elif bool_op == 'And':
        values = buildNodeList(provider, node.values, source_ref)
        for value in values[:-1]:
            value.setCompatibleSourceReference(values[-1].getSourceReference())
        source_ref = values[-1].getSourceReference()
        return makeAndNode(values=values, source_ref=source_ref)
    elif bool_op == 'Not':
        return makeNotExpression(expression=buildNode(provider, node.operand, source_ref))
    else:
        assert False, bool_op

def makeOrNode(values, source_ref):
    if False:
        return 10
    values = list(values)
    result = values.pop()
    assert values
    while values:
        result = ExpressionConditionalOr(left=values.pop(), right=result, source_ref=source_ref)
    return result

def makeAndNode(values, source_ref):
    if False:
        print('Hello World!')
    values = list(values)
    result = values.pop()
    while values:
        result = ExpressionConditionalAnd(left=values.pop(), right=result, source_ref=source_ref)
    return result