""" Node children checkers.

The role of checkers is to make sure that node children have specific value
types only.

"""

def checkStatementsSequenceOrNone(value):
    if False:
        for i in range(10):
            print('nop')
    if value is not None:
        assert value.kind == 'STATEMENTS_SEQUENCE', value
        if not value.subnode_statements:
            return None
    return value

def checkStatementsSequence(value):
    if False:
        print('Hello World!')
    assert value is not None and value.kind == 'STATEMENTS_SEQUENCE', value
    return value

def convertNoneConstantToNone(node):
    if False:
        print('Hello World!')
    if node is None or node.isExpressionConstantNoneRef():
        return None
    else:
        return node

def convertEmptyStrConstantToNone(node):
    if False:
        while True:
            i = 10
    if node is None or node.isExpressionConstantStrEmptyRef():
        return None
    else:
        return node