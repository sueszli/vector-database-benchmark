from graphql import OperationType, parse
from strawberry.utils.operation import get_first_operation

def test_document_without_operation_definition_notes():
    if False:
        while True:
            i = 10
    document = parse('\n        fragment Test on Query {\n            hello\n        }\n    ')
    assert get_first_operation(document) is None

def test_single_operation_definition_note():
    if False:
        for i in range(10):
            print('nop')
    document = parse('\n        query Operation1 {\n            hello\n        }\n    ')
    assert get_first_operation(document) is not None
    assert get_first_operation(document).operation == OperationType.QUERY

def test_multiple_operation_definition_notes():
    if False:
        i = 10
        return i + 15
    document = parse('\n        mutation Operation1 {\n            hello\n        }\n        query Operation2 {\n            hello\n        }\n    ')
    assert get_first_operation(document) is not None
    assert get_first_operation(document).operation == OperationType.MUTATION