import pytest
from tribler.core.components.database.db.layers.knowledge_data_access_layer import Operation, ResourceType
from tribler.core.components.knowledge.community.knowledge_validator import is_valid_resource, validate_operation, validate_resource, validate_resource_type
VALID_TAGS = ['nl', 'tag', 'Tag', 'Тэг', 'Tag with space']
INVALID_TAGS = ['', 't', 't' * 51]

@pytest.mark.parametrize('tag', VALID_TAGS)
def test_valid_tags(tag):
    if False:
        for i in range(10):
            print('nop')
    validate_resource(tag)
    assert is_valid_resource(tag)

@pytest.mark.parametrize('tag', INVALID_TAGS)
def test_invalid(tag):
    if False:
        for i in range(10):
            print('nop')
    assert not is_valid_resource(tag)
    with pytest.raises(ValueError):
        validate_resource(tag)

def test_correct_operation():
    if False:
        while True:
            i = 10
    for operation in Operation:
        validate_operation(operation)
        validate_operation(operation.value)

def test_incorrect_operation():
    if False:
        print('Hello World!')
    max_operation = max(Operation)
    with pytest.raises(ValueError):
        validate_operation(max_operation.value + 1)

def test_correct_relation():
    if False:
        while True:
            i = 10
    for relation in ResourceType:
        validate_resource_type(relation)
        validate_resource_type(relation.value)

def test_incorrect_relation():
    if False:
        i = 10
        return i + 15
    max_relation = max(ResourceType)
    with pytest.raises(ValueError):
        validate_operation(max_relation.value + 1)