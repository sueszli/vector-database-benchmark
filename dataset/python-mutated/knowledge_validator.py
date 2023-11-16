from tribler.core.components.database.db.layers.knowledge_data_access_layer import Operation, ResourceType
from tribler.core.components.knowledge.knowledge_constants import MAX_RESOURCE_LENGTH, MIN_RESOURCE_LENGTH

def validate_resource(resource: str):
    if False:
        for i in range(10):
            print('nop')
    'Validate the resource. Raises ValueError, in the case the resource is not valid.'
    if len(resource) < MIN_RESOURCE_LENGTH or len(resource) > MAX_RESOURCE_LENGTH:
        raise ValueError(f'Tag length should be in range [{MIN_RESOURCE_LENGTH}..{MAX_RESOURCE_LENGTH}]')

def is_valid_resource(resource: str) -> bool:
    if False:
        while True:
            i = 10
    'Validate the resource. Returns False, in the case the resource is not valid.'
    try:
        validate_resource(resource)
    except ValueError:
        return False
    return True

def validate_operation(operation: int):
    if False:
        while True:
            i = 10
    'Validate the incoming operation. Raises ValueError, in the case the operation is not valid.'
    Operation(operation)

def validate_resource_type(t: int):
    if False:
        return 10
    'Validate the resource type. Raises ValueError, in the case the type is not valid.'
    ResourceType(t)