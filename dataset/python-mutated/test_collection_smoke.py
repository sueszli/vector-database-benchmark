import botocore.session
import pytest
from botocore import xform_name
from boto3.resources.model import ResourceModel
from boto3.session import Session
COMMON_PAGINATION_PARAM_NAMES = ['nextToken', 'NextToken', 'marker', 'Marker', 'NextMarker', 'nextPageToken', 'NextPageToken']

def operation_looks_paginated(operation_model):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether an operation looks like it can be paginated\n\n    :type operation_model: botocore.model.OperationModel\n    :param operation_model: The model for a particular operation\n\n    :returns: True if determines it can be paginated. False otherwise.\n    '
    has_input_param = _shape_has_pagination_param(operation_model.input_shape)
    has_output_param = _shape_has_pagination_param(operation_model.output_shape)
    return has_input_param and has_output_param

def _shape_has_pagination_param(shape):
    if False:
        i = 10
        return i + 15
    if shape:
        members = shape.members
        for param in COMMON_PAGINATION_PARAM_NAMES:
            for member in members:
                if param == member:
                    return True
    return False

def _collection_test_args():
    if False:
        while True:
            i = 10
    botocore_session = botocore.session.get_session()
    session = Session(botocore_session=botocore_session)
    loader = botocore_session.get_component('data_loader')
    for service_name in session.get_available_resources():
        client = session.client(service_name, region_name='us-east-1')
        json_resource_model = loader.load_service_model(service_name, 'resources-1')
        resource_defs = json_resource_model['resources']
        resource_models = []
        service_resource_model = ResourceModel(service_name, json_resource_model['service'], resource_defs)
        resource_models.append(service_resource_model)
        for (resource_name, resource_defintion) in resource_defs.items():
            resource_models.append(ResourceModel(resource_name, resource_defintion, resource_defs))
        for resource_model in resource_models:
            for collection_model in resource_model.collections:
                yield (client, service_name, resource_name, collection_model)

@pytest.mark.parametrize('collection_args', _collection_test_args())
def test_all_collections_have_paginators_if_needed(collection_args):
    if False:
        while True:
            i = 10
    _assert_collection_has_paginator_if_needed(*collection_args)

def _assert_collection_has_paginator_if_needed(client, service_name, resource_name, collection_model):
    if False:
        while True:
            i = 10
    underlying_operation_name = collection_model.request.operation
    can_paginate_operation = client.can_paginate(xform_name(underlying_operation_name))
    looks_paginated = operation_looks_paginated(client.meta.service_model.operation_model(underlying_operation_name))
    if not can_paginate_operation:
        error_msg = f'Collection {collection_model.name} on resource {resource_name} of service {service_name} uses the operation {underlying_operation_name}, but the operation has no paginator even though it looks paginated.'
        assert not looks_paginated, error_msg