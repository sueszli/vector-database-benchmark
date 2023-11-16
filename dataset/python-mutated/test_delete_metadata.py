import base64
import graphene
from .....core.error_codes import MetadataErrorCode
from .....core.models import ModelWithMetadata
from ....tests.utils import get_graphql_content
from . import PUBLIC_KEY, PUBLIC_KEY2, PUBLIC_VALUE, PUBLIC_VALUE2
from .test_update_metadata import item_contains_proper_public_metadata
DELETE_PUBLIC_METADATA_MUTATION = '\nmutation DeletePublicMetadata($id: ID!, $keys: [String!]!) {\n    deleteMetadata(\n        id: $id\n        keys: $keys\n    ) {\n        errors{\n            field\n            code\n        }\n        item {\n            metadata{\n                key\n                value\n            }\n            ...on %s{\n                id\n            }\n        }\n    }\n}\n'

def execute_clear_public_metadata_for_item(client, permissions, item_id, item_type, key=PUBLIC_KEY):
    if False:
        return 10
    variables = {'id': item_id, 'keys': [key]}
    response = client.post_graphql(DELETE_PUBLIC_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response)
    return response

def execute_clear_public_metadata_for_multiple_items(client, permissions, item_id, item_type, key=PUBLIC_KEY, key2=PUBLIC_KEY2):
    if False:
        i = 10
        return i + 15
    variables = {'id': item_id, 'keys': [key, key2]}
    response = client.post_graphql(DELETE_PUBLIC_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response)
    return response

def item_without_public_metadata(item_from_response, item, item_id, key=PUBLIC_KEY, value=PUBLIC_VALUE):
    if False:
        for i in range(10):
            print('nop')
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return item.get_value_from_metadata(key) != value

def item_without_multiple_public_metadata(item_from_response, item, item_id, key=PUBLIC_KEY, value=PUBLIC_VALUE, key2=PUBLIC_KEY2, value2=PUBLIC_VALUE2):
    if False:
        while True:
            i = 10
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return all([item.get_value_from_metadata(key) != value, item.get_value_from_metadata(key2) != value2])

def test_delete_public_metadata_for_non_exist_item(staff_api_client, permission_manage_payments):
    if False:
        return 10
    payment_id = 'Payment: 0'
    payment_id = base64.b64encode(str.encode(payment_id)).decode('utf-8')
    response = execute_clear_public_metadata_for_item(staff_api_client, permission_manage_payments, payment_id, 'Checkout')
    errors = response['data']['deleteMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_delete_public_metadata_for_item_without_meta(api_client, permission_group_manage_users):
    if False:
        for i in range(10):
            print('nop')
    group = permission_group_manage_users
    assert not issubclass(type(group), ModelWithMetadata)
    group_id = graphene.Node.to_global_id('Group', group.pk)
    response = execute_clear_public_metadata_for_item(api_client, None, group_id, 'User')
    errors = response['data']['deleteMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_delete_public_metadata_for_not_exist_key(api_client, checkout):
    if False:
        i = 10
        return i + 15
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.metadata_storage.save(update_fields=['metadata'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    response = execute_clear_public_metadata_for_item(api_client, None, checkout_id, 'Checkout', key='Not-exits')
    assert item_contains_proper_public_metadata(response['data']['deleteMetadata']['item'], checkout.metadata_storage, checkout_id)

def test_delete_public_metadata_for_one_key(api_client, checkout):
    if False:
        while True:
            i = 10
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE, 'to_clear': PUBLIC_VALUE})
    checkout.metadata_storage.save(update_fields=['metadata'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    response = execute_clear_public_metadata_for_item(api_client, None, checkout_id, 'Checkout', key='to_clear')
    assert item_contains_proper_public_metadata(response['data']['deleteMetadata']['item'], checkout.metadata_storage, checkout_id)
    assert item_without_public_metadata(response['data']['deleteMetadata']['item'], checkout.metadata_storage, checkout_id, key='to_clear')