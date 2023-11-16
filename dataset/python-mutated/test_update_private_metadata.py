import base64
import graphene
from .....core.error_codes import MetadataErrorCode
from .....core.models import ModelWithMetadata
from ....tests.utils import assert_no_permission, get_graphql_content
from . import PRIVATE_KEY, PRIVATE_VALUE, PUBLIC_KEY, PUBLIC_KEY2, PUBLIC_VALUE, PUBLIC_VALUE2
UPDATE_PRIVATE_METADATA_MUTATION = '\nmutation UpdatePrivateMetadata($id: ID!, $input: [MetadataInput!]!) {\n    updatePrivateMetadata(\n        id: $id\n        input: $input\n    ) {\n        errors{\n            field\n            code\n        }\n        item {\n            privateMetadata{\n                key\n                value\n            }\n            ...on %s{\n                id\n            }\n        }\n    }\n}\n'

def execute_update_private_metadata_for_item(client, permissions, item_id, item_type, key=PRIVATE_KEY, value=PRIVATE_VALUE):
    if False:
        while True:
            i = 10
    variables = {'id': item_id, 'input': [{'key': key, 'value': value}]}
    response = client.post_graphql(UPDATE_PRIVATE_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response)
    return response

def execute_update_private_metadata_for_multiple_items(client, permissions, item_id, item_type, key=PUBLIC_KEY, value=PUBLIC_VALUE, key2=PUBLIC_KEY2, value2=PUBLIC_VALUE2):
    if False:
        i = 10
        return i + 15
    variables = {'id': item_id, 'input': [{'key': key, 'value': value}, {'key': key2, 'value': value2}]}
    response = client.post_graphql(UPDATE_PRIVATE_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response)
    return response

def item_contains_proper_private_metadata(item_from_response, item, item_id, key=PRIVATE_KEY, value=PRIVATE_VALUE):
    if False:
        for i in range(10):
            print('nop')
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return item.get_value_from_private_metadata(key) == value

def item_contains_multiple_proper_private_metadata(item_from_response, item, item_id, key=PUBLIC_KEY, value=PUBLIC_VALUE, key2=PUBLIC_KEY2, value2=PUBLIC_VALUE2):
    if False:
        for i in range(10):
            print('nop')
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return all([item.get_value_from_private_metadata(key) == value, item.get_value_from_private_metadata(key2) == value2])

def test_update_private_metadata_for_item(staff_api_client, checkout, permission_manage_checkouts):
    if False:
        return 10
    checkout.metadata_storage.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_KEY})
    checkout.metadata_storage.save(update_fields=['private_metadata'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    response = execute_update_private_metadata_for_item(staff_api_client, permission_manage_checkouts, checkout.token, 'Checkout', value='NewMetaValue')
    assert item_contains_proper_private_metadata(response['data']['updatePrivateMetadata']['item'], checkout.metadata_storage, checkout_id, value='NewMetaValue')

def test_update_private_metadata_for_non_exist_item(staff_api_client, permission_manage_payments):
    if False:
        while True:
            i = 10
    payment_id = 'Payment: 0'
    payment_id = base64.b64encode(str.encode(payment_id)).decode('utf-8')
    response = execute_update_private_metadata_for_item(staff_api_client, permission_manage_payments, payment_id, 'Payment')
    errors = response['data']['updatePrivateMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_update_private_metadata_for_item_without_meta(api_client, permission_group_manage_users):
    if False:
        i = 10
        return i + 15
    group = permission_group_manage_users
    assert not issubclass(type(group), ModelWithMetadata)
    group_id = graphene.Node.to_global_id('Group', group.pk)
    response = execute_update_private_metadata_for_item(api_client, None, group_id, 'User')
    errors = response['data']['updatePrivateMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_update_private_metadata_by_customer(user_api_client, payment):
    if False:
        while True:
            i = 10
    payment_id = graphene.Node.to_global_id('Payment', payment.pk)
    variables = {'id': payment_id, 'input': [{'key': PRIVATE_KEY, 'value': 'NewMetaValue'}]}
    response = user_api_client.post_graphql(UPDATE_PRIVATE_METADATA_MUTATION % 'Payment', variables, permissions=None)
    assert_no_permission(response)