import base64
from unittest.mock import patch
import before_after
import graphene
import pytest
from django.core.exceptions import ValidationError
from django.db import transaction
from .....checkout.models import Checkout
from .....core.error_codes import MetadataErrorCode
from .....core.models import ModelWithMetadata
from ....tests.utils import get_graphql_content
from . import PUBLIC_KEY, PUBLIC_KEY2, PUBLIC_VALUE, PUBLIC_VALUE2
UPDATE_PUBLIC_METADATA_MUTATION = '\nmutation UpdatePublicMetadata($id: ID!, $input: [MetadataInput!]!) {\n    updateMetadata(\n        id: $id\n        input: $input\n    ) {\n        errors{\n            field\n            code\n            message\n        }\n        item {\n            metadata{\n                key\n                value\n            }\n            ...on %s{\n                id\n            }\n        }\n    }\n}\n'

def execute_update_public_metadata_for_item(client, permissions, item_id, item_type, key=PUBLIC_KEY, value=PUBLIC_VALUE, ignore_errors=False):
    if False:
        print('Hello World!')
    variables = {'id': item_id, 'input': [{'key': key, 'value': value}]}
    response = client.post_graphql(UPDATE_PUBLIC_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response, ignore_errors=ignore_errors)
    return response

def execute_update_public_metadata_for_multiple_items(client, permissions, item_id, item_type, key=PUBLIC_KEY, value=PUBLIC_VALUE, key2=PUBLIC_KEY2, value2=PUBLIC_VALUE2):
    if False:
        print('Hello World!')
    variables = {'id': item_id, 'input': [{'key': key, 'value': value}, {'key': key2, 'value': value2}]}
    response = client.post_graphql(UPDATE_PUBLIC_METADATA_MUTATION % item_type, variables, permissions=[permissions] if permissions else None)
    response = get_graphql_content(response)
    return response

def item_contains_proper_public_metadata(item_from_response, item, item_id, key=PUBLIC_KEY, value=PUBLIC_VALUE):
    if False:
        print('Hello World!')
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return item.get_value_from_metadata(key) == value

def item_contains_multiple_proper_public_metadata(item_from_response, item, item_id, key=PUBLIC_KEY, value=PUBLIC_VALUE, key2=PUBLIC_KEY2, value2=PUBLIC_VALUE2):
    if False:
        for i in range(10):
            print('nop')
    if item_from_response['id'] != item_id:
        return False
    item.refresh_from_db()
    return all([item.get_value_from_metadata(key) == value, item.get_value_from_metadata(key2) == value2])

def test_meta_mutations_handle_validation_errors(staff_api_client):
    if False:
        i = 10
        return i + 15
    invalid_id = '6QjoLs5LIqb3At7hVKKcUlqXceKkFK'
    variables = {'id': invalid_id, 'input': [{'key': 'year', 'value': 'of-saleor'}]}
    response = staff_api_client.post_graphql(UPDATE_PUBLIC_METADATA_MUTATION % 'Checkout', variables)
    content = get_graphql_content(response)
    errors = content['data']['updateMetadata']['errors']
    assert errors
    assert errors[0]['code'] == MetadataErrorCode.INVALID.name

@patch('saleor.plugins.manager.PluginsManager.checkout_updated')
def test_base_metadata_mutation_handles_errors_from_extra_action(mock_checkout_updated, api_client, checkout):
    if False:
        print('Hello World!')
    error_field = 'field'
    error_msg = 'boom'
    mock_checkout_updated.side_effect = ValidationError({error_field: error_msg})
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    response = execute_update_public_metadata_for_item(api_client, None, checkout_id, 'Checkout')
    errors = response['data']['updateMetadata']['errors']
    assert errors[0]['field'] == error_field
    assert errors[0]['message'] == error_msg

def test_update_public_metadata_for_item(api_client, checkout):
    if False:
        for i in range(10):
            print('nop')
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.metadata_storage.save(update_fields=['metadata'])
    checkout_id = graphene.Node.to_global_id('Checkout', checkout.pk)
    response = execute_update_public_metadata_for_item(api_client, None, checkout.token, 'Checkout', value='NewMetaValue')
    assert item_contains_proper_public_metadata(response['data']['updateMetadata']['item'], checkout.metadata_storage, checkout_id, value='NewMetaValue')

@pytest.mark.django_db(transaction=True)
def test_update_public_metadata_for_item_on_deleted_instance(api_client, checkout):
    if False:
        return 10
    checkout.metadata_storage.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    checkout.metadata_storage.save(update_fields=['metadata'])

    def delete_checkout_object(*args, **kwargs):
        if False:
            return 10
        with transaction.atomic():
            Checkout.objects.filter(pk=checkout.pk).delete()
    with before_after.before('saleor.graphql.meta.mutations.update_metadata.save_instance', delete_checkout_object):
        response = execute_update_public_metadata_for_item(api_client, None, checkout.token, 'Checkout', value='NewMetaValue', ignore_errors=True)
    assert not Checkout.objects.filter(pk=checkout.pk).first()
    assert response['data']['updateMetadata']['errors'][0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_update_public_metadata_for_non_exist_item(staff_api_client, permission_manage_payments):
    if False:
        while True:
            i = 10
    payment_id = 'Payment: 0'
    payment_id = base64.b64encode(str.encode(payment_id)).decode('utf-8')
    response = execute_update_public_metadata_for_item(staff_api_client, permission_manage_payments, payment_id, 'Payment')
    errors = response['data']['updateMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name

def test_update_public_metadata_for_item_without_meta(api_client, permission_group_manage_users):
    if False:
        return 10
    group = permission_group_manage_users
    assert not issubclass(type(group), ModelWithMetadata)
    group_id = graphene.Node.to_global_id('Group', group.pk)
    response = execute_update_public_metadata_for_item(api_client, None, group_id, 'User')
    errors = response['data']['updateMetadata']['errors']
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == MetadataErrorCode.NOT_FOUND.name