import json
from unittest import mock
import graphene
import pytest
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import get_graphql_content
ATTRIBUTE_DELETE_MUTATION = '\n    mutation deleteAttribute($id: ID!) {\n        attributeDelete(id: $id) {\n            errors {\n                field\n                message\n            }\n            attribute {\n                id\n            }\n        }\n    }\n'

def test_delete_attribute(staff_api_client, color_attribute, permission_manage_product_types_and_attributes, product_type):
    if False:
        print('Hello World!')
    attribute = color_attribute
    query = ATTRIBUTE_DELETE_MUTATION
    node_id = graphene.Node.to_global_id('Attribute', attribute.id)
    variables = {'id': node_id}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    data = content['data']['attributeDelete']
    assert data['attribute']['id'] == variables['id']
    with pytest.raises(attribute._meta.model.DoesNotExist):
        attribute.refresh_from_db()

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_delete_attribute_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, color_attribute, permission_manage_product_types_and_attributes, product_type, settings):
    if False:
        while True:
            i = 10
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    node_id = graphene.Node.to_global_id('Attribute', color_attribute.id)
    variables = {'id': node_id}
    response = staff_api_client.post_graphql(ATTRIBUTE_DELETE_MUTATION, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    data = content['data']['attributeDelete']
    assert not data['errors']
    assert data['attribute']['id'] == variables['id']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': graphene.Node.to_global_id('Attribute', color_attribute.id), 'name': color_attribute.name, 'slug': color_attribute.slug, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.ATTRIBUTE_DELETED, [any_webhook], color_attribute, SimpleLazyObject(lambda : staff_api_client.user))

def test_delete_file_attribute(staff_api_client, file_attribute, permission_manage_product_types_and_attributes, product_type):
    if False:
        return 10
    attribute = file_attribute
    query = ATTRIBUTE_DELETE_MUTATION
    node_id = graphene.Node.to_global_id('Attribute', attribute.id)
    variables = {'id': node_id}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    data = content['data']['attributeDelete']
    assert data['attribute']['id'] == variables['id']
    with pytest.raises(attribute._meta.model.DoesNotExist):
        attribute.refresh_from_db()
ATTRIBUTE_DELETE_BY_EXTERNAL_REFERENCE_MUTATION = '\n    mutation deleteAttribute($id: ID, $externalReference: String) {\n        attributeDelete(id: $id, externalReference: $externalReference) {\n            attribute {\n                id\n                externalReference\n            }\n            errors {\n                field\n                message\n            }\n        }\n    }\n'

def test_delete_attribute_by_external_reference(staff_api_client, color_attribute, permission_manage_product_types_and_attributes):
    if False:
        i = 10
        return i + 15
    attribute = color_attribute
    query = ATTRIBUTE_DELETE_BY_EXTERNAL_REFERENCE_MUTATION
    ext_ref = 'test-ext-ref'
    attribute.external_reference = ext_ref
    attribute.save(update_fields=['external_reference'])
    variables = {'externalReference': ext_ref}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    data = content['data']['attributeDelete']
    with pytest.raises(attribute._meta.model.DoesNotExist):
        attribute.refresh_from_db()
    assert data['attribute']['externalReference'] == ext_ref
    assert graphene.Node.to_global_id('Attribute', attribute.id) == data['attribute']['id']

def test_delete_attribute_by_both_id_and_external_reference(staff_api_client, permission_manage_product_types_and_attributes):
    if False:
        print('Hello World!')
    query = ATTRIBUTE_DELETE_BY_EXTERNAL_REFERENCE_MUTATION
    variables = {'externalReference': 'whatever', 'id': 'whatever'}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    errors = content['data']['attributeDelete']['errors']
    assert errors[0]['message'] == "Argument 'id' cannot be combined with 'external_reference'"

def test_delete_attribute_by_external_reference_not_existing(staff_api_client, permission_manage_product_types_and_attributes):
    if False:
        while True:
            i = 10
    query = ATTRIBUTE_DELETE_BY_EXTERNAL_REFERENCE_MUTATION
    ext_ref = 'non-existing-ext-ref'
    variables = {'externalReference': ext_ref}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    errors = content['data']['attributeDelete']['errors']
    assert errors[0]['message'] == f"Couldn't resolve to a node: {ext_ref}"