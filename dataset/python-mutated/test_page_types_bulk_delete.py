from unittest import mock
import graphene
import pytest
from .....page.models import Page
from ....attribute.utils import associate_attribute_values_to_instance
from ....tests.utils import assert_no_permission, get_graphql_content
PAGE_TYPE_BULK_DELETE_MUTATION = '\n    mutation PageTypeBulkDelete($ids: [ID!]!) {\n        pageTypeBulkDelete(ids: $ids) {\n            count\n            errors {\n                code\n                field\n                message\n            }\n        }\n    }\n'

def test_page_type_bulk_delete_by_staff(staff_api_client, page_type_list, permission_manage_page_types_and_attributes):
    if False:
        while True:
            i = 10
    staff_api_client.user.user_permissions.add(permission_manage_page_types_and_attributes)
    page_type_count = len(page_type_list)
    pages_pks = list(Page.objects.filter(page_type__in=page_type_list).values_list('pk', flat=True))
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = staff_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeBulkDelete']
    assert not data['errors']
    assert data['count'] == page_type_count
    assert not Page.objects.filter(pk__in=pages_pks)

@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_page_type_bulk_delete_trigger_webhooks(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, page_type_list, permission_manage_page_types_and_attributes, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    staff_api_client.user.user_permissions.add(permission_manage_page_types_and_attributes)
    page_type_count = len(page_type_list)
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = staff_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeBulkDelete']
    assert not data['errors']
    assert data['count'] == page_type_count
    assert mocked_webhook_trigger.call_count == page_type_count

def test_page_type_bulk_delete_by_staff_no_perm(staff_api_client, page_type_list, permission_manage_page_types_and_attributes):
    if False:
        while True:
            i = 10
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = staff_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    assert_no_permission(response)

def test_page_type_bulk_delete_by_app(app_api_client, page_type_list, permission_manage_page_types_and_attributes):
    if False:
        i = 10
        return i + 15
    app_api_client.app.permissions.add(permission_manage_page_types_and_attributes)
    page_type_count = len(page_type_list)
    pages_pks = list(Page.objects.filter(page_type__in=page_type_list).values_list('pk', flat=True))
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = app_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeBulkDelete']
    assert not data['errors']
    assert data['count'] == page_type_count
    assert not Page.objects.filter(pk__in=pages_pks)

def test_page_type_bulk_delete_by_app_no_perm(app_api_client, page_type_list, permission_manage_page_types_and_attributes):
    if False:
        for i in range(10):
            print('nop')
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = app_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    assert_no_permission(response)

def test_page_type_bulk_delete_with_file_attribute(app_api_client, page_type_list, page_file_attribute, permission_manage_page_types_and_attributes):
    if False:
        i = 10
        return i + 15
    app_api_client.app.permissions.add(permission_manage_page_types_and_attributes)
    page_type = page_type_list[1]
    page_type_count = len(page_type_list)
    page = Page.objects.filter(page_type=page_type.pk)[0]
    value = page_file_attribute.values.first()
    page_type.page_attributes.add(page_file_attribute)
    associate_attribute_values_to_instance(page, page_file_attribute, value)
    pages_pks = list(Page.objects.filter(page_type__in=page_type_list).values_list('pk', flat=True))
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    response = app_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeBulkDelete']
    assert not data['errors']
    assert data['count'] == page_type_count
    with pytest.raises(page_type._meta.model.DoesNotExist):
        page_type.refresh_from_db()
    with pytest.raises(value._meta.model.DoesNotExist):
        value.refresh_from_db()
    assert not Page.objects.filter(pk__in=pages_pks)

def test_page_type_bulk_delete_by_app_with_invalid_ids(app_api_client, page_type_list, permission_manage_page_types_and_attributes):
    if False:
        i = 10
        return i + 15
    variables = {'ids': [graphene.Node.to_global_id('PageType', page_type.pk) for page_type in page_type_list]}
    variables['ids'][0] = 'invalid_id'
    response = app_api_client.post_graphql(PAGE_TYPE_BULK_DELETE_MUTATION, variables, permissions=[permission_manage_page_types_and_attributes])
    content = get_graphql_content(response)
    errors = content['data']['pageTypeBulkDelete']['errors'][0]
    assert errors['code'] == 'GRAPHQL_ERROR'