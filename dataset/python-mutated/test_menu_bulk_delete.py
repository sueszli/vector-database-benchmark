from unittest import mock
import graphene
import pytest
from .....menu.models import Menu
from ....tests.utils import get_graphql_content

@pytest.fixture
def menu_list():
    if False:
        for i in range(10):
            print('nop')
    menu_1 = Menu.objects.create(name='test-navbar-1', slug='test-navbar-1')
    menu_2 = Menu.objects.create(name='test-navbar-2', slug='test-navbar-2')
    menu_3 = Menu.objects.create(name='test-navbar-3', slug='test-navbar-3')
    return (menu_1, menu_2, menu_3)
BULK_DELETE_MENUS_MUTATION = '\n    mutation menuBulkDelete($ids: [ID!]!) {\n        menuBulkDelete(ids: $ids) {\n            count\n        }\n    }\n    '

def test_delete_menus(staff_api_client, menu_list, permission_manage_menus):
    if False:
        while True:
            i = 10
    variables = {'ids': [graphene.Node.to_global_id('Menu', collection.id) for collection in menu_list]}
    response = staff_api_client.post_graphql(BULK_DELETE_MENUS_MUTATION, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuBulkDelete']['count'] == 3
    assert not Menu.objects.filter(id__in=[menu.id for menu in menu_list]).exists()

@mock.patch('saleor.graphql.menu.bulk_mutations.menu_bulk_delete.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_delete_menus_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, staff_api_client, menu_list, permission_manage_menus, any_webhook, settings):
    if False:
        for i in range(10):
            print('nop')
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    variables = {'ids': [graphene.Node.to_global_id('Menu', collection.id) for collection in menu_list]}
    response = staff_api_client.post_graphql(BULK_DELETE_MENUS_MUTATION, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuBulkDelete']['count'] == 3
    assert mocked_webhook_trigger.call_count == len(menu_list)