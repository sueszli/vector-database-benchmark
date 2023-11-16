import json
from unittest import mock
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....menu.error_codes import MenuErrorCode
from .....menu.models import Menu
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import get_graphql_content
UPDATE_MENU_WITH_SLUG_MUTATION = '\n    mutation updatemenu($id: ID!, $name: String! $slug: String) {\n        menuUpdate(id: $id, input: {name: $name, slug: $slug}) {\n            menu {\n                name\n                slug\n            }\n            errors {\n                field\n                code\n            }\n        }\n    }\n'

def test_update_menu(staff_api_client, menu, permission_manage_menus):
    if False:
        while True:
            i = 10
    query = '\n        mutation updatemenu($id: ID!, $name: String!) {\n            menuUpdate(id: $id, input: {name: $name}) {\n                menu {\n                    name\n                    slug\n                }\n                errors {\n                    field\n                    code\n                }\n            }\n        }\n    '
    name = 'Blue oyster menu'
    variables = {'id': graphene.Node.to_global_id('Menu', menu.pk), 'name': name}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuUpdate']['menu']['name'] == name
    assert content['data']['menuUpdate']['menu']['slug'] == menu.slug

def test_update_menu_with_slug(staff_api_client, menu, permission_manage_menus):
    if False:
        return 10
    name = 'Blue oyster menu'
    variables = {'id': graphene.Node.to_global_id('Menu', menu.pk), 'name': name, 'slug': 'new-slug'}
    response = staff_api_client.post_graphql(UPDATE_MENU_WITH_SLUG_MUTATION, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuUpdate']['menu']['name'] == name
    assert content['data']['menuUpdate']['menu']['slug'] == 'new-slug'

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_update_menu_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, menu, permission_manage_menus, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    name = 'Blue oyster menu'
    variables = {'id': graphene.Node.to_global_id('Menu', menu.pk), 'name': name, 'slug': 'new-slug'}
    response = staff_api_client.post_graphql(UPDATE_MENU_WITH_SLUG_MUTATION, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuUpdate']['menu']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': variables['id'], 'slug': variables['slug'], 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.MENU_UPDATED, [any_webhook], menu, SimpleLazyObject(lambda : staff_api_client.user))

def test_update_menu_with_slug_already_exists(staff_api_client, menu, permission_manage_menus):
    if False:
        print('Hello World!')
    existing_menu = Menu.objects.create(name='test-slug-menu', slug='test-slug-menu')
    variables = {'id': graphene.Node.to_global_id('Menu', menu.pk), 'name': 'Blue oyster menu', 'slug': existing_menu.slug}
    response = staff_api_client.post_graphql(UPDATE_MENU_WITH_SLUG_MUTATION, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    error = content['data']['menuUpdate']['errors'][0]
    assert error['field'] == 'slug'
    assert error['code'] == MenuErrorCode.UNIQUE.name