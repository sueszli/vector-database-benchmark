import json
from unittest import mock
import graphene
import pytest
from django.core.exceptions import ValidationError
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....menu.models import Menu
from .....product.models import Category
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....menu.mutations.menu_item_create import _validate_menu_item_instance
from ....tests.utils import get_graphql_content

def test_validate_menu_item_instance(category, page):
    if False:
        i = 10
        return i + 15
    _validate_menu_item_instance({'category': category}, 'category', Category)
    with pytest.raises(ValidationError):
        _validate_menu_item_instance({'category': page}, 'category', Category)
    _validate_menu_item_instance({}, 'category', Category)
    _validate_menu_item_instance({'category': None}, 'category', Category)
CREATE_MENU_QUERY = '\n    mutation mc($name: String!, $collection: ID,\n            $category: ID, $page: ID, $url: String) {\n\n        menuCreate(input: {\n            name: $name,\n            items: [\n                {name: "Collection item", collection: $collection},\n                {name: "Page item", page: $page},\n                {name: "Category item", category: $category},\n                {name: "Url item", url: $url}]\n        }) {\n            menu {\n                name\n                slug\n                items {\n                    id\n                }\n            }\n        }\n    }\n    '

def test_create_menu(staff_api_client, published_collection, category, page, permission_manage_menus):
    if False:
        print('Hello World!')
    category_id = graphene.Node.to_global_id('Category', category.pk)
    collection_id = graphene.Node.to_global_id('Collection', published_collection.pk)
    page_id = graphene.Node.to_global_id('Page', page.pk)
    url = 'http://www.example.com'
    variables = {'name': 'test-menu', 'collection': collection_id, 'category': category_id, 'page': page_id, 'url': url}
    response = staff_api_client.post_graphql(CREATE_MENU_QUERY, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuCreate']['menu']['name'] == 'test-menu'
    assert content['data']['menuCreate']['menu']['slug'] == 'test-menu'

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_create_menu_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, published_collection, category, page, permission_manage_menus, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    category_id = graphene.Node.to_global_id('Category', category.pk)
    collection_id = graphene.Node.to_global_id('Collection', published_collection.pk)
    page_id = graphene.Node.to_global_id('Page', page.pk)
    url = 'http://www.example.com'
    variables = {'name': 'test-menu', 'collection': collection_id, 'category': category_id, 'page': page_id, 'url': url}
    response = staff_api_client.post_graphql(CREATE_MENU_QUERY, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    menu = Menu.objects.last()
    assert content['data']['menuCreate']['menu']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': graphene.Node.to_global_id('Menu', menu.id), 'slug': menu.slug, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.MENU_CREATED, [any_webhook], menu, SimpleLazyObject(lambda : staff_api_client.user))

def test_create_menu_slug_already_exists(staff_api_client, collection, category, page, permission_manage_menus):
    if False:
        print('Hello World!')
    query = '\n        mutation MenuCreate(\n            $name: String!\n        ) {\n            menuCreate(input: { name: $name}) {\n                menu {\n                    name\n                    slug\n                }\n            }\n        }\n    '
    existing_menu = Menu.objects.create(name='test-menu', slug='test-menu')
    variables = {'name': 'test-menu'}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuCreate']['menu']['name'] == existing_menu.name
    assert content['data']['menuCreate']['menu']['slug'] == f'{existing_menu.slug}-2'

def test_create_menu_provided_slug(staff_api_client, collection, category, page, permission_manage_menus):
    if False:
        return 10
    query = '\n        mutation MenuCreate(\n            $name: String!\n            $slug: String\n        ) {\n            menuCreate(input: { name: $name, slug: $slug}) {\n                menu {\n                    name\n                    slug\n                }\n            }\n        }\n    '
    variables = {'name': 'test-menu', 'slug': 'test-slug'}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_menus])
    content = get_graphql_content(response)
    assert content['data']['menuCreate']['menu']['name'] == 'test-menu'
    assert content['data']['menuCreate']['menu']['slug'] == 'test-slug'