import json
from unittest import mock
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....app.models import App
from .....core.utils.json_serializer import CustomJsonEncoder
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import assert_no_permission, get_graphql_content
APP_DEACTIVATE_MUTATION = '\n    mutation AppDeactivate($id: ID!){\n      appDeactivate(id:$id){\n        app{\n          id\n          isActive\n        }\n        errors{\n          field\n          message\n          code\n        }\n      }\n    }\n'

def test_deactivate_app(app, staff_api_client, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    app.is_active = True
    app.save()
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    response = staff_api_client.post_graphql(query, variables=variables, permissions=(permission_manage_apps,))
    get_graphql_content(response)
    app.refresh_from_db()
    assert not app.is_active

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_deactivate_app_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, app, staff_api_client, permission_manage_apps, settings):
    if False:
        while True:
            i = 10
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    app.is_active = True
    app.save()
    variables = {'id': graphene.Node.to_global_id('App', app.id)}
    staff_api_client.post_graphql(APP_DEACTIVATE_MUTATION, variables=variables, permissions=(permission_manage_apps,))
    app.refresh_from_db()
    assert not app.is_active
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': variables['id'], 'is_active': app.is_active, 'name': app.name, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.APP_STATUS_CHANGED, [any_webhook], app, SimpleLazyObject(lambda : staff_api_client.user))

def test_deactivate_app_by_app(app, app_api_client, permission_manage_apps):
    if False:
        while True:
            i = 10
    app = App.objects.create(name='Sample app objects', is_active=True)
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    app_api_client.app.permissions.set([permission_manage_apps])
    response = app_api_client.post_graphql(query, variables=variables)
    get_graphql_content(response)
    app.refresh_from_db()
    assert not app.is_active

def test_deactivate_app_missing_permission(app, staff_api_client, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    app.is_active = True
    app.save()
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    response = staff_api_client.post_graphql(query, variables=variables, permissions=(permission_manage_orders,))
    assert_no_permission(response)
    app.refresh_from_db()
    assert app.is_active

def test_activate_app_by_app_missing_permission(app, app_api_client, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    app = App.objects.create(name='Sample app objects', is_active=True)
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    app_api_client.app.permissions.set([permission_manage_orders])
    response = app_api_client.post_graphql(query, variables=variables)
    assert_no_permission(response)
    assert app.is_active

def test_app_has_more_permission_than_user_requestor(app, staff_api_client, permission_manage_orders, permission_manage_apps):
    if False:
        while True:
            i = 10
    app.permissions.add(permission_manage_orders)
    app.is_active = True
    app.save()
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    response = staff_api_client.post_graphql(query, variables=variables, permissions=(permission_manage_apps,))
    content = get_graphql_content(response)
    app_data = content['data']['appDeactivate']['app']
    app_errors = content['data']['appDeactivate']['errors']
    app.refresh_from_db()
    assert not app_errors
    assert not app.is_active
    assert app_data['isActive'] is False

def test_app_has_more_permission_than_app_requestor(app_api_client, permission_manage_orders, permission_manage_apps):
    if False:
        print('Hello World!')
    app = App.objects.create(name='Sample app objects', is_active=True)
    app.permissions.add(permission_manage_orders)
    query = APP_DEACTIVATE_MUTATION
    id = graphene.Node.to_global_id('App', app.id)
    variables = {'id': id}
    response = app_api_client.post_graphql(query, variables=variables, permissions=(permission_manage_apps,))
    content = get_graphql_content(response)
    app_data = content['data']['appDeactivate']['app']
    app_errors = content['data']['appDeactivate']['errors']
    app.refresh_from_db()
    assert not app_errors
    assert not app.is_active
    assert app_data['isActive'] is False