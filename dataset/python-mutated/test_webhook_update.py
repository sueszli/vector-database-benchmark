import json
import graphene
from .....app.models import App
from ....core.enums import WebhookErrorCode
from ....tests.utils import assert_no_permission, get_graphql_content
from ...enums import WebhookEventTypeAsyncEnum
WEBHOOK_UPDATE = '\n    mutation webhookUpdate ($id: ID!, $input: WebhookUpdateInput!) {\n      webhookUpdate(id: $id, input: $input) {\n        errors {\n          field\n          message\n          code\n        }\n        webhook {\n          syncEvents {\n            eventType\n          }\n          asyncEvents {\n            eventType\n          }\n          isActive\n          customHeaders\n        }\n      }\n    }\n'

def test_webhook_update_by_app(app_api_client, app, webhook):
    if False:
        for i in range(10):
            print('nop')
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    custom_headers = {'x-key': 'Value', 'authorization-key': 'Value'}
    variables = {'id': webhook_id, 'input': {'asyncEvents': [WebhookEventTypeAsyncEnum.ORDER_CREATED.name], 'isActive': False, 'customHeaders': json.dumps(custom_headers)}}
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    content = get_graphql_content(response)
    webhook.refresh_from_db()
    assert webhook.is_active is False
    events = webhook.events.all()
    assert len(events) == 1
    assert events[0].event_type == WebhookEventTypeAsyncEnum.ORDER_CREATED.value
    data = content['data']['webhookUpdate']
    assert not data['errors']
    assert len(data['webhook']['asyncEvents']) == 1
    assert data['webhook']['asyncEvents'][0]['eventType'] == WebhookEventTypeAsyncEnum.ORDER_CREATED.name
    assert data['webhook']['isActive'] is False
    assert data['webhook']['customHeaders'] == json.dumps(custom_headers)

def test_webhook_update_by_other_app(app_api_client, webhook):
    if False:
        for i in range(10):
            print('nop')
    other_app = App.objects.create(name='other')
    webhook.app = other_app
    webhook.save()
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'isActive': False}}
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    content = get_graphql_content(response)
    errors = content['data']['webhookUpdate']['errors']
    webhook.refresh_from_db()
    assert errors[0]['code'] == 'NOT_FOUND'
    assert webhook.is_active is True

def test_webhook_update_by_inactive_app(app_api_client, webhook):
    if False:
        while True:
            i = 10
    app = webhook.app
    app.is_active = False
    app.save()
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'isActive': False}}
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    assert_no_permission(response)

def test_webhook_update_app_cant_change_webhooks_ownership(app_api_client, app, webhook):
    if False:
        while True:
            i = 10
    other_app = App.objects.create(name='other')
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    other_app_id = graphene.Node.to_global_id('App', other_app.pk)
    variables = {'id': webhook_id, 'input': {'app': other_app_id, 'isActive': False}}
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    content = get_graphql_content(response)
    errors = content['data']['webhookUpdate']['errors']
    webhook.refresh_from_db()
    assert len(errors) == 0
    assert webhook.app_id == app.id
    assert webhook.is_active is False

def test_webhook_update_by_app_and_missing_webhook(app_api_client, webhook):
    if False:
        for i in range(10):
            print('nop')
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'isActive': False}}
    webhook.delete()
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    content = get_graphql_content(response)
    errors = content['data']['webhookUpdate']['errors']
    assert errors[0]['code'] == 'NOT_FOUND'

def test_webhook_update_when_app_doesnt_exist(app_api_client, app):
    if False:
        print('Hello World!')
    app.delete()
    webhook_id = graphene.Node.to_global_id('Webhook', 1)
    variables = {'id': webhook_id, 'input': {'isActive': False}}
    response = app_api_client.post_graphql(WEBHOOK_UPDATE, variables=variables)
    assert_no_permission(response)

def test_webhook_update_by_staff(staff_api_client, webhook, permission_manage_apps):
    if False:
        print('Hello World!')
    query = WEBHOOK_UPDATE
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    custom_headers = {'x-key': 'Value', 'authorization-key': 'Value'}
    variables = {'id': webhook_id, 'input': {'asyncEvents': [WebhookEventTypeAsyncEnum.CUSTOMER_CREATED.name, WebhookEventTypeAsyncEnum.CUSTOMER_CREATED.name], 'isActive': False, 'customHeaders': json.dumps(custom_headers)}}
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    response = staff_api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    webhook.refresh_from_db()
    assert webhook.is_active is False
    assert webhook.custom_headers == {'x-key': 'Value', 'authorization-key': 'Value'}
    events = webhook.events.all()
    assert len(events) == 1
    assert events[0].event_type == WebhookEventTypeAsyncEnum.CUSTOMER_CREATED.value
    data = content['data']['webhookUpdate']
    assert not data['errors']
    assert data['webhook']['asyncEvents'][0]['eventType'] == WebhookEventTypeAsyncEnum.CUSTOMER_CREATED.name
    assert data['webhook']['isActive'] is False
    assert data['webhook']['customHeaders'] == json.dumps(custom_headers)

def test_webhook_update_by_staff_without_permission(staff_api_client, app, webhook):
    if False:
        i = 10
        return i + 15
    query = WEBHOOK_UPDATE
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'asyncEvents': [WebhookEventTypeAsyncEnum.ORDER_CREATED.name, WebhookEventTypeAsyncEnum.CUSTOMER_CREATED.name], 'isActive': False}}
    response = staff_api_client.post_graphql(query, variables=variables)
    assert_no_permission(response)

def test_webhook_update_inherit_events_from_query(staff_api_client, app, webhook, permission_manage_apps, subscription_order_updated_webhook):
    if False:
        for i in range(10):
            print('nop')
    query = WEBHOOK_UPDATE
    subscription_query = subscription_order_updated_webhook.subscription_query
    initial_event = webhook.events.all()[0].event_type
    assert WebhookEventTypeAsyncEnum.ORDER_CREATED.value == initial_event
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'query': subscription_query}}
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    response = staff_api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    webhook.refresh_from_db()
    data = content['data']['webhookUpdate']
    assert not data['errors']
    events = webhook.events.all()
    assert len(events) == 1
    assert WebhookEventTypeAsyncEnum.ORDER_UPDATED.value == events[0].event_type

def test_webhook_update_invalid_custom_headers(staff_api_client, webhook, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    query = WEBHOOK_UPDATE
    custom_headers = {'DisallowedKey': 'Value'}
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    variables = {'id': webhook_id, 'input': {'customHeaders': json.dumps(custom_headers)}}
    response = staff_api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['webhookUpdate']
    assert not data['webhook']
    error = data['errors'][0]
    assert error['field'] == 'customHeaders'
    assert error['message'] == '"DisallowedKey" does not match allowed key pattern: "X-*" or "Authorization*".'
    assert error['code'] == WebhookErrorCode.INVALID_CUSTOM_HEADERS.name

def test_webhook_update_notify_user_with_another_event(app_api_client, webhook):
    if False:
        i = 10
        return i + 15
    query = WEBHOOK_UPDATE
    webhook_id = graphene.Node.to_global_id('Webhook', webhook.pk)
    variables = {'id': webhook_id, 'input': {'name': 'NOTIFY_USER with another event fails to save', 'targetUrl': 'https://www.example.com', 'asyncEvents': [WebhookEventTypeAsyncEnum.ORDER_CREATED.name, WebhookEventTypeAsyncEnum.NOTIFY_USER.name]}}
    response = app_api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['webhookUpdate']
    assert not data['webhook']
    error = data['errors'][0]
    assert error['field'] == 'asyncEvents'
    assert error['code'] == WebhookErrorCode.INVALID_NOTIFY_WITH_SUBSCRIPTION.name