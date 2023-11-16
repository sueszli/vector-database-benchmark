import json
from unittest import mock
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....giftcard import GiftCardEvents
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import assert_no_permission, get_graphql_content
DEACTIVATE_GIFT_CARD_MUTATION = '\n    mutation giftCardDeactivate($id: ID!) {\n        giftCardDeactivate(id: $id) {\n            errors {\n                field\n                message\n            }\n            giftCard {\n                isActive\n                events {\n                    type\n                    user {\n                        email\n                    }\n                    app {\n                        name\n                    }\n                }\n            }\n        }\n    }\n'

def test_deactivate_gift_card_by_staff(staff_api_client, gift_card, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        print('Hello World!')
    assert gift_card.is_active
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = staff_api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardDeactivate']['giftCard']
    assert data['isActive'] is False
    events = data['events']
    assert len(events) == 1
    assert events[0]['type'] == GiftCardEvents.DEACTIVATED.upper()
    assert events[0]['user']['email'] == staff_api_client.user.email
    assert events[0]['app'] is None

def test_deactivate_gift_card_by_app(app_api_client, gift_card, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        for i in range(10):
            print('nop')
    assert gift_card.is_active
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = app_api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardDeactivate']['giftCard']
    assert data['isActive'] is False
    events = data['events']
    assert len(events) == 1
    assert events[0]['type'] == GiftCardEvents.DEACTIVATED.upper()
    assert events[0]['user'] is None
    assert events[0]['app']['name'] == app_api_client.app.name

def test_deactivate_gift_card_by_customer(api_client, gift_card):
    if False:
        print('Hello World!')
    gift_card.is_active = False
    gift_card.save(update_fields=['is_active'])
    assert not gift_card.is_active
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables)
    assert_no_permission(response)

def test_deactivate_gift_card_without_premissions(staff_api_client, gift_card):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = staff_api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables)
    assert_no_permission(response)

def test_deactivate_inactive_gift_card(staff_api_client, gift_card, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        return 10
    gift_card.is_active = False
    gift_card.save(update_fields=['is_active'])
    assert not gift_card.is_active
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = staff_api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardDeactivate']['giftCard']
    assert data['isActive'] is False
    assert not data['events']

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_deactivate_gift_card_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, gift_card, permission_manage_gift_card, permission_manage_users, permission_manage_apps, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    assert gift_card.is_active
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.id)}
    response = staff_api_client.post_graphql(DEACTIVATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardDeactivate']['giftCard']
    assert not data['isActive']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': variables['id'], 'is_active': False, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.GIFT_CARD_STATUS_CHANGED, [any_webhook], gift_card, SimpleLazyObject(lambda : staff_api_client.user))