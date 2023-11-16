import json
from datetime import date, timedelta
from unittest import mock
import graphene
import pytest
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....giftcard import GiftCardEvents
from .....giftcard.error_codes import GiftCardErrorCode
from .....giftcard.models import GiftCard
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....tests.utils import assert_no_permission, get_graphql_content
CREATE_GIFT_CARD_MUTATION = '\n    mutation giftCardCreate($input: GiftCardCreateInput!){\n        giftCardCreate(input: $input) {\n            giftCard {\n                id\n                code\n                last4CodeChars\n                isActive\n                expiryDate\n                tags {\n                    name\n                }\n                created\n                lastUsedOn\n                initialBalance {\n                    currency\n                    amount\n                }\n                currentBalance {\n                    currency\n                    amount\n                }\n                createdBy {\n                    email\n                }\n                usedBy {\n                    email\n                }\n                createdByEmail\n                usedByEmail\n                app {\n                    name\n                }\n                product {\n                    name\n                }\n                events {\n                    type\n                    message\n                    user {\n                        email\n                    }\n                    app {\n                        name\n                    }\n                    balance {\n                        initialBalance {\n                            amount\n                            currency\n                        }\n                        oldInitialBalance {\n                            amount\n                            currency\n                        }\n                        currentBalance {\n                            amount\n                            currency\n                        }\n                        oldCurrentBalance {\n                            amount\n                            currency\n                        }\n                    }\n                }\n            }\n            errors {\n                field\n                message\n                code\n            }\n        }\n    }\n'

@mock.patch('saleor.graphql.giftcard.mutations.gift_card_create.send_gift_card_notification')
def test_create_never_expiry_gift_card(send_notification_mock, staff_api_client, customer_user, channel_USD, permission_manage_gift_card, permission_manage_users, permission_manage_apps, gift_card_tag_list):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    new_tag = 'gift-card-tag'
    existing_tag_name = gift_card_tag_list[0].name
    tags = [new_tag, existing_tag_name]
    note = 'This is gift card note that will be save in gift card event.'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'channel': channel_USD.slug, 'addTags': tags, 'note': note, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code']
    assert data['last4CodeChars']
    assert not data['expiryDate']
    assert len(data['tags']) == 2
    assert {tag['name'] for tag in data['tags']} == set(tags)
    assert data['createdBy']['email'] == staff_api_client.user.email
    assert data['createdByEmail'] == staff_api_client.user.email
    assert not data['usedBy']
    assert not data['usedByEmail']
    assert not data['app']
    assert not data['lastUsedOn']
    assert data['isActive']
    assert data['initialBalance']['amount'] == initial_balance
    assert data['currentBalance']['amount'] == initial_balance
    assert len(data['events']) == 2
    (created_event, note_added) = data['events']
    assert created_event['type'] == GiftCardEvents.ISSUED.upper()
    assert created_event['user']['email'] == staff_api_client.user.email
    assert not created_event['app']
    assert created_event['balance']['initialBalance']['amount'] == initial_balance
    assert created_event['balance']['initialBalance']['currency'] == currency
    assert created_event['balance']['currentBalance']['amount'] == initial_balance
    assert created_event['balance']['currentBalance']['currency'] == currency
    assert not created_event['balance']['oldInitialBalance']
    assert not created_event['balance']['oldCurrentBalance']
    assert note_added['type'] == GiftCardEvents.NOTE_ADDED.upper()
    assert note_added['user']['email'] == staff_api_client.user.email
    assert not note_added['app']
    assert note_added['message'] == note
    gift_card = GiftCard.objects.get()
    send_notification_mock.assert_called_once_with(staff_api_client.user, None, customer_user, customer_user.email, gift_card, mock.ANY, channel_slug=channel_USD.slug, resending=False)

@mock.patch('saleor.graphql.giftcard.mutations.gift_card_create.send_gift_card_notification')
def test_create_gift_card_by_app(send_notification_mock, app_api_client, permission_manage_gift_card, permission_manage_users):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    note = 'This is gift card note that will be save in gift card event.'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'addTags': [tag], 'note': note, 'expiryDate': None, 'isActive': False}}
    response = app_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code']
    assert data['last4CodeChars']
    assert not data['expiryDate']
    assert len(data['tags']) == 1
    assert data['tags'][0]['name'] == tag
    assert not data['createdBy']
    assert not data['createdByEmail']
    assert not data['usedBy']
    assert not data['usedByEmail']
    assert data['app']['name'] == app_api_client.app.name
    assert not data['lastUsedOn']
    assert data['isActive'] is False
    assert data['initialBalance']['amount'] == initial_balance
    assert data['currentBalance']['amount'] == initial_balance
    assert len(data['events']) == 2
    (created_event, note_added) = data['events']
    assert created_event['type'] == GiftCardEvents.ISSUED.upper()
    assert not created_event['user']
    assert created_event['app']['name'] == app_api_client.app.name
    assert created_event['balance']['initialBalance']['amount'] == initial_balance
    assert created_event['balance']['initialBalance']['currency'] == currency
    assert created_event['balance']['currentBalance']['amount'] == initial_balance
    assert created_event['balance']['currentBalance']['currency'] == currency
    assert not created_event['balance']['oldInitialBalance']
    assert not created_event['balance']['oldCurrentBalance']
    assert note_added['type'] == GiftCardEvents.NOTE_ADDED.upper()
    assert not note_added['user']
    assert note_added['app']['name'] == app_api_client.app.name
    assert note_added['message'] == note
    send_notification_mock.assert_not_called()

def test_create_gift_card_by_customer(api_client, customer_user, channel_USD):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'channel': channel_USD.slug, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'expiryDate': None, 'isActive': True}}
    response = api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables)
    assert_no_permission(response)

def test_create_gift_card_no_premissions(staff_api_client):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'expiryDate': None, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables)
    assert_no_permission(response)

def test_create_gift_card_with_too_many_decimal_places_in_balance_amount(staff_api_client, customer_user, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        return 10
    initial_balance = 10.123
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not data
    assert len(errors) == 1
    assert errors[0]['field'] == 'balance'
    assert errors[0]['code'] == GiftCardErrorCode.INVALID.name

def test_create_gift_card_with_malformed_email(staff_api_client, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        for i in range(10):
            print('nop')
    initial_balance = 10
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': 'malformed', 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardCreate']['giftCard']
    errors = content['data']['giftCardCreate']['errors']
    assert not data
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'email'
    assert error['code'] == GiftCardErrorCode.INVALID.name

def test_create_gift_card_lack_of_channel(staff_api_client, customer_user, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        print('Hello World!')
    initial_balance = 10
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['giftCardCreate']['giftCard']
    errors = content['data']['giftCardCreate']['errors']
    assert not data
    assert len(errors) == 1
    error = errors[0]
    assert error['field'] == 'channel'
    assert error['code'] == GiftCardErrorCode.REQUIRED.name

def test_create_gift_card_with_zero_balance_amount(staff_api_client, customer_user, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        for i in range(10):
            print('nop')
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': 0, 'currency': currency}, 'userEmail': customer_user.email, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not data
    assert len(errors) == 1
    assert errors[0]['field'] == 'balance'
    assert errors[0]['code'] == GiftCardErrorCode.INVALID.name

@mock.patch('saleor.graphql.giftcard.mutations.gift_card_create.send_gift_card_notification')
def test_create_gift_card_with_expiry_date(send_notification_mock, staff_api_client, customer_user, channel_USD, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    date_value = date.today() + timedelta(days=365)
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'channel': channel_USD.slug, 'addTags': [tag], 'expiryDate': date_value, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code']
    assert data['last4CodeChars']
    assert data['expiryDate'] == date_value.isoformat()
    assert len(data['events']) == 1
    created_event = data['events'][0]
    assert created_event['type'] == GiftCardEvents.ISSUED.upper()
    assert created_event['user']['email'] == staff_api_client.user.email
    assert not created_event['app']
    assert created_event['balance']['initialBalance']['amount'] == initial_balance
    assert created_event['balance']['initialBalance']['currency'] == currency
    assert created_event['balance']['currentBalance']['amount'] == initial_balance
    assert created_event['balance']['currentBalance']['currency'] == currency
    assert not created_event['balance']['oldInitialBalance']
    assert not created_event['balance']['oldCurrentBalance']
    gift_card = GiftCard.objects.get()
    send_notification_mock.assert_called_once_with(staff_api_client.user, None, customer_user, customer_user.email, gift_card, mock.ANY, channel_slug=channel_USD.slug, resending=False)

@pytest.mark.parametrize('date_value', [date(1999, 1, 1), date.today()])
def test_create_gift_card_with_expiry_date_type_invalid(date_value, staff_api_client, customer_user, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        i = 10
        return i + 15
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'userEmail': customer_user.email, 'addTags': [tag], 'note': 'This is gift card note that will be save in gift card event.', 'expiryDate': date_value, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not data
    assert len(errors) == 1
    assert errors[0]['field'] == 'expiryDate'
    assert errors[0]['code'] == GiftCardErrorCode.INVALID.name

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_create_gift_card_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, app_api_client, permission_manage_gift_card, permission_manage_users, settings):
    if False:
        for i in range(10):
            print('nop')
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    note = 'This is gift card note that will be save in gift card event.'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'addTags': [tag], 'note': note, 'expiryDate': None, 'isActive': False}}
    response = app_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users])
    gift_card = GiftCard.objects.last()
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code']
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': graphene.Node.to_global_id('GiftCard', gift_card.id), 'is_active': gift_card.is_active, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : app_api_client.app)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.GIFT_CARD_CREATED, [any_webhook], gift_card, SimpleLazyObject(lambda : app_api_client.app))

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_create_gift_card_with_email_triggers_gift_card_sent_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, app_api_client, channel_USD, customer_user, permission_manage_gift_card, permission_manage_users, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    initial_balance = 100
    currency = 'USD'
    tag = 'gift-card-tag'
    note = 'This is gift card note that will be save in gift card event.'
    variables = {'input': {'balance': {'amount': initial_balance, 'currency': currency}, 'addTags': [tag], 'note': note, 'expiryDate': None, 'isActive': False, 'channel': channel_USD.slug, 'userEmail': customer_user.email}}
    response = app_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users])
    gift_card = GiftCard.objects.last()
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code']
    mocked_webhook_trigger.assert_any_call(json.dumps({'id': graphene.Node.to_global_id('GiftCard', gift_card.id), 'is_active': gift_card.is_active, 'channel_slug': channel_USD.slug, 'sent_to_email': customer_user.email, 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : app_api_client.app)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.GIFT_CARD_SENT, [any_webhook], {'gift_card': gift_card, 'channel_slug': channel_USD.slug, 'sent_to_email': customer_user.email}, SimpleLazyObject(lambda : app_api_client.app))

def test_create_gift_card_with_code(staff_api_client, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        for i in range(10):
            print('nop')
    code = 'custom-code'
    variables = {'input': {'balance': {'amount': 1, 'currency': 'USD'}, 'code': code, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not errors
    assert data['code'] == code

def test_create_gift_card_with_to_short_code(staff_api_client, permission_manage_gift_card, permission_manage_users, permission_manage_apps):
    if False:
        while True:
            i = 10
    code = 'short'
    variables = {'input': {'balance': {'amount': 1, 'currency': 'USD'}, 'code': code, 'isActive': True}}
    response = staff_api_client.post_graphql(CREATE_GIFT_CARD_MUTATION, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps])
    content = get_graphql_content(response)
    errors = content['data']['giftCardCreate']['errors']
    data = content['data']['giftCardCreate']['giftCard']
    assert not data
    assert len(errors) == 1
    assert errors[0]['field'] == 'code'
    assert errors[0]['code'] == 'INVALID'