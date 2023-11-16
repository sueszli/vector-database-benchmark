from datetime import date, timedelta
from unittest import mock
import graphene
from .....giftcard import GiftCardEvents
from .....giftcard.error_codes import GiftCardErrorCode
from .....giftcard.models import GiftCard, GiftCardEvent
from ....tests.utils import assert_no_permission, get_graphql_content
MUTATION_GIFT_CARD_BULK_ACTIVATE = '\n    mutation GiftCardBulkActivate($ids: [ID!]!) {\n        giftCardBulkActivate(ids: $ids) {\n            count\n            errors {\n                code\n                field\n            }\n        }\n    }\n'

def test_gift_card_bulk_activate_by_staff(staff_api_client, gift_card, gift_card_expiry_date, permission_manage_gift_card):
    if False:
        while True:
            i = 10
    gift_card.is_active = False
    gift_card_expiry_date.is_active = True
    gift_cards = [gift_card, gift_card_expiry_date]
    GiftCard.objects.bulk_update(gift_cards, ['is_active'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkActivate']
    assert data['count'] == len(ids)
    events = GiftCardEvent.objects.all()
    assert events.count() == 1
    assert {event.gift_card_id for event in events} == {gift_card.id}
    assert {event.type for event in events} == {GiftCardEvents.ACTIVATED}

def test_gift_card_bulk_activate_by_app(app_api_client, gift_card, gift_card_expiry_date, permission_manage_gift_card):
    if False:
        return 10
    gift_card.is_active = False
    gift_card_expiry_date.is_active = False
    gift_cards = [gift_card, gift_card_expiry_date]
    GiftCard.objects.bulk_update(gift_cards, ['is_active'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = app_api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkActivate']
    assert data['count'] == len(ids)
    events = GiftCardEvent.objects.all()
    assert events.count() == len(ids)
    assert {event.gift_card_id for event in events} == {card.id for card in gift_cards}
    assert {event.type for event in events} == {GiftCardEvents.ACTIVATED}

def test_gift_card_bulk_activate_all_cards_already_active(staff_api_client, gift_card, gift_card_expiry_date, permission_manage_gift_card):
    if False:
        for i in range(10):
            print('nop')
    gift_cards = [gift_card, gift_card_expiry_date]
    GiftCard.objects.bulk_update(gift_cards, ['is_active'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkActivate']
    assert data['count'] == len(ids)
    events = GiftCardEvent.objects.all()
    assert events.count() == 0

def test_gift_card_bulk_activate_by_customer(api_client, gift_card, gift_card_expiry_date):
    if False:
        while True:
            i = 10
    gift_card.is_active = False
    gift_card_expiry_date.is_active = False
    gift_cards = [gift_card, gift_card_expiry_date]
    GiftCard.objects.bulk_update(gift_cards, ['is_active'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables)
    assert_no_permission(response)

def test_gift_card_bulk_activate_expired_cards(staff_api_client, gift_card, gift_card_expiry_date, gift_card_created_by_staff, permission_manage_gift_card):
    if False:
        print('Hello World!')
    gift_card.is_active = False
    gift_card_expiry_date.is_active = True
    gift_card_created_by_staff.expiry_date = date.today() - timedelta(days=2)
    gift_card.expiry_date = date.today() - timedelta(days=1)
    gift_cards = [gift_card, gift_card_expiry_date, gift_card_created_by_staff]
    GiftCard.objects.bulk_update(gift_cards, ['is_active', 'expiry_date'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkActivate']
    errors = data['errors']
    assert len(errors) == 2
    expected_errors = [{'field': graphene.Node.to_global_id('GiftCard', card.pk), 'code': GiftCardErrorCode.INVALID.name} for card in [gift_card, gift_card_created_by_staff]]
    for error in errors:
        assert error in expected_errors

@mock.patch('saleor.graphql.giftcard.bulk_mutations.gift_card_bulk_activate.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_gift_card_bulk_activate_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, gift_card, gift_card_expiry_date, permission_manage_gift_card, settings):
    if False:
        print('Hello World!')
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    gift_card.is_active = False
    gift_card_expiry_date.is_active = False
    gift_cards = [gift_card, gift_card_expiry_date]
    GiftCard.objects.bulk_update(gift_cards, ['is_active'])
    ids = [graphene.Node.to_global_id('GiftCard', card.pk) for card in gift_cards]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(MUTATION_GIFT_CARD_BULK_ACTIVATE, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkActivate']
    assert data['count'] == len(ids)
    assert mocked_webhook_trigger.call_count == len(ids)