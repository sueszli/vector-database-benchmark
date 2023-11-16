from unittest import mock
import graphene
from .....giftcard.models import GiftCard
from ....tests.utils import assert_no_permission, get_graphql_content
GIFT_CARD_BULK_DELETE_MUTATION = '\n    mutation GiftCardBulkDelete($ids: [ID!]!) {\n        giftCardBulkDelete(ids: $ids) {\n            count\n            errors {\n                code\n                field\n                message\n            }\n        }\n    }\n'

def test_gift_card_bulk_delete_by_staff(staff_api_client, permission_manage_gift_card, gift_card, gift_card_expiry_date):
    if False:
        print('Hello World!')
    gift_card_pks = [gift_card.pk, gift_card_expiry_date.pk]
    ids = [graphene.Node.to_global_id('GiftCard', pk) for pk in gift_card_pks]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(GIFT_CARD_BULK_DELETE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkDelete']
    assert not data['errors']
    assert data['count'] == len(ids)
    assert not GiftCard.objects.filter(id__in=gift_card_pks)

def test_gift_card_bulk_delete_by_app(app_api_client, permission_manage_gift_card, gift_card, gift_card_expiry_date):
    if False:
        print('Hello World!')
    gift_card_pks = [gift_card.pk, gift_card_expiry_date.pk]
    ids = [graphene.Node.to_global_id('GiftCard', pk) for pk in gift_card_pks]
    variables = {'ids': ids}
    response = app_api_client.post_graphql(GIFT_CARD_BULK_DELETE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkDelete']
    assert not data['errors']
    assert data['count'] == len(ids)
    assert not GiftCard.objects.filter(id__in=gift_card_pks)

def test_gift_card_bulk_delete_by_customer(app_api_client, gift_card, gift_card_expiry_date):
    if False:
        print('Hello World!')
    gift_card_pks = [gift_card.pk, gift_card_expiry_date.pk]
    ids = [graphene.Node.to_global_id('GiftCard', pk) for pk in gift_card_pks]
    variables = {'ids': ids}
    response = app_api_client.post_graphql(GIFT_CARD_BULK_DELETE_MUTATION, variables)
    assert_no_permission(response)

@mock.patch('saleor.graphql.giftcard.bulk_mutations.gift_card_bulk_delete.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_gift_card_bulk_delete_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, permission_manage_gift_card, gift_card, gift_card_expiry_date, settings):
    if False:
        for i in range(10):
            print('nop')
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    gift_card_pks = [gift_card.pk, gift_card_expiry_date.pk]
    ids = [graphene.Node.to_global_id('GiftCard', pk) for pk in gift_card_pks]
    variables = {'ids': ids}
    response = staff_api_client.post_graphql(GIFT_CARD_BULK_DELETE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardBulkDelete']
    assert not data['errors']
    assert not GiftCard.objects.filter(id__in=gift_card_pks)
    assert mocked_webhook_trigger.call_count == len(gift_card_pks)