import graphene
import pytest
from .....giftcard.models import GiftCard
from ....tests.utils import get_graphql_content
FRAGMENT_EVENTS = '\n    fragment GiftCardEvents on GiftCardEvent {\n        id\n        date\n        type\n        user {\n            email\n        }\n        app {\n            name\n        }\n        message\n        email\n        orderId\n        orderNumber\n        tags\n        oldTags\n        balance {\n            initialBalance {\n                amount\n                currency\n            }\n            oldInitialBalance {\n                amount\n                currency\n            }\n            currentBalance {\n                amount\n                currency\n            }\n            oldCurrentBalance {\n                amount\n                currency\n            }\n        }\n        expiryDate\n        oldExpiryDate\n    }\n'
FRAGMENT_GIFT_CARD_DETAILS = FRAGMENT_EVENTS + '\n        fragment GiftCardDetails on GiftCard {\n            id\n            code\n            last4CodeChars\n            isActive\n            expiryDate\n            tags {\n                name\n            }\n            created\n            lastUsedOn\n            boughtInChannel\n            initialBalance {\n                currency\n                amount\n            }\n            currentBalance {\n                currency\n                amount\n            }\n            createdBy {\n                email\n            }\n            usedBy {\n                email\n            }\n            createdByEmail\n            usedByEmail\n            app {\n                name\n            }\n            product {\n                name\n            }\n            events {\n                ...GiftCardEvents\n            }\n        }\n    '

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_query_gift_card_details(staff_api_client, gift_card, gift_card_event, permission_manage_gift_card, permission_manage_users, permission_manage_apps, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = FRAGMENT_GIFT_CARD_DETAILS + '\n        query giftCard($id: ID!) {\n            giftCard(id: $id){\n                ...GiftCardDetails\n            }\n        }\n    '
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables, permissions=[permission_manage_gift_card, permission_manage_users, permission_manage_apps]))
    assert content['data']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_query_gift_cards(staff_api_client, gift_cards_for_benchmarks, permission_manage_gift_card, permission_manage_apps, permission_manage_users, count_queries):
    if False:
        while True:
            i = 10
    query = FRAGMENT_GIFT_CARD_DETAILS + '\n        query {\n            giftCards(first: 20){\n                edges {\n                    node {\n                        ...GiftCardDetails\n                    }\n                }\n            }\n        }\n    '
    content = get_graphql_content(staff_api_client.post_graphql(query, {}, permissions=[permission_manage_gift_card, permission_manage_apps, permission_manage_users]))
    assert content['data']
    assert len(content['data']['giftCards']['edges']) == len(gift_cards_for_benchmarks)

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_filter_gift_cards_by_tags(staff_api_client, gift_cards_for_benchmarks, permission_manage_gift_card, permission_manage_apps, permission_manage_users, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = FRAGMENT_GIFT_CARD_DETAILS + '\n        query giftCards($filter: GiftCardFilterInput){\n            giftCards(first: 20, filter: $filter) {\n                edges {\n                    node {\n                        ...GiftCardDetails\n                    }\n                }\n            }\n        }\n    '
    content = get_graphql_content(staff_api_client.post_graphql(query, {'filter': {'tags': ['benchmark-test-tag']}}, permissions=[permission_manage_gift_card, permission_manage_apps, permission_manage_users]))
    assert content['data']
    assert len(content['data']['giftCards']['edges']) == len(gift_cards_for_benchmarks)

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_filter_gift_cards_by_used_by_user(staff_api_client, customer_user, gift_cards_for_benchmarks, permission_manage_gift_card, permission_manage_apps, permission_manage_users, count_queries):
    if False:
        for i in range(10):
            print('nop')
    cards_to_update = gift_cards_for_benchmarks[:10]
    for card in cards_to_update:
        card.used_by = customer_user
    GiftCard.objects.bulk_update(cards_to_update, ['used_by'])
    query = FRAGMENT_GIFT_CARD_DETAILS + '\n        query giftCards($filter: GiftCardFilterInput){\n            giftCards(first: 20, filter: $filter) {\n                edges {\n                    node {\n                        ...GiftCardDetails\n                    }\n                }\n            }\n        }\n    '
    content = get_graphql_content(staff_api_client.post_graphql(query, {'filter': {'usedBy': [graphene.Node.to_global_id('User', customer_user.pk)]}}, permissions=[permission_manage_gift_card, permission_manage_apps, permission_manage_users]))
    assert content['data']
    assert len(content['data']['giftCards']['edges']) == len(cards_to_update)

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_filter_gift_cards_by_products(staff_api_client, shippable_gift_card_product, gift_cards_for_benchmarks, permission_manage_gift_card, permission_manage_apps, permission_manage_users, count_queries):
    if False:
        for i in range(10):
            print('nop')
    cards_to_update = gift_cards_for_benchmarks[:10]
    for card in cards_to_update:
        card.product = shippable_gift_card_product
    GiftCard.objects.bulk_update(cards_to_update, ['product'])
    query = FRAGMENT_GIFT_CARD_DETAILS + '\n        query giftCards($filter: GiftCardFilterInput){\n            giftCards(first: 20, filter: $filter) {\n                edges {\n                    node {\n                        ...GiftCardDetails\n                    }\n                }\n            }\n        }\n    '
    variables = {'filter': {'products': [graphene.Node.to_global_id('Product', shippable_gift_card_product.pk)]}}
    content = get_graphql_content(staff_api_client.post_graphql(query, variables, permissions=[permission_manage_gift_card, permission_manage_apps, permission_manage_users]))
    assert content['data']
    assert len(content['data']['giftCards']['edges']) == len(cards_to_update)