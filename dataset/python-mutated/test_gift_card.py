import graphene
from ....tests.utils import assert_no_permission, get_graphql_content
from .utils import PRIVATE_KEY, PRIVATE_VALUE, PUBLIC_KEY, PUBLIC_VALUE
QUERY_GIFT_CARD_PRIVATE_META = '\n    query giftCardMeta($id: ID!){\n        giftCard(id: $id){\n            privateMetadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_private_meta_for_gift_card_as_anonymous_user(api_client, gift_card):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_gift_card_as_customer(user_api_client, gift_card):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = user_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables)
    assert_no_permission(response)

def test_query_private_meta_for_gift_card_as_staff(staff_api_client, gift_card, permission_manage_gift_card):
    if False:
        for i in range(10):
            print('nop')
    gift_card.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    gift_card.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = staff_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE

def test_query_private_meta_for_gift_card_as_app(app_api_client, gift_card, permission_manage_gift_card):
    if False:
        return 10
    gift_card.store_value_in_private_metadata({PRIVATE_KEY: PRIVATE_VALUE})
    gift_card.save(update_fields=['private_metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = app_api_client.post_graphql(QUERY_GIFT_CARD_PRIVATE_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['privateMetadata'][0]
    assert metadata['key'] == PRIVATE_KEY
    assert metadata['value'] == PRIVATE_VALUE
QUERY_GIFT_CARD_PUBLIC_META = '\n    query giftCardMeta($id: ID!){\n        giftCard(id: $id){\n            metadata{\n                key\n                value\n            }\n        }\n    }\n'

def test_query_public_meta_for_gift_card_as_anonymous_user(api_client, gift_card):
    if False:
        print('Hello World!')
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_gift_card_as_customer(user_api_client, gift_card):
    if False:
        print('Hello World!')
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = user_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables)
    assert_no_permission(response)

def test_query_public_meta_for_gift_card_as_staff(staff_api_client, gift_card, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = staff_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE

def test_query_public_meta_for_gift_card_as_app(app_api_client, gift_card, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    gift_card.store_value_in_metadata({PUBLIC_KEY: PUBLIC_VALUE})
    gift_card.save(update_fields=['metadata'])
    variables = {'id': graphene.Node.to_global_id('GiftCard', gift_card.pk)}
    response = app_api_client.post_graphql(QUERY_GIFT_CARD_PUBLIC_META, variables, [permission_manage_gift_card], check_no_permissions=False)
    content = get_graphql_content(response)
    metadata = content['data']['giftCard']['metadata'][0]
    assert metadata['key'] == PUBLIC_KEY
    assert metadata['value'] == PUBLIC_VALUE