from ...utils import get_graphql_content
GIFT_CARD_BULK_CREATE_MUTATION = '\nmutation GiftCardBulkCreate($input: GiftCardBulkCreateInput!) {\n  giftCardBulkCreate(input: $input) {\n    giftCards {\n      id\n      code\n      initialBalance{\n        amount\n      }\n    }\n    errors {\n      code\n      field\n      message\n    }\n  }\n}\n'

def bulk_create_gift_card(staff_api_client, cards_amount, amount, currency='USD', active=True):
    if False:
        return 10
    variables = {'input': {'count': cards_amount, 'tags': ['tag_test'], 'balance': {'amount': amount, 'currency': currency}, 'isActive': active}}
    response = staff_api_client.post_graphql(GIFT_CARD_BULK_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['giftCardBulkCreate']['errors'] == []
    data = content['data']['giftCardBulkCreate']['giftCards']
    assert data is not None
    assert len(data) == cards_amount
    return data