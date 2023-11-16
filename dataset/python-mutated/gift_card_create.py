from ...utils import get_graphql_content
GIFT_CARD_CREATE_MUTATION = '\nmutation GiftCardCreate($input: GiftCardCreateInput!) {\n  giftCardCreate(input: $input) {\n    giftCard {\n      id\n      code\n      initialBalance {\n        amount\n      }\n    }\n    errors {\n      code\n      field\n      message\n    }\n  }\n}\n'

def create_gift_card(staff_api_client, amount, currency='USD', active=True, email=None, channel=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': {'note': 'note', 'addTags': ['tag_test'], 'userEmail': email, 'channel': channel, 'balance': {'amount': amount, 'currency': currency}, 'isActive': active}}
    response = staff_api_client.post_graphql(GIFT_CARD_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['giftCardCreate']['errors'] == []
    data = content['data']['giftCardCreate']['giftCard']
    assert data['id'] is not None
    assert data['code'] is not None
    return data