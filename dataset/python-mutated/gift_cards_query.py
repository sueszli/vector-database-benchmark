from ...utils import get_graphql_content
GIFT_CARDS_QUERY = '\nquery GiftCards($first: Int) {\n  giftCards(first: $first) {\n    edges {\n      node {\n        id\n        code\n        initialBalance {\n          amount\n        }\n      }\n    }\n  }\n}\n'

def get_gift_cards(staff_api_client, first=10):
    if False:
        for i in range(10):
            print('nop')
    variables = {'first': first}
    response = staff_api_client.post_graphql(GIFT_CARDS_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['giftCards']['edges']
    return data