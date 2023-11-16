import pytest
from saleor.giftcard.search import update_gift_cards_search_vector
from ....tests.utils import get_graphql_content
QUERY_GIFT_CARDS = '\n    query giftCards(\n        $sortBy: GiftCardSortingInput,\n        $filter: GiftCardFilterInput,\n        $search: String,\n    ){\n        giftCards(first: 10, filter: $filter, sortBy: $sortBy, search: $search) {\n            edges {\n                node {\n                    id\n                    code\n                }\n            }\n            totalCount\n        }\n    }\n'

@pytest.mark.parametrize(('search', 'indexes'), [('expiry', [0, 1]), ('staff_test@example.com', [2]), ('banana', [])])
def test_query_gift_cards_with_search(search, indexes, staff_api_client, gift_card, gift_card_expiry_date, gift_card_used, permission_manage_gift_card):
    if False:
        while True:
            i = 10
    gift_card_list = [gift_card, gift_card_expiry_date, gift_card_used]
    update_gift_cards_search_vector(gift_card_list)
    variables = {'search': search}
    response = staff_api_client.post_graphql(QUERY_GIFT_CARDS, variables, permissions=[permission_manage_gift_card])
    content = get_graphql_content(response)
    data = content['data']['giftCards']['edges']
    assert len(data) == len(indexes)
    assert {card['node']['code'] for card in data} == {gift_card_list[index].code for index in indexes}