import pytest
from ....tests.utils import get_graphql_content
QUERY_GIFT_CARD_TAGS = '\n    query giftCardTags($filter: GiftCardTagFilterInput){\n        giftCardTags(first: 10, filter:  $filter) {\n            edges {\n                node {\n                    id\n                    name\n                }\n            }\n            totalCount\n        }\n    }\n'

@pytest.mark.parametrize(('value', 'expected_tag_indexes'), [('tag', [0, 1, 2, 3, 4]), ('test-tag', [0, 1, 2, 3, 4]), ('test', [0, 1, 2, 3, 4]), ('0', [0]), ('tag-3', [3])])
def test_filter_gift_card_tags_by_name(staff_api_client, gift_card_tag_list, permission_manage_gift_card, value, expected_tag_indexes):
    if False:
        while True:
            i = 10
    query = QUERY_GIFT_CARD_TAGS
    variables = {'filter': {'search': value}}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_gift_card])
    content = get_graphql_content(response)
    data = content['data']['giftCardTags']['edges']
    assert len(data) == len(expected_tag_indexes)
    assert {tag['node']['name'] for tag in data} == {gift_card_tag_list[index].name for index in expected_tag_indexes}