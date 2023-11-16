from ....tests.utils import assert_no_permission, get_graphql_content
QUERY_GIFT_CARD_TAGS = '\n    query giftCardTags{\n        giftCardTags(first: 10) {\n            edges {\n                node {\n                    id\n                    name\n                }\n            }\n            totalCount\n        }\n    }\n'

def test_query_gift_card_tags_by_staff(staff_api_client, gift_card_tag_list, permission_manage_gift_card):
    if False:
        return 10
    query = QUERY_GIFT_CARD_TAGS
    response = staff_api_client.post_graphql(query, permissions=[permission_manage_gift_card])
    content = get_graphql_content(response)
    data = content['data']['giftCardTags']['edges']
    assert len(data) == len(gift_card_tag_list)
    assert {tag['node']['name'] for tag in data} == {tag.name for tag in gift_card_tag_list}

def test_query_gift_card_tags_by_app(app_api_client, gift_card_tag_list, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    query = QUERY_GIFT_CARD_TAGS
    response = app_api_client.post_graphql(query, permissions=[permission_manage_gift_card])
    content = get_graphql_content(response)
    data = content['data']['giftCardTags']['edges']
    assert len(data) == len(gift_card_tag_list)
    assert {tag['node']['name'] for tag in data} == {tag.name for tag in gift_card_tag_list}

def test_query_gift_card_tags_by_customer(api_client, gift_card_tag_list):
    if False:
        print('Hello World!')
    query = QUERY_GIFT_CARD_TAGS
    response = api_client.post_graphql(query)
    assert_no_permission(response)