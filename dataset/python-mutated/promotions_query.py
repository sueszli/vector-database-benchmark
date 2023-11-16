from ...utils import get_graphql_content
PROMOTIONS_QUERY = '\nquery Promotions($first: Int, $sortBy: PromotionSortingInput,\n$where: PromotionWhereInput) {\n  promotions(first: $first, sortBy: $sortBy, where: $where) {\n    totalCount\n    edges {\n      node {\n        id\n        events {\n          __typename\n        }\n        name\n        createdAt\n        startDate\n        endDate\n        metadata {\n          key\n          value\n        }\n        privateMetadata {\n          key\n          value\n        }\n        rules {\n          id\n          name\n          description\n          rewardValueType\n          cataloguePredicate\n          rewardValue\n          channels {\n            name\n          }\n        }\n      }\n    }\n  }\n}\n'

def promotions_query(staff_api_client, first=10, sort_by=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'first': first, 'sortBy': sort_by, 'where': where}
    response = staff_api_client.post_graphql(PROMOTIONS_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['promotions']['edges']
    return data