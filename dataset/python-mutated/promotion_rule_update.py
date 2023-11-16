from ...utils import get_graphql_content
PROMOTION_RULE_UPDATE_MUTATION = '\nmutation promotionRuleCreate($id: ID!, $input: PromotionRuleUpdateInput!) {\n  promotionRuleUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    promotionRule {\n      id\n      name\n      description\n      rewardValueType\n      rewardValue\n      cataloguePredicate\n      channels {\n        id\n      }\n    }\n  }\n}\n'

def update_promotion_rule(staff_api_client, promotion_rule_id, input):
    if False:
        print('Hello World!')
    variables = {'id': promotion_rule_id, 'input': input}
    response = staff_api_client.post_graphql(PROMOTION_RULE_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['promotionRuleUpdate']['errors'] == []
    data = content['data']['promotionRuleUpdate']['promotionRule']
    return data