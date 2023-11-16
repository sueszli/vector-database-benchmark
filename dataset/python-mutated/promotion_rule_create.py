from ...utils import get_graphql_content
PROMOTION_RULE_CREATE_MUTATION = '\nmutation promotionRuleCreate($input: PromotionRuleCreateInput!) {\n  promotionRuleCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    promotionRule {\n      id\n      name\n      description\n      rewardValueType\n      rewardValue\n      cataloguePredicate\n      channels{\n        id\n      }\n    }\n  }\n}\n'

def create_promotion_rule(staff_api_client, promotion_id, catalogue_predicate, reward_value_type='PERCENTAGE', reward_value=5.0, promotion_rule_name='Test rule', channel_id=None, description=None):
    if False:
        while True:
            i = 10
    if not channel_id:
        channel_id = []
    variables = {'input': {'promotion': promotion_id, 'name': promotion_rule_name, 'rewardValueType': reward_value_type, 'rewardValue': reward_value, 'channels': channel_id, 'cataloguePredicate': catalogue_predicate, 'description': description}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['promotionRuleCreate']['errors'] == []
    data = content['data']['promotionRuleCreate']['promotionRule']
    assert data['id'] is not None
    assert data['name'] == promotion_rule_name
    assert data['rewardValueType'] == reward_value_type
    assert data['rewardValue'] == reward_value
    return data