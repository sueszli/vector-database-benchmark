from ...utils import get_graphql_content
PROMOTION_CREATE_MUTATION = '\nmutation CreatePromotion($input: PromotionCreateInput!) {\n  promotionCreate(input: $input) {\n    errors {\n      message\n      field\n      code\n    }\n    promotion {\n      id\n      name\n      startDate\n      endDate\n      description\n      createdAt\n      metadata {\n        key\n        value\n      }\n      privateMetadata {\n        key\n        value\n      }\n    }\n  }\n}\n'

def create_promotion(staff_api_client, promotion_name, start_date=None, end_date=None, description=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': {'name': promotion_name, 'startDate': start_date, 'endDate': end_date, 'description': description}}
    response = staff_api_client.post_graphql(PROMOTION_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['promotionCreate']['errors'] == []
    data = content['data']['promotionCreate']['promotion']
    assert data['id'] is not None
    return data