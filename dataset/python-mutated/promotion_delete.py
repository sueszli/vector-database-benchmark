from ...utils import get_graphql_content
PROMOTION_DELETE_MUTATION = '\nmutation DeletePromotion($id:ID!) {\n  promotionDelete(id: $id) {\n    errors {\n      code\n      field\n      message\n    }\n    promotion {\n      id\n    }\n  }\n}\n'

def delete_promotion(staff_api_client, promotion_id):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': promotion_id}
    response = staff_api_client.post_graphql(PROMOTION_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['promotionDelete']
    assert data['errors'] == []
    return data