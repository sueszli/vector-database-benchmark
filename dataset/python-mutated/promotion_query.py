from ...utils import get_graphql_content
PROMOTION_QUERY = '\nquery PromotionQuery($id:ID!) {\n  promotion(id:$id) {\n    id\n  }\n}\n'

def promotion_query(staff_api_client, promotion_id):
    if False:
        return 10
    variables = {'id': promotion_id}
    response = staff_api_client.post_graphql(PROMOTION_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']
    return data