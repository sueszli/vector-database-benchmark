from ...utils import get_graphql_content
ME_QUERY = '\nquery Me{\n  me{\n    id\n    orders(first:10){\n      edges{\n        node{\n          number\n          status\n        }\n      }\n    }\n  }\n}\n'

def get_own_data(api_client):
    if False:
        return 10
    variables = {}
    response = api_client.post_graphql(ME_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['me']
    assert data is not None
    return data