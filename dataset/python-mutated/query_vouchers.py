from ...utils import get_graphql_content
VOUCHERS_QUERY = '\nquery vouchersQuery{\n    vouchers(first: 10) {\n    edges {\n      node {\n        id\n      }\n    }\n  }\n}\n'

def get_vouchers(api_client):
    if False:
        while True:
            i = 10
    response = api_client.post_graphql(VOUCHERS_QUERY)
    content = get_graphql_content(response)
    return content['data']