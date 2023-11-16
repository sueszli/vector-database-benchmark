from ...utils import get_graphql_content
USER_QUERY = '\nquery CustomerDetails($id:ID!){\n  user(id: $id) {\n    id\n    email\n    isConfirmed\n    isActive\n    orders(first: 10){\n      edges {\n        node {\n          id\n          number\n          paymentStatus\n          created\n        }\n      }\n    }\n  }\n}\n'

def get_user(staff_api_client, user_id):
    if False:
        print('Hello World!')
    variables = {'id': user_id}
    response = staff_api_client.post_graphql(USER_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['user']
    return data