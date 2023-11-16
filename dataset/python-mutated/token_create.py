from ...utils import get_graphql_content
TOKEN_CREATE_MUTATION = '\nmutation TokenCreate($email: String!, $password: String!) {\n  tokenCreate(email: $email, password: $password) {\n    errors {\n      field\n      message\n      code\n    }\n    token\n    refreshToken\n    user {\n      id\n      email\n      isActive\n      isConfirmed\n    }\n  }\n}\n'

def raw_token_create(e2e_not_logged_api_client, email, password):
    if False:
        for i in range(10):
            print('nop')
    variables = {'email': email, 'password': password}
    response = e2e_not_logged_api_client.post_graphql(TOKEN_CREATE_MUTATION, variables)
    content = get_graphql_content(response, ignore_errors=True)
    return content

def token_create(e2e_not_logged_api_client, email, password):
    if False:
        while True:
            i = 10
    response = raw_token_create(e2e_not_logged_api_client, email, password)
    data = response['data']['tokenCreate']
    assert data['errors'] == []
    assert data['token'] is not None
    assert data['refreshToken'] is not None
    user = data['user']
    assert user['id'] is not None
    assert user['email'] == email
    return data