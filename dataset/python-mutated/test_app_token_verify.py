from ....tests.utils import get_graphql_content
APP_TOKEN_VERIFY_MUTATION = '\nmutation AppTokenVerify($token: String!){\n    appTokenVerify(token:$token){\n        valid\n    }\n}\n'

def test_app_token_verify_valid_token(app, api_client):
    if False:
        for i in range(10):
            print('nop')
    (_, token) = app.tokens.create()
    query = APP_TOKEN_VERIFY_MUTATION
    variables = {'token': token}
    response = api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    assert content['data']['appTokenVerify']['valid']

def test_app_token_verify_invalid_token(app, api_client):
    if False:
        i = 10
        return i + 15
    (_, token) = app.tokens.create()
    token += 'incorrect'
    query = APP_TOKEN_VERIFY_MUTATION
    variables = {'token': token}
    response = api_client.post_graphql(query, variables=variables)
    content = get_graphql_content(response)
    assert not content['data']['appTokenVerify']['valid']