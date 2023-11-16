from ...utils import get_graphql_content
CHECKOUT_EMAIL_UPDATE_MUTATION = '\nmutation checkoutEmailUpdate($checkoutId: ID!, $email: String!){\n  checkoutEmailUpdate(\n    id: $checkoutId\n    email: $email\n  ) {\n    checkout {\n      email\n    }\n    errors {\n      field\n      message\n    }\n  }\n}\n'

def checkout_update_email(staff_api_client, checkout_id, email):
    if False:
        while True:
            i = 10
    variables = {'checkoutId': checkout_id, 'email': email}
    response = staff_api_client.post_graphql(CHECKOUT_EMAIL_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['checkoutEmailUpdate']['errors'] == []
    data = content['data']['checkoutEmailUpdate']['checkout']
    return data