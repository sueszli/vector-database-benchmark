from ...utils import get_graphql_content
CUSTOMER_UPDATE_MUTATION = '\nmutation CustomerUpdate($id: ID!, $input: CustomerInput!) {\n  customerUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    user {\n      id\n      email\n      isActive\n      isConfirmed\n      isStaff\n      metadata {\n        key\n        value\n      }\n      privateMetadata {\n        key\n        value\n      }\n    }\n  }\n}\n'

def customer_update(api_client, user_id, input_data):
    if False:
        print('Hello World!')
    variables = {'id': user_id, 'input': input_data}
    response = api_client.post_graphql(CUSTOMER_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['customerUpdate']
    assert data['errors'] == []
    user_data = data['user']
    return user_data