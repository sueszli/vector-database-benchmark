from ...utils import get_graphql_content
CUSTOMER_CREATE_MUTATION = '\nmutation CreateCustomer ($input: UserCreateInput!) {\n  customerCreate(\n    input: $input\n  ) {\n    errors {\n      field\n      message\n      code\n    }\n    user {\n      id\n      metadata {\n        key\n        value\n      }\n      privateMetadata {\n        key\n        value\n      }\n    }\n  }\n}\n'

def create_customer(staff_api_client, email, metadata=None, private_metadata=None, is_active=False):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': {'email': email, 'metadata': metadata, 'privateMetadata': private_metadata, 'isActive': is_active}}
    response = staff_api_client.post_graphql(CUSTOMER_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['customerCreate']['errors'] == []
    data = content['data']['customerCreate']['user']
    assert data['id'] is not None
    return data