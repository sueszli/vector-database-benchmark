from ...utils import get_graphql_content
STAFF_CREATE_MUTATION = '\nmutation CreateStaff($input: StaffCreateInput!){\n  staffCreate(input:$input) {\n    user {\n      id\n      metadata {\n        key\n        value\n      }\n      privateMetadata {\n        key\n        value\n      }\n    }\n    errors {\n      field\n      message\n    }\n  }\n}\n'

def create_staff(staff_api_client, input_data):
    if False:
        return 10
    variables = {'input': input_data}
    response = staff_api_client.post_graphql(STAFF_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['staffCreate']['errors'] == []
    data = content['data']['staffCreate']['user']
    return data