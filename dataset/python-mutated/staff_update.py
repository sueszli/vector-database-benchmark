from ...utils import get_graphql_content
STAFF_UPDATE_MUTATION = '\nmutation StaffUpdate($id: ID!, $input: StaffUpdateInput!){\n  staffUpdate(id: $id,\n    input: $input) {\n    user {\n      id\n      metadata {\n        key\n        value\n      }\n      privateMetadata {\n        key\n        value\n      }\n    }\n    errors {\n      field\n      message\n    }\n  }\n}\n'

def update_staff(staff_api_client, staff_id, input_data):
    if False:
        while True:
            i = 10
    variables = {'id': staff_id, 'input': input_data}
    response = staff_api_client.post_graphql(STAFF_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['staffUpdate']['errors'] == []
    data = content['data']['staffUpdate']['user']
    return data