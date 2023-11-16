from ...utils import get_graphql_content
ORDER_VOID_MUTATION = '\nmutation VoidOrder ($id:ID!){\n  orderVoid(id: $id) {\n    errors {\n      code\n      field\n      message\n    }\n    order {\n      id\n      status\n      payments {\n        id\n      }\n      statusDisplay\n      events {\n        type\n      }\n    }\n  }\n}\n'

def raw_order_void(staff_api_client, order_id):
    if False:
        print('Hello World!')
    variables = {'id': order_id}
    raw_response = staff_api_client.post_graphql(ORDER_VOID_MUTATION, variables=variables)
    content = get_graphql_content(raw_response)
    raw_data = content['data']['orderVoid']
    return raw_data

def order_void(staff_api_client, order_id):
    if False:
        i = 10
        return i + 15
    response = raw_order_void(staff_api_client, order_id)
    assert response['errors'] == []
    return response