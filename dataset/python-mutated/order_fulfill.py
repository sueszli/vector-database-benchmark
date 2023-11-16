from ...utils import get_graphql_content
ORDER_FULFILL_MUTATION = '\nmutation orderFulfill ($order: ID!, $input: OrderFulfillInput!) {\n  orderFulfill(order: $order, input: $input) {\n    order {\n      status\n      fulfillments {\n        id\n        status\n        created\n      }\n      id\n    }\n    fulfillments {\n      id\n      status\n    }\n    errors {\n      message\n      code\n      field\n    }\n  }\n}\n'

def order_fulfill(staff_api_client, id, input):
    if False:
        for i in range(10):
            print('nop')
    variables = {'order': id, 'input': input}
    response = staff_api_client.post_graphql(ORDER_FULFILL_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderFulfill']
    errors = data['errors']
    assert errors == []
    return data