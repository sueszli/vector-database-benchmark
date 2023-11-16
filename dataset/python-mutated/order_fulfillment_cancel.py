from ...utils import get_graphql_content
ORDER_FULFILLMENT_CANCEL_MUTATION = '\nmutation OrderFulfillmentCancel($id: ID!, $input: FulfillmentCancelInput!) {\n  orderFulfillmentCancel(id: $id, input: $input) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n      id\n      status\n      fulfillments {\n        id\n        status\n      }\n    }\n  }\n}\n'

def order_fulfillment_cancel(staff_api_client, fulfillment_id, warehouse_id):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': fulfillment_id, 'input': {'warehouseId': warehouse_id}}
    response = staff_api_client.post_graphql(ORDER_FULFILLMENT_CANCEL_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderFulfillmentCancel']
    errors = data['errors']
    assert errors == []
    return data