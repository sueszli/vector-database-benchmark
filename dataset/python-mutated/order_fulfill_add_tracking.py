from ...utils import get_graphql_content
ORDER_FULFILLMENT_UPDATE_TRACKING = '\nmutation OrderFulfillmentUpdateTracking(\n  $id: ID!\n  $input: FulfillmentUpdateTrackingInput!\n) {\n  orderFulfillmentUpdateTracking(id: $id, input: $input) {\n    errors {\n      message\n      code\n      field\n    }\n    order {\n      id\n      status\n      fulfillments {\n        id\n        status\n        trackingNumber\n      }\n    }\n  }\n}\n'

def order_add_tracking(staff_api_client, fulfillment_id, tracking_number):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': fulfillment_id, 'input': {'trackingNumber': tracking_number, 'notifyCustomer': True}}
    response = staff_api_client.post_graphql(ORDER_FULFILLMENT_UPDATE_TRACKING, variables)
    content = get_graphql_content(response)
    assert content['data']['orderFulfillmentUpdateTracking']['errors'] == []
    data = content['data']['orderFulfillmentUpdateTracking']['order']
    return data