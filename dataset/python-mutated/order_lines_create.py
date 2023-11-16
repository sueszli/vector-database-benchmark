from saleor.graphql.tests.utils import get_graphql_content
ORDER_LINES_CREATE_MUTATION = '\nmutation orderLinesCreate($id: ID!, $input: [OrderLineCreateInput!]!) {\n  orderLinesCreate(id: $id, input: $input) {\n    order {\n      id\n      shippingMethods {\n        id\n        price {\n          amount\n        }\n      }\n      total {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      isShippingRequired\n      lines {\n        id\n        quantity\n        variant {\n          id\n        }\n        totalPrice {\n          gross {\n            amount\n          }\n          net {\n            amount\n          }\n          tax {\n            amount\n          }\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        unitDiscountReason\n        undiscountedUnitPrice {\n          gross {\n            amount\n          }\n        }\n      }\n    }\n    errors {\n      code\n      field\n      message\n    }\n  }\n}\n'

def order_lines_create(api_client, order_id, input):
    if False:
        while True:
            i = 10
    variables = {'id': order_id, 'input': input}
    response = api_client.post_graphql(ORDER_LINES_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    return content['data']['orderLinesCreate']