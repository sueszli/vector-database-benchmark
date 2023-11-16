from ...utils import get_graphql_content
CHECKOUT_LINES_ADD_MUTATION = '\nmutation checkoutLinesAdd($checkoutId: ID!, $lines: [CheckoutLineInput!]!) {\n  checkoutLinesAdd(id: $checkoutId, lines: $lines) {\n    checkout {\n      lines {\n        quantity\n        totalPrice {\n          gross {\n            amount\n          }\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        undiscountedUnitPrice {\n          amount\n        }\n      }\n      availablePaymentGateways {\n        id\n      }\n    }\n    errors {\n      field\n      lines\n      message\n      variants\n      code\n    }\n  }\n}\n'

def checkout_lines_add(staff_api_client, checkout_id, lines):
    if False:
        while True:
            i = 10
    variables = {'checkoutId': checkout_id, 'lines': lines}
    response = staff_api_client.post_graphql(CHECKOUT_LINES_ADD_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['checkoutLinesAdd']['errors'] == []
    return content['data']['checkoutLinesAdd']['checkout']