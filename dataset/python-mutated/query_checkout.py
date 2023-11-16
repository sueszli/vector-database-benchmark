from ...utils import get_graphql_content
CHECKOUT_QUERY = '\nquery Checkout($checkoutId: ID!){\n  checkout(id: $checkoutId){\n    id\n    voucherCode\n    discount {\n        amount\n      }\n    totalPrice{\n      gross{\n        amount\n      }\n      net{\n        amount\n      }\n      tax{\n        amount\n      }\n    }\n    availablePaymentGateways{\n      id\n      name\n    }\n    shippingMethods{\n      id\n      name\n      price{\n        amount\n      }\n    }\n  }\n}\n'

def get_checkout(api_client, checkout_id):
    if False:
        while True:
            i = 10
    variables = {'checkoutId': checkout_id}
    response = api_client.post_graphql(CHECKOUT_QUERY, variables)
    content = get_graphql_content(response)
    return content['data']['checkout']