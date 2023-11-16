from ...utils import get_graphql_content
CHECKOUT_COMPLETE_MUTATION = '\nmutation CheckoutComplete($checkoutId: ID!) {\n  checkoutComplete(id: $checkoutId) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n      id\n      status\n      paymentStatus\n      isPaid\n      isShippingRequired\n      total {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      paymentStatus\n      statusDisplay\n      status\n      isPaid\n      subtotal {\n        gross {\n          amount\n        }\n      }\n      checkoutId\n      deliveryMethod {\n        ... on ShippingMethod {\n          id\n          price {\n            amount\n          }\n        }\n        ... on Warehouse {\n          id\n        }\n      }\n      shippingPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      lines {\n        id\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        unitDiscount {\n          amount\n        }\n        unitDiscountType\n        unitDiscountReason\n        unitDiscountValue\n        undiscountedUnitPrice {\n          gross {\n            amount\n          }\n        }\n      }\n      discounts {\n        type\n        value\n      }\n      voucher {\n        code\n      }\n      giftCards {\n        id\n        last4CodeChars\n      }\n    }\n  }\n}\n'

def raw_checkout_complete(api_client, checkout_id):
    if False:
        print('Hello World!')
    variables = {'checkoutId': checkout_id}
    response = api_client.post_graphql(CHECKOUT_COMPLETE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    raw_data = content['data']['checkoutComplete']
    return raw_data

def checkout_complete(api_client, checkout_id):
    if False:
        print('Hello World!')
    checkout_response = raw_checkout_complete(api_client, checkout_id)
    assert checkout_response['errors'] == []
    data = checkout_response['order']
    assert data['id'] is not None
    assert data['checkoutId'] == checkout_id
    return data