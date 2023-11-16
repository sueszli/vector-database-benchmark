from ...utils import get_graphql_content
CHECKOUT_DELIVERY_METHOD_UPDATE_MUTATION = '\nmutation checkoutDeliveryMethodUpdate($checkoutId: ID!, $deliveryMethodId: ID) {\n  checkoutDeliveryMethodUpdate(\n    id: $checkoutId\n    deliveryMethodId: $deliveryMethodId\n  ) {\n    errors {\n      field\n      code\n      message\n    }\n    checkout {\n      id\n      totalPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      subtotalPrice {\n        gross {\n          amount\n        }\n      }\n      shippingPrice{\n        gross {\n        amount\n        }\n      }\n      shippingMethods {\n        id\n      }\n      shippingPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      deliveryMethod {\n        ... on ShippingMethod {\n          id\n          price {\n            amount\n          }\n        }\n        ... on Warehouse {\n          id\n        }\n      }\n    }\n  }\n}\n'

def checkout_delivery_method_update(staff_api_client, checkout_id, delivery_method_id=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'checkoutId': checkout_id, 'deliveryMethodId': delivery_method_id}
    response = staff_api_client.post_graphql(CHECKOUT_DELIVERY_METHOD_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['checkoutDeliveryMethodUpdate']['errors'] == []
    data = content['data']['checkoutDeliveryMethodUpdate']['checkout']
    assert data['id'] is not None
    return data