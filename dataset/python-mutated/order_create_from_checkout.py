from saleor.graphql.tests.utils import get_graphql_content
ORDER_CREATE_FROM_CHECKOUT_MUTATION = '\nmutation orderCreateFromCheckout($id: ID!) {\n  orderCreateFromCheckout(id: $id) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n        id\n        created\n        status\n        paymentStatus\n        channel { id }\n        discounts {\n            amount {\n                amount\n            }\n        }\n        channel {\n            orderSettings {\n                expireOrdersAfter\n                deleteExpiredOrdersAfter\n            }\n        }\n        billingAddress {\n        streetAddress1\n        }\n        shippingAddress {\n            streetAddress1\n        }\n        shippingMethods {\n            id\n        }\n        lines {\n            productVariantId\n            quantity\n            undiscountedUnitPrice {\n                gross {\n                amount\n                }\n            }\n            unitPrice {\n                gross {\n                    amount\n                }\n            }\n            totalPrice {\n                gross {\n                    amount\n                }\n            }\n        }\n    }\n  }\n}\n'

def order_create_from_checkout(api_client, id):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': id}
    response = api_client.post_graphql(ORDER_CREATE_FROM_CHECKOUT_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['orderCreateFromCheckout']
    order_id = data['order']['id']
    errors = data['errors']
    assert errors == []
    assert order_id is not None
    return data