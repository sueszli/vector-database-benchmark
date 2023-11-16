from saleor.graphql.tests.utils import get_graphql_content
DRAFT_ORDER_COMPLETE_MUTATION = '\nmutation DraftOrderComplete($id: ID!) {\n  draftOrderComplete(id: $id) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n      id\n      undiscountedTotal {\n        gross {\n          amount\n        }\n      }\n      totalBalance {\n        amount\n      }\n      total {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      subtotal {\n        gross {\n          amount\n        }\n      }\n      shippingPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      displayGrossPrices\n      status\n      paymentStatus\n      isPaid\n      channel {\n        orderSettings {\n          markAsPaidStrategy\n        }\n      }\n      lines {\n        productVariantId\n        quantity\n        unitDiscount {\n          amount\n        }\n        undiscountedUnitPrice {\n          gross {\n            amount\n          }\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        unitDiscountReason\n        unitDiscountType\n        unitDiscountValue\n      }\n    }\n  }\n}\n'

def raw_draft_order_complete(api_client, id):
    if False:
        print('Hello World!')
    variables = {'id': id}
    response = api_client.post_graphql(DRAFT_ORDER_COMPLETE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderComplete']
    return data

def draft_order_complete(api_client, id):
    if False:
        print('Hello World!')
    response = raw_draft_order_complete(api_client, id)
    assert response['order'] is not None
    errors = response['errors']
    assert errors == []
    return response