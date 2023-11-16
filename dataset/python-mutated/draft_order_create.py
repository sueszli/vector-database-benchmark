from saleor.graphql.tests.utils import get_graphql_content
DRAFT_ORDER_CREATE_MUTATION = '\nmutation OrderDraftCreate($input: DraftOrderCreateInput!) {\n  draftOrderCreate(input: $input) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n      id\n      created\n      discounts {\n        amount {\n          amount\n        }\n      }\n      billingAddress {\n        streetAddress1\n      }\n      shippingAddress {\n        streetAddress1\n      }\n      isShippingRequired\n      shippingMethods {\n        id\n      }\n      lines {\n        productVariantId\n        quantity\n        undiscountedUnitPrice {\n          gross {\n            amount\n          }\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        totalPrice {\n          gross {\n            amount\n          }\n        }\n      }\n    }\n  }\n}\n'

def draft_order_create(api_client, input):
    if False:
        while True:
            i = 10
    variables = {'input': input}
    response = api_client.post_graphql(DRAFT_ORDER_CREATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderCreate']
    order_id = data['order']['id']
    errors = data['errors']
    assert errors == []
    assert order_id is not None
    return data