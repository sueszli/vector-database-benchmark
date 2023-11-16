from saleor.graphql.tests.utils import get_graphql_content
DRAFT_ORDER_UPDATE_MUTATION = '\nmutation DraftOrderUpdate($input: DraftOrderInput!, $id: ID!) {\n  draftOrderUpdate(input: $input, id: $id) {\n    errors {\n      message\n      field\n      code\n    }\n    order {\n      id\n      lines {\n        totalPrice {\n          gross {\n            amount\n          }\n        }\n        unitPrice {\n          gross {\n            amount\n          }\n        }\n        unitDiscountReason\n      }\n      subtotal {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      totalBalance {\n        amount\n      }\n      total {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      billingAddress {\n        firstName\n        lastName\n        companyName\n        streetAddress1\n        streetAddress2\n        postalCode\n        country {\n          code\n        }\n        city\n        countryArea\n        phone\n      }\n      shippingAddress {\n        firstName\n        lastName\n        companyName\n        streetAddress1\n        streetAddress2\n        postalCode\n        country {\n          code\n        }\n        city\n        countryArea\n        phone\n      }\n      isShippingRequired\n      shippingPrice {\n        gross {\n          amount\n        }\n        net {\n          amount\n        }\n        tax {\n          amount\n        }\n      }\n      shippingMethod {\n        id\n      }\n      shippingMethods {\n        id\n      }\n      channel {\n        id\n        name\n      }\n      userEmail\n      deliveryMethod {\n        __typename\n        ... on ShippingMethod {\n          id\n          __typename\n        }\n      }\n    }\n  }\n}\n'

def draft_order_update(api_client, id, input):
    if False:
        while True:
            i = 10
    variables = {'id': id, 'input': input}
    response = api_client.post_graphql(DRAFT_ORDER_UPDATE_MUTATION, variables=variables)
    content = get_graphql_content(response)
    data = content['data']['draftOrderUpdate']
    order_id = data['order']['id']
    errors = data['errors']
    assert errors == []
    assert order_id == id
    return data