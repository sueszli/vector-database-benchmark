from saleor.graphql.tests.utils import get_graphql_content
ORDER_QUERY = '\nquery OrderDetails($id: ID!) {\n  order(id: $id) {\n    availableShippingMethods {\n      id\n      active\n    }\n    paymentStatus\n    isPaid\n    payments {\n      id\n      gateway\n      paymentMethodType\n      chargeStatus\n      token\n    }\n    events {\n        type\n      }\n    channel {\n      id\n      name\n    }\n    updatedAt\n    fulfillments {\n      created\n      id\n    }\n    deliveryMethod {\n      ... on ShippingMethod {\n        id\n        name\n        active\n      }\n    }\n    shippingMethods {\n      id\n    }\n    shippingAddress {\n      country {\n        code\n      }\n      countryArea\n      firstName\n      cityArea\n      city\n      phone\n      postalCode\n      streetAddress1\n      streetAddress2\n    }\n    statusDisplay\n    status\n    transactions {\n      id\n    }\n  }\n}\n\n'

def order_query(api_client, order_id):
    if False:
        return 10
    variables = {'id': order_id}
    response = api_client.post_graphql(ORDER_QUERY, variables)
    content = get_graphql_content(response)
    return content['data']['order']