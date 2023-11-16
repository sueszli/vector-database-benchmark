from ...utils import get_graphql_content
SHIPPING_PRICE_UPDATE_MUTATION = '\nmutation ShippingPriceUpdate($id: ID!, $input: ShippingPriceInput!) {\n  shippingPriceUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    shippingMethod {\n      id\n      name\n      type\n      taxClass {\n        id\n      }\n      channelListings {\n        channel {\n          id\n          slug\n        }\n        price {\n          amount\n        }\n        maximumOrderPrice {\n          amount\n        }\n        minimumOrderPrice {\n          amount\n        }\n      }\n      maximumDeliveryDays\n      postalCodeRules {\n        id\n        start\n        end\n        inclusionType\n      }\n      excludedProducts(first: 10) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n    }\n  }\n}\n'

def update_shipping_price(staff_api_client, shipping_method_id, input):
    if False:
        return 10
    variables = {'id': shipping_method_id, 'input': input}
    response = staff_api_client.post_graphql(SHIPPING_PRICE_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['shippingPriceUpdate']['errors'] == []
    data = content['data']['shippingPriceUpdate']['shippingMethod']
    assert data['id'] is not None
    return data