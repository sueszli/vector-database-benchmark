from ...utils import get_graphql_content
SHIPPING_PRICE_CREATE_MUTATION = '\nmutation CreateShippingRate($input: ShippingPriceInput!) {\n  shippingPriceCreate(input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    shippingZone {\n      id\n    }\n    shippingMethod {\n      id\n    }\n  }\n}\n'

def create_shipping_method(staff_api_client, shipping_zone_id, name='Test shipping method', type='PRICE'):
    if False:
        i = 10
        return i + 15
    variables = {'input': {'shippingZone': shipping_zone_id, 'name': name, 'type': type}}
    response = staff_api_client.post_graphql(SHIPPING_PRICE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['shippingPriceCreate']['errors'] == []
    data = content['data']['shippingPriceCreate']['shippingMethod']
    assert data['id'] is not None
    return data