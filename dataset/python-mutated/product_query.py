from ...utils import get_graphql_content
PRODUCT_QUERY = '\nquery Product($id: ID!, $channel: String) {\n  product(id: $id, channel: $channel) {\n    id\n    name\n    attributes {\n      attribute {\n        id\n      }\n      values {\n        name\n      }\n    }\n    pricing {\n      onSale\n    }\n    variants {\n      id\n      name\n      pricing {\n        onSale\n        discount {\n          gross {\n            amount\n          }\n        }\n        priceUndiscounted {\n          gross {\n            amount\n          }\n        }\n        price{\n          gross{\n            amount\n          }\n        }\n      }\n      attributes {\n        attribute {\n          id\n        }\n        values {\n          name\n        }\n      }\n      channelListings {\n        channel {\n          id\n        }\n        price {\n          amount\n          currency\n        }\n      }\n      stocks {\n        warehouse {\n          id\n        }\n        quantity\n      }\n    }\n  }\n}\n'

def get_product(staff_api_client, product_id, slug='default-channel'):
    if False:
        i = 10
        return i + 15
    variables = {'id': product_id, 'channel': slug}
    response = staff_api_client.post_graphql(PRODUCT_QUERY, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['product']
    assert data['id'] is not None
    return data