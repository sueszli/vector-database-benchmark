from ...utils import get_graphql_content
SHIPPING_ZONE_CREATE_MUTATION = '\nmutation createShipping($input: ShippingZoneCreateInput!) {\n  shippingZoneCreate(input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    shippingZone {\n      id\n      name\n      description\n      warehouses {\n        name\n      }\n      channels {\n        id\n      }\n    }\n  }\n}\n'

def create_shipping_zone(staff_api_client, name='Test shipping zone', countries=['US'], warehouse_ids=None, channel_ids=None):
    if False:
        for i in range(10):
            print('nop')
    if not warehouse_ids:
        warehouse_ids = []
    if not channel_ids:
        channel_ids = []
    variables = {'input': {'name': name, 'countries': countries, 'addWarehouses': warehouse_ids, 'addChannels': channel_ids}}
    response = staff_api_client.post_graphql(SHIPPING_ZONE_CREATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['shippingZoneCreate']['errors'] == []
    data = content['data']['shippingZoneCreate']['shippingZone']
    assert data['id'] is not None
    assert data['name'] == name
    return data