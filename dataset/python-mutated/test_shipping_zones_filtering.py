import graphene
import pytest
from ....tests.utils import get_graphql_content
QUERY_SHIPPING_ZONES_WITH_FILTER = '\n    query ShippingZones($filter: ShippingZoneFilterInput) {\n        shippingZones(filter: $filter, first: 100) {\n            edges {\n                node {\n                    id\n                    name\n                }\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('lookup', 'expected_zones'), [('Poland', {'Poland'}), ('pol', {'Poland'}), ('USA', {'USA'}), ('us', {'USA'}), ('', {'Poland', 'USA'})])
def test_query_shipping_zone_search_by_name(staff_api_client, shipping_zones, permission_manage_shipping, lookup, expected_zones):
    if False:
        print('Hello World!')
    variables = {'filter': {'search': lookup}}
    response = staff_api_client.post_graphql(QUERY_SHIPPING_ZONES_WITH_FILTER, variables=variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingZones']['edges']
    assert len(data) == len(expected_zones)
    assert {zone['node']['name'] for zone in data} == expected_zones

def test_query_shipping_zone_search_by_channels(staff_api_client, shipping_zones, permission_manage_shipping, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    shipping_zone_usd = shipping_zones[0]
    shipping_zone_usd_id = graphene.Node.to_global_id('ShippingZone', shipping_zone_usd.id)
    variables = {'filter': {'channels': [channel_id]}}
    response = staff_api_client.post_graphql(QUERY_SHIPPING_ZONES_WITH_FILTER, variables=variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingZones']['edges']
    assert data[0]['node']['name'] == shipping_zone_usd.name
    assert data[0]['node']['id'] == shipping_zone_usd_id

def test_query_shipping_zone_search_by_channels_no_matter_of_input(staff_api_client, shipping_zones_with_different_channels, permission_manage_shipping, channel_USD, channel_PLN):
    if False:
        print('Hello World!')
    shipping_zones = shipping_zones_with_different_channels
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    shipping_zone_pln = shipping_zones[0]
    shipping_zone_pln_id = graphene.Node.to_global_id('ShippingZone', shipping_zone_pln.id)
    variables = {'filter': {'channels': [channel_id]}}
    response = staff_api_client.post_graphql(QUERY_SHIPPING_ZONES_WITH_FILTER, variables=variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingZones']['edges']
    assert len(data) == 1
    assert data[0]['node']['name'] == shipping_zone_pln.name
    assert data[0]['node']['id'] == shipping_zone_pln_id