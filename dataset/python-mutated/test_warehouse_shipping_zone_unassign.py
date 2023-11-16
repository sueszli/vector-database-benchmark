import graphene
MUTATION_UNASSIGN_SHIPPING_ZONE_WAREHOUSE = '\nmutation unassignWarehouseShippingZone($id: ID!, $shippingZoneIds: [ID!]!) {\n  unassignWarehouseShippingZone(id: $id, shippingZoneIds: $shippingZoneIds) {\n    errors {\n      field\n      message\n      code\n    }\n  }\n}\n\n'

def test_shipping_zone_unassign_from_warehouse(staff_api_client, warehouse, shipping_zone, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    assert warehouse.shipping_zones.first().pk == shipping_zone.pk
    staff_api_client.user.user_permissions.add(permission_manage_products)
    variables = {'id': graphene.Node.to_global_id('Warehouse', warehouse.pk), 'shippingZoneIds': [graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)]}
    staff_api_client.post_graphql(MUTATION_UNASSIGN_SHIPPING_ZONE_WAREHOUSE, variables=variables)
    warehouse.refresh_from_db()
    shipping_zone.refresh_from_db()
    assert not warehouse.shipping_zones.all()