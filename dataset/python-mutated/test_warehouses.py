import graphene
from .....warehouse.models import Warehouse
from ....tests.utils import assert_no_permission, get_graphql_content
QUERY_WAREHOUSES = '\nquery {\n    warehouses(first:100) {\n        totalCount\n        edges {\n            node {\n                id\n                name\n                companyName\n                email\n                shippingZones(first:100) {\n                    edges {\n                        node {\n                            name\n                            countries {\n                                country\n                            }\n                        }\n                    }\n                }\n                address {\n                    city\n                    postalCode\n                    country {\n                        country\n                    }\n                    phone\n                }\n            }\n        }\n    }\n}\n'

def test_query_warehouses_as_staff_with_manage_orders(staff_api_client, warehouse, permission_manage_orders):
    if False:
        print('Hello World!')
    response = staff_api_client.post_graphql(QUERY_WAREHOUSES, permissions=[permission_manage_orders])
    content = get_graphql_content(response)['data']
    assert content['warehouses']['totalCount'] == Warehouse.objects.count()
    warehouses = content['warehouses']['edges']
    warehouse_id = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    warehouse_first = warehouses[0]['node']
    assert warehouse_first['id'] == warehouse_id
    assert warehouse_first['name'] == warehouse.name
    assert len(warehouse_first['shippingZones']['edges']) == warehouse.shipping_zones.count()

def test_query_warehouses_as_staff_with_manage_shipping(staff_api_client, warehouse, permission_manage_shipping):
    if False:
        for i in range(10):
            print('nop')
    response = staff_api_client.post_graphql(QUERY_WAREHOUSES, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)['data']
    assert content['warehouses']['totalCount'] == Warehouse.objects.count()
    warehouses = content['warehouses']['edges']
    warehouse_id = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    warehouse_first = warehouses[0]['node']
    assert warehouse_first['id'] == warehouse_id
    assert warehouse_first['name'] == warehouse.name
    assert len(warehouse_first['shippingZones']['edges']) == warehouse.shipping_zones.count()

def test_query_warehouses_as_staff_with_manage_apps(staff_api_client, warehouse, permission_manage_apps):
    if False:
        return 10
    response = staff_api_client.post_graphql(QUERY_WAREHOUSES, permissions=[permission_manage_apps])
    assert_no_permission(response)

def test_query_warehouses_as_customer(user_api_client, warehouse, permission_manage_apps):
    if False:
        print('Hello World!')
    response = user_api_client.post_graphql(QUERY_WAREHOUSES)
    assert_no_permission(response)

def test_query_warehouses(staff_api_client, warehouse, permission_manage_products):
    if False:
        i = 10
        return i + 15
    response = staff_api_client.post_graphql(QUERY_WAREHOUSES, permissions=[permission_manage_products])
    content = get_graphql_content(response)['data']
    assert content['warehouses']['totalCount'] == Warehouse.objects.count()
    warehouses = content['warehouses']['edges']
    warehouse_id = graphene.Node.to_global_id('Warehouse', warehouse.pk)
    warehouse_first = warehouses[0]['node']
    assert warehouse_first['id'] == warehouse_id
    assert warehouse_first['name'] == warehouse.name
    assert len(warehouse_first['shippingZones']['edges']) == warehouse.shipping_zones.count()