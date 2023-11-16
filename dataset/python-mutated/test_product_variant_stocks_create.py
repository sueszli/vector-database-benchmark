import graphene
from .....warehouse.error_codes import StockErrorCode
from .....warehouse.models import Stock, Warehouse
from ....tests.utils import get_graphql_content
VARIANT_STOCKS_CREATE_MUTATION = '\n    mutation ProductVariantStocksCreate($variantId: ID!, $stocks: [StockInput!]!){\n        productVariantStocksCreate(variantId: $variantId, stocks: $stocks){\n            productVariant{\n                id\n                stocks {\n                    quantity\n                    quantityAllocated\n                    id\n                    warehouse{\n                        slug\n                    }\n                }\n            }\n            errors{\n                code\n                field\n                message\n                index\n            }\n        }\n    }\n'

def test_variant_stocks_create(staff_api_client, variant, warehouse, permission_manage_products):
    if False:
        return 10
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    second_warehouse = Warehouse.objects.get(pk=warehouse.pk)
    second_warehouse.slug = 'second warehouse'
    second_warehouse.pk = None
    second_warehouse.save()
    stocks = [{'warehouse': graphene.Node.to_global_id('Warehouse', warehouse.id), 'quantity': 20}, {'warehouse': graphene.Node.to_global_id('Warehouse', second_warehouse.id), 'quantity': 100}]
    variables = {'variantId': variant_id, 'stocks': stocks}
    response = staff_api_client.post_graphql(VARIANT_STOCKS_CREATE_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantStocksCreate']
    expected_result = [{'quantity': stocks[0]['quantity'], 'quantityAllocated': 0, 'warehouse': {'slug': warehouse.slug}}, {'quantity': stocks[1]['quantity'], 'quantityAllocated': 0, 'warehouse': {'slug': second_warehouse.slug}}]
    assert not data['errors']
    assert len(data['productVariant']['stocks']) == len(stocks)
    result = []
    for stock in data['productVariant']['stocks']:
        stock.pop('id')
        result.append(stock)
    for res in result:
        assert res in expected_result

def test_variant_stocks_create_empty_stock_input(staff_api_client, variant, permission_manage_products):
    if False:
        return 10
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    variables = {'variantId': variant_id, 'stocks': []}
    response = staff_api_client.post_graphql(VARIANT_STOCKS_CREATE_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantStocksCreate']
    assert not data['errors']
    assert len(data['productVariant']['stocks']) == variant.stocks.count()
    assert data['productVariant']['id'] == variant_id

def test_variant_stocks_create_stock_already_exists(staff_api_client, variant, warehouse, permission_manage_products):
    if False:
        return 10
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    second_warehouse = Warehouse.objects.get(pk=warehouse.pk)
    second_warehouse.slug = 'second warehouse'
    second_warehouse.pk = None
    second_warehouse.save()
    Stock.objects.create(product_variant=variant, warehouse=warehouse, quantity=10)
    stocks = [{'warehouse': graphene.Node.to_global_id('Warehouse', warehouse.id), 'quantity': 20}, {'warehouse': graphene.Node.to_global_id('Warehouse', second_warehouse.id), 'quantity': 100}]
    variables = {'variantId': variant_id, 'stocks': stocks}
    response = staff_api_client.post_graphql(VARIANT_STOCKS_CREATE_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantStocksCreate']
    errors = data['errors']
    assert errors
    assert errors[0]['code'] == StockErrorCode.UNIQUE.name
    assert errors[0]['field'] == 'warehouse'
    assert errors[0]['index'] == 0

def test_variant_stocks_create_stock_duplicated_warehouse(staff_api_client, variant, warehouse, permission_manage_products):
    if False:
        print('Hello World!')
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    second_warehouse = Warehouse.objects.get(pk=warehouse.pk)
    second_warehouse.slug = 'second warehouse'
    second_warehouse.pk = None
    second_warehouse.save()
    second_warehouse_id = graphene.Node.to_global_id('Warehouse', second_warehouse.id)
    stocks = [{'warehouse': graphene.Node.to_global_id('Warehouse', warehouse.id), 'quantity': 20}, {'warehouse': second_warehouse_id, 'quantity': 100}, {'warehouse': second_warehouse_id, 'quantity': 120}]
    variables = {'variantId': variant_id, 'stocks': stocks}
    response = staff_api_client.post_graphql(VARIANT_STOCKS_CREATE_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantStocksCreate']
    errors = data['errors']
    assert errors
    assert errors[0]['code'] == StockErrorCode.UNIQUE.name
    assert errors[0]['field'] == 'warehouse'
    assert errors[0]['index'] == 2

def test_variant_stocks_create_stock_duplicated_warehouse_and_warehouse_already_exists(staff_api_client, variant, warehouse, permission_manage_products):
    if False:
        i = 10
        return i + 15
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    second_warehouse = Warehouse.objects.get(pk=warehouse.pk)
    second_warehouse.slug = 'second warehouse'
    second_warehouse.pk = None
    second_warehouse.save()
    second_warehouse_id = graphene.Node.to_global_id('Warehouse', second_warehouse.id)
    Stock.objects.create(product_variant=variant, warehouse=second_warehouse, quantity=10)
    stocks = [{'warehouse': graphene.Node.to_global_id('Warehouse', warehouse.id), 'quantity': 20}, {'warehouse': second_warehouse_id, 'quantity': 100}, {'warehouse': second_warehouse_id, 'quantity': 120}]
    variables = {'variantId': variant_id, 'stocks': stocks}
    response = staff_api_client.post_graphql(VARIANT_STOCKS_CREATE_MUTATION, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    data = content['data']['productVariantStocksCreate']
    errors = data['errors']
    assert len(errors) == 3
    assert {error['code'] for error in errors} == {StockErrorCode.UNIQUE.name}
    assert {error['field'] for error in errors} == {'warehouse'}
    assert {error['index'] for error in errors} == {1, 2}
VARIANT_UPDATE_AND_STOCKS_CREATE_MUTATION = '\n  fragment ProductVariant on ProductVariant {\n    id\n    name\n    stocks {\n      quantity\n      warehouse {\n        id\n        name\n      }\n    }\n  }\n\n  mutation VariantUpdate($id: ID!, $stocks: [StockInput!]!) {\n    productVariantUpdate(id: $id, input: {}) {\n      productVariant {\n        ...ProductVariant\n      }\n    }\n    productVariantStocksCreate(variantId: $id, stocks: $stocks) {\n      productVariant {\n        ...ProductVariant\n      }\n    }\n  }\n'

def test_invalidate_stocks_dataloader_on_create_stocks(staff_api_client, variant_with_many_stocks, permission_manage_products):
    if False:
        while True:
            i = 10
    variant = variant_with_many_stocks
    variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    warehouse_ids = [graphene.Node.to_global_id('Warehouse', stock.warehouse.id) for stock in variant_with_many_stocks.stocks.all()]
    variant.stocks.all().delete()
    variables = {'id': variant_id, 'stocks': [{'warehouse': warehouse_id, 'quantity': 10} for warehouse_id in warehouse_ids]}
    response = staff_api_client.post_graphql(VARIANT_UPDATE_AND_STOCKS_CREATE_MUTATION, variables=variables, permissions=(permission_manage_products,))
    content = get_graphql_content(response)
    variant_data = content['data']['productVariantUpdate']['productVariant']
    create_stocks_data = content['data']['productVariantStocksCreate']['productVariant']
    assert variant_data['stocks'] == []
    assert len(create_stocks_data['stocks']) == len(warehouse_ids)