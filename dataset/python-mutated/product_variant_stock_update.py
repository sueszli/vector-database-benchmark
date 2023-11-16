from ...utils import get_graphql_content
PRODUCT_VARIANT_STOCK_UPDATE_MUTATION = '\nmutation productVariantStocksUpdate ($stocks: [StockInput!]!, $id: ID!) {\n  productVariantStocksUpdate(stocks: $stocks, variantId: $id) {\n    errors {\n      message\n      field\n    }\n    productVariant {\n      id\n      stocks {\n        quantity\n        warehouse {\n          id\n        }\n      }\n    }\n  }\n}\n'

def product_variant_stock_update(staff_api_client, warehouse_id, quantity, product_variant_id):
    if False:
        return 10
    variables = {'stocks': [{'quantity': quantity, 'warehouse': warehouse_id}], 'id': product_variant_id}
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_STOCK_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['productVariantStocksUpdate']['errors'] == []
    data = content['data']['productVariantStocksUpdate']['productVariant']
    return data