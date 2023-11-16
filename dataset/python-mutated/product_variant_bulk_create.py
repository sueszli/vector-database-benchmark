from ...utils import get_graphql_content
PRODUCT_VARIANT_BULK_CREATE_MUTATION = '\nmutation ProductVariantBulkCreate($id: ID!, $input: [ProductVariantBulkCreateInput!]!) {\n  productVariantBulkCreate(product: $id, variants: $input) {\n    errors {\n      field\n      code\n      index\n      channels\n      message\n    }\n    productVariants {\n      id\n      name\n      attributes{\n        attribute{\n          id\n        }\n        values{\n          name\n        }\n      }\n      product{\n        id\n      }\n      channelListings{\n        channel{\n          id\n          slug\n        }\n        price{\n          amount\n          currency\n        }\n      }\n      stocks{\n        warehouse{\n          id\n        }\n        quantity\n      }\n    }\n  }\n}\n'

def create_variants_in_bulk(staff_api_client, product_id, variants_input):
    if False:
        print('Hello World!')
    variables = {'id': product_id, 'input': variants_input}
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['productVariantBulkCreate']['errors'] == []
    data = content['data']['productVariantBulkCreate']['productVariants']
    assert data[0]['id'] is not None
    return data