from ...utils import get_graphql_content
PRODUCT_TYPE_UPDATE_MUTATION = '\nmutation ProductTypeUpdate($id: ID!, $input: ProductTypeInput!) {\n  productTypeUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    productType {\n      id\n      name\n      slug\n      kind\n      isShippingRequired\n      isDigital\n      hasVariants\n      productAttributes {\n        id\n      }\n      assignedVariantAttributes {\n        attribute {\n          id\n        }\n        variantSelection\n      }\n      taxClass {\n        id\n        name\n      }\n    }\n    __typename\n  }\n}\n'

def update_product_type(staff_api_client, product_type_id, input):
    if False:
        return 10
    variables = {'id': product_type_id, 'input': input}
    response = staff_api_client.post_graphql(PRODUCT_TYPE_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['productTypeUpdate']['errors'] == []
    data = content['data']['productTypeUpdate']['productType']
    assert data['id'] is not None
    return data