from ...utils import get_graphql_content
PRODUCT_UPDATE_MUTATION = '\nmutation ProductUpdate($id: ID!, $input: ProductInput!) {\n  productUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    product {\n      id\n      name\n      productType {\n        id\n      }\n      category {\n        id\n      }\n      attributes {\n        attribute {\n          id\n        }\n        values {\n          name\n        }\n      }\n      collections {\n        id\n      }\n      taxClass {\n        id\n        name\n      }\n    }\n  }\n}\n'

def update_product(staff_api_client, product_id, input):
    if False:
        return 10
    variables = {'id': product_id, 'input': input}
    response = staff_api_client.post_graphql(PRODUCT_UPDATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['productUpdate']['errors'] == []
    data = content['data']['productUpdate']['product']
    assert data['id'] is not None
    return data