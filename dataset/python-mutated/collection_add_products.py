from ...utils import get_graphql_content
COLLECTION_ADD_PRODUCTS_MUTATION = '\nmutation CollectionAssignProduct($id: ID!, $productIds: [ID!]!) {\n  collectionAddProducts(collectionId: $id, products: $productIds) {\n    errors {\n      code\n      message\n      field\n    }\n    collection {\n      id\n      products(first: 5) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n    }\n  }\n}\n'

def add_product_to_collection(staff_api_client, collection_id, products_ids):
    if False:
        return 10
    variables = {'id': collection_id, 'productIds': products_ids}
    response = staff_api_client.post_graphql(COLLECTION_ADD_PRODUCTS_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['collectionAddProducts']['collection']
    assert content['data']['collectionAddProducts']['errors'] == []
    return data