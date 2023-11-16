from ...utils import get_graphql_content
SALE_CATALOGUES_ADD_MUTATION = '\nmutation SaleCataloguesAdd($id: ID!, $input: CatalogueInput!, $first: Int) {\n  saleCataloguesAdd(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    sale {\n      id\n      name\n      channelListings {\n        channel {\n          id\n        }\n        discountValue\n      }\n      categories(first: $first) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n      collections(first: $first) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n      products(first: $first) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n      variants(first: $first) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n    }\n  }\n}\n'

def sale_catalogues_add(staff_api_client, sale_id, input, first=1):
    if False:
        print('Hello World!')
    variables = {'id': sale_id, 'first': first, 'input': input}
    response = staff_api_client.post_graphql(SALE_CATALOGUES_ADD_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['saleCataloguesAdd']['errors'] == []
    data = content['data']['saleCataloguesAdd']['sale']
    return data