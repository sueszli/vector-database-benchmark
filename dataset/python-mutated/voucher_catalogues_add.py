from ...utils import get_graphql_content
VOUCHER_CATALOGUE_ADD_MUTATION = '\nmutation VoucherCataloguesAdd($id: ID!, $input: CatalogueInput!) {\n  voucherCataloguesAdd(id: $id, input: $input) {\n    errors {\n      message\n      field\n    }\n    voucher {\n      id\n      code\n      type\n      discountValueType\n      channelListings { id }\n      products(first: 10) {\n        edges {\n          node {\n            id\n          }\n        }\n      }\n    }\n  }\n}\n'

def add_catalogue_to_voucher(staff_api_client, voucher_id, include_categories=False, include_collections=False, include_products=False, products=None, first=20):
    if False:
        while True:
            i = 10
    variables = {'id': voucher_id, 'first': first, 'includeCategories': include_categories, 'includeCollections': include_collections, 'includeProducts': include_products, 'input': {'products': products}}
    response = staff_api_client.post_graphql(VOUCHER_CATALOGUE_ADD_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['voucherCataloguesAdd']['errors'] == []
    data = content['data']['voucherCataloguesAdd']['voucher']
    return data