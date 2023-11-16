from ...utils import get_graphql_content
VOUCHER_QUERY = '\nquery voucherQuery ($id:ID!){\n  voucher(id: $id) {\n    discountValue\n    discountValueType\n    id\n    name\n    onlyForStaff\n    singleUse\n    startDate\n    endDate\n    used\n    usageLimit\n    codes(first: 10) {\n      edges {\n        node {\n          code\n          id\n          isActive\n          used\n        }\n      }\n    }\n  }\n  checkouts(first: 10) {\n    edges {\n      node {\n        id\n        lines {\n          quantity\n          variant {\n            id\n          }\n        }\n      }\n    }\n  }\n}\n'

def get_voucher(api_client, voucher_id):
    if False:
        print('Hello World!')
    variables = {'id': voucher_id}
    response = api_client.post_graphql(VOUCHER_QUERY, variables)
    content = get_graphql_content(response)
    return content['data']