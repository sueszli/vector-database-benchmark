from ...utils import get_graphql_content
VOUCHER_CREATE_MUTATION = '\nmutation VoucherCreate($input: VoucherInput!) {\n  voucherCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    voucher {\n      id\n      startDate\n      discountValueType\n      type\n      codes(first: 10) {\n        edges {\n          node {\n            id\n            code\n            isActive\n            used\n          }\n        }\n        totalCount\n      }\n    }\n  }\n}\n'

def create_voucher(staff_api_client, input):
    if False:
        return 10
    variables = {'input': input}
    response = staff_api_client.post_graphql(VOUCHER_CREATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['voucherCreate']['errors'] == []
    data = content['data']['voucherCreate']['voucher']
    assert data['id'] is not None
    return data