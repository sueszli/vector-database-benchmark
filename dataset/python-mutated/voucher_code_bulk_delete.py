from ...utils import get_graphql_content
VOUCHER_CODE_BULK_DELETE_MUTATION = '\nmutation VoucherCodeBulkDelete ($ids: [ID!]!) {\n  voucherCodeBulkDelete(ids: $ids) {\n    errors {\n      message\n      code\n      voucherCodes\n    }\n  }\n}\n'

def voucher_code_bulk_delete(staff_api_client, voucher_code_ids):
    if False:
        for i in range(10):
            print('nop')
    variables = {'ids': voucher_code_ids}
    response = staff_api_client.post_graphql(VOUCHER_CODE_BULK_DELETE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']
    assert data['voucherCodeBulkDelete']['errors'] == []
    return data